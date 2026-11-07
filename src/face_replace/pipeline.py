"""Safe orchestration, resumable job state and output finalization for the FaceFusion worker."""

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path

from face_replace.__main__ import FACEFUSION_REVISION, InputError, probe

TARGET_HEIGHT = 720
JOB_VERSION = 1
SEGMENT_SECONDS = 300
# Upper bound for each worker run; a scan covers the whole movie.
WORKER_SECONDS_PER_VIDEO_SECOND = 30
# Peak use is the 720p copy, the segments, their joined copy and the labeled result. A film clip
# measured 1.45 MB/s at the segments' CRF 5 and 0.4 MB/s at CRF 18; this adds a 50% margin.
DISK_BYTES_PER_VIDEO_SECOND = 6_000_000
DISK_MINIMUM_BYTES = 2_000_000_000
INFO_PREFIX = "face-replace: "
ERROR_PREFIX = "face-replace-error: "
JOB_PREFIX = "face-replace-job: "
# Only fixed worker messages and upstream progress bars are shown: backend logs can hold paths.
PROGRESS_LINE = re.compile(r"^[\w ]+:\s+\d+%")
PROGRESS_INTERVAL_SECONDS = 30
GPU_CRASH_MESSAGE = "the GPU driver reported a CUDA error; start again to resume this job"


class PipelineError(RuntimeError):
    def __init__(self, message: str, refused: bool = False) -> None:
        super().__init__(message)
        self.refused = refused


def info(message: str) -> None:
    print(INFO_PREFIX + message, file=sys.stderr, flush=True)


def _kill_tree(process: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True, check=False
        )
    else:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def _environment() -> dict:
    env = {**os.environ, "PYTHONUTF8": "1"}
    if os.name != "nt":
        # On Linux the venv's CUDA and cuDNN libraries must be visible before the process starts.
        libraries = (Path(sys.prefix) / "lib").glob("python*/site-packages/nvidia/*/lib")
        paths = [str(path) for path in libraries]
        if env.get("LD_LIBRARY_PATH"):
            paths.append(env["LD_LIBRARY_PATH"])
        if paths:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(paths)
    return env


def _run(command: list[str], timeout: int) -> None:
    process = None
    errors: list[str] = []
    reader = None
    gpu_crash = threading.Event()

    def forward_stderr() -> None:
        last_progress = 0.0
        for raw in process.stderr:
            line = raw.strip()
            if line.startswith(ERROR_PREFIX):
                errors.append(line.removeprefix(ERROR_PREFIX))
            elif line.startswith(INFO_PREFIX):
                print(line, file=sys.stderr, flush=True)
            elif PROGRESS_LINE.match(line):
                if time.monotonic() - last_progress >= PROGRESS_INTERVAL_SECONDS:
                    last_progress = time.monotonic()
                    print(INFO_PREFIX + line, file=sys.stderr, flush=True)
            elif "CUDA" in line and ("error" in line.lower() or "failure" in line.lower()):
                gpu_crash.set()

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
            env=_environment(),
        )
        reader = threading.Thread(target=forward_stderr, daemon=True)
        reader.start()
        process.wait(timeout=timeout)
        reader.join()
        if process.returncode:
            # The worker exits with 2 for a final refusal and 4 for a problem that can resume.
            fallback = GPU_CRASH_MESSAGE if gpu_crash.is_set() else "media processing failed"
            message = errors[-1] if errors else fallback
            raise PipelineError(message, refused=bool(errors) and process.returncode == 2)
    except subprocess.TimeoutExpired as exc:
        _kill_tree(process)
        raise PipelineError("media processing timed out") from exc
    except KeyboardInterrupt:
        if process is not None:
            _kill_tree(process)
        raise
    except (subprocess.SubprocessError, OSError) as exc:
        raise PipelineError("media processing failed") from exc
    finally:
        if reader is not None:
            reader.join(timeout=5)
            process.stderr.close()


def _run_worker(command: list[str], timeout: int) -> None:
    """Retry once after a crash such as a GPU driver error; a refusal is final."""
    try:
        _run(command, timeout)
    except PipelineError as exc:
        if exc.refused:
            raise
        info("the worker stopped unexpectedly; retrying once")
        _run(command, timeout)


def _convert_to_720p(video: Path, destination: Path, timeout: int) -> None:
    """Normalize any readable movie to constant-rate H.264/AAC MP4, never upscaling."""
    scale = (
        "scale=w='trunc(oh*iw*sar/ih/2)*2'"
        f":h='trunc(min({TARGET_HEIGHT},ih)/2)*2':flags=lanczos,setsar=1,format=yuv420p"
    )
    _run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            scale,
            "-fps_mode",
            "cfr",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        timeout,
    )


def _label_filter() -> str:
    # Static or Windows FFmpeg builds may lack a default font; drawtext can then crash.
    font = ""
    if os.name == "nt":
        font = ":fontfile='C\\:/Windows/Fonts/arial.ttf'"
    elif Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf").is_file():
        font = ":fontfile='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'"
    return (
        f"drawtext=text='AI-ALTERED'{font}:x=12:y=12:fontsize='min(24,w/14)'"
        ":fontcolor=white:box=1:boxcolor=black@0.65"
    )


def job_directory(video: Path, face: Path, output: Path, worker: Path) -> Path:
    """Resume only for the same movie, photo and app code."""
    stat = video.stat()
    identity = f"{JOB_VERSION}|{FACEFUSION_REVISION}|{video}|{stat.st_size}|{stat.st_mtime_ns}|"
    code = worker.read_bytes() + Path(__file__).read_bytes()
    key = hashlib.sha256(identity.encode() + code + face.read_bytes()).hexdigest()[:16]
    return output.parent / f".face-replace-job-{key}"


def _check_disk(directory: Path, duration: float, job: Path) -> None:
    # Work already in the job folder counts towards what the whole job needs.
    used = sum(path.stat().st_size for path in job.rglob("*") if path.is_file())
    needed = max(DISK_MINIMUM_BYTES, int(duration * DISK_BYTES_PER_VIDEO_SECOND)) - used
    free = shutil.disk_usage(directory).free
    if free < needed:
        raise InputError(
            f"not enough free disk space: need about {needed / 1e9:.1f} GB, "
            f"{free / 1e9:.1f} GB available"
        )


def _timing_matches(result: dict, source: dict, frame: float) -> bool:
    """R3: duration within one frame, and the audio offset from video kept within one frame."""
    if abs(result["duration_seconds"] - source["duration_seconds"]) > frame:
        return False
    if not source["audio_streams"]:
        return True
    # A source with audio must have every start time, or the sync check fails closed.
    starts = ("audio_start", "video_start")
    if any(media.get(key) is None for media in (result, source) for key in starts):
        return False
    offset = result["audio_start"] - result["video_start"]
    return abs(offset - (source["audio_start"] - source["video_start"])) <= frame


def _probe_backend(path: Path, kind: str) -> dict:
    """Media the app made itself is a processing failure, not bad input, when unreadable."""
    try:
        return probe(path, kind)
    except InputError as exc:
        raise PipelineError(f"{kind} is not readable media") from exc


def _remove_job(job: Path) -> None:
    try:
        shutil.rmtree(job)
    except OSError:
        info(f"could not remove {job.name}; delete this folder by hand as it holds media")


def _lock_job(job: Path):
    """One run at a time per job folder; the system releases the lock even after a kill."""
    file = (job / "job.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        file.close()
        raise InputError("this job is already running") from exc
    return file


def _concat(segments: list[Path], destination: Path, timeout: int) -> None:
    # The concat demuxer resolves names against the listing's own folder.
    listing = segments[0].parent / "segments.txt"
    listing.write_text("".join(f"file '{s.name}'\n" for s in segments), encoding="utf-8")
    _run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-map",
            "0:v:0",
            "-c",
            "copy",
            str(destination),
        ],
        timeout,
    )


def run_pipeline(
    video: Path,
    face: Path,
    output: Path,
    worker: Path,
    timeout: int | None = None,
    character_time: float | None = None,
    character: int = 0,
    scan_only: bool = False,
    segment_seconds: int = SEGMENT_SECONDS,
    confirmed_job: str | None = None,
) -> dict:
    """Convert, scan, swap resumable segments, label, verify, then atomically publish."""
    video, face, output, worker = map(
        Path, (video.resolve(), face.resolve(), output.resolve(), worker.resolve())
    )
    if output.suffix.lower() != ".mp4" or output.exists() or output in (video, face):
        raise InputError("output must be a new MP4 path distinct from both inputs")
    if not output.parent.is_dir():
        raise InputError("output parent directory must already exist")
    raw_source = probe(video, "source video")
    probe(face, "reference image")
    duration = raw_source["duration_seconds"]
    if timeout is None:
        timeout = max(3600, int(duration * WORKER_SECONDS_PER_VIDEO_SECOND))
    job = job_directory(video, face, output, worker)
    # A character number means a person only within the scan that was confirmed.
    if confirmed_job is not None and job.name != confirmed_job:
        raise InputError("the movie or photo changed after the character was confirmed; scan again")
    _check_disk(output.parent, duration, job)
    job.mkdir(exist_ok=True)
    os.chmod(job, 0o700)
    lock = _lock_job(job)
    print(f"{JOB_PREFIX}{job.name}", file=sys.stderr, flush=True)
    try:
        result = _run_job(
            video,
            face,
            output,
            worker,
            job,
            timeout,
            character_time,
            character,
            scan_only,
            segment_seconds,
        )
    except PipelineError as exc:
        # A refusal is final, so its media must not stay behind; other failures can resume.
        if exc.refused:
            lock.close()
            _remove_job(job)
        raise
    finally:
        lock.close()
    if result["status"] == "created":
        _remove_job(job)
    return result


def _run_job(
    video: Path,
    face: Path,
    output: Path,
    worker: Path,
    job: Path,
    timeout: int,
    character_time: float | None,
    character: int,
    scan_only: bool,
    segment_seconds: int,
) -> dict:
    prepared = job / "source-720p.mp4"
    if not prepared.is_file():
        info("converting source to 720p MP4")
        partial = job / "converting.mp4"
        _convert_to_720p(video, partial, timeout)
        partial.replace(prepared)
    else:
        info("resuming: reusing the converted 720p copy")
    source = probe(prepared, "source video")

    base = [sys.executable, str(worker), "--video", str(prepared), "--face", str(face)]
    base += ["--work", str(job), "--upstream", str(worker.parent.parent / "upstream-facefusion")]

    state_path = job / "scan.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    if not state.get("done"):
        if state:
            info("resuming the scan from its last checkpoint")
        _run_worker([*base, "--mode", "scan", "--state", str(state_path)], timeout)
        state = json.loads(state_path.read_text(encoding="utf-8"))
    candidates = state["candidates"]
    if scan_only:
        keys = ("index", "time", "appearances", "preview")
        # Previews are named relative to the output folder, so no private path reaches a log.
        summary = [
            {**{key: c[key] for key in keys}, "preview": f"{job.name}/{Path(c['preview']).name}"}
            for c in candidates
        ]
        print(f"face-replace-candidates: {json.dumps(summary)}", file=sys.stderr, flush=True)
        return {"status": "scanned", "candidates": summary}

    fps = Fraction(source["frame_rate"])
    if character_time is not None:
        frame = round(character_time * float(fps))
        if frame >= source["video_frames"]:
            raise InputError("character time is past the end of the video")
        pick_path = job / f"pick-{frame}.json"
        if not pick_path.is_file():
            _run_worker(
                [*base, "--mode", "pick", "--state", str(pick_path)]
                + ["--reference-frame", str(frame)],
                timeout,
            )
        chosen = json.loads(pick_path.read_text(encoding="utf-8"))
    else:
        if not 0 <= character < len(candidates):
            raise InputError("that character number does not exist")
        chosen = candidates[character]

    segment_dir = job / f"swap-{chosen['frame']}-{chosen['position']}"
    segment_dir.mkdir(exist_ok=True)
    total = source["video_frames"]
    step = max(1, round(segment_seconds * float(fps)))
    ranges = [(start, min(start + step, total)) for start in range(0, total, step)]
    segments = []
    started = time.monotonic()
    processed = 0
    for number, (start, end) in enumerate(ranges, 1):
        segment = segment_dir / f"segment-{number:05}.mp4"
        segments.append(segment)
        if segment.is_file():
            continue
        partial = segment_dir / f"working-{number:05}.mp4"
        _run_worker(
            [
                *base,
                "--mode",
                "swap",
                "--output",
                str(partial),
                "--reference-frame",
                str(chosen["frame"]),
                "--reference-position",
                str(chosen["position"]),
                "--start-frame",
                str(start),
                "--end-frame",
                str(end),
            ],
            timeout,
        )
        if not partial.is_file():
            raise PipelineError("backend produced no output")
        if _probe_backend(partial, "swapped segment")["video_frames"] != end - start:
            raise PipelineError("swapped segment frame count differs from source")
        partial.replace(segment)
        processed += 1
        left = (time.monotonic() - started) / processed * (len(ranges) - number)
        hours, seconds = divmod(int(left), 3600)
        left_text = f"{hours:02d}:{seconds // 60:02d}:{seconds % 60:02d}"
        percent = number / len(ranges)
        info(f"swapped segment {number}/{len(ranges)}, {percent:.1%}, {left_text} left")

    intermediate = job / "swapped.mp4"
    _concat(segments, intermediate, timeout)
    candidate = job / "labeled.mp4"
    info("adding the AI-ALTERED label and verifying the result")
    # Keep the backend's video and the prepared AAC audio from the source.
    _run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-i",
            str(intermediate),
            "-protocol_whitelist",
            "file,pipe",
            "-i",
            str(prepared),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-vf",
            _label_filter(),
            "-metadata",
            "comment=AI-ALTERED face replacement",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(candidate),
        ],
        timeout,
    )
    result = _probe_backend(candidate, "generated output")
    if not _timing_matches(result, source, float(1 / fps)):
        raise PipelineError("generated output duration or audio sync differs from source")
    if result["frame_rate"] != source["frame_rate"]:
        raise PipelineError("generated output timing or audio validation failed")
    if result["audio_streams"] != source["audio_streams"]:
        raise PipelineError("generated output audio stream count differs from source")
    if result["video_frames"] != source["video_frames"]:
        raise PipelineError("generated output frame count differs from source")
    if (result["width"], result["height"]) != (source["width"], source["height"]):
        raise PipelineError("generated output dimensions differ from source")
    _run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-protocol_whitelist",
            "file,pipe",
            "-i",
            str(candidate),
            "-map",
            "0",
            "-f",
            "null",
            "-",
        ],
        timeout,
    )
    # Hard-link publication is atomic and fails if destination appeared meanwhile.
    try:
        os.link(candidate, output)
    except FileExistsError as exc:
        raise InputError("output appeared during processing; refusing to overwrite") from exc
    except OSError as exc:
        raise PipelineError("cannot atomically publish output on destination filesystem") from exc
    # The preview shows a movie frame, so it is published only with a finished result.
    preview = output.with_name(f"{output.stem}-character.png")
    try:
        data = Path(chosen["preview"]).read_bytes()
        with preview.open("xb") as file:
            file.write(data)
    except FileExistsError:
        info("kept the existing character preview")
    except OSError:
        # The verified MP4 is already published; a missing preview must not fail the job.
        preview.unlink(missing_ok=True)
        info("the movie is done, but the character preview could not be saved")
    return {"status": "created", "output": str(output), "media": result}
