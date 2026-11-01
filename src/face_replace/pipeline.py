"""Safe orchestration and output finalization for the FaceFusion worker."""

import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from fractions import Fraction
from pathlib import Path

from face_replace.__main__ import InputError, probe

TARGET_HEIGHT = 720
# Upper bound for a whole job, measured against the 720p throughput in docs/BENCHMARKS.md.
WORKER_SECONDS_PER_VIDEO_SECOND = 30
INFO_PREFIX = "face-replace: "
ERROR_PREFIX = "face-replace-error: "
# Only fixed worker messages and upstream progress bars are shown: backend logs can hold paths.
PROGRESS_LINE = re.compile(r"^[\w ]+:\s+\d+%")
PROGRESS_INTERVAL_SECONDS = 30


class PipelineError(RuntimeError):
    pass


def _kill_tree(process: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True, check=False
        )
    else:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def _run(command: list[str], timeout: int) -> None:
    process = None
    errors: list[str] = []
    reader = None

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

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        reader = threading.Thread(target=forward_stderr, daemon=True)
        reader.start()
        process.wait(timeout=timeout)
        reader.join()
        if process.returncode:
            raise PipelineError(errors[-1] if errors else "media processing failed")
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
    # Windows FFmpeg builds lack a default fontconfig; without a font file drawtext can crash.
    font = ":fontfile='C\\:/Windows/Fonts/arial.ttf'" if os.name == "nt" else ""
    return (
        f"drawtext=text='AI-ALTERED'{font}:x=12:y=12:fontsize='min(24,w/14)'"
        ":fontcolor=white:box=1:boxcolor=black@0.65"
    )


def run_pipeline(
    video: Path,
    face: Path,
    output: Path,
    worker: Path,
    timeout: int | None = None,
    character_time: float | None = None,
) -> dict:
    """Convert to 720p, run worker in private scratch, label, verify, then atomically publish."""
    video, face, output, worker = map(
        Path, (video.resolve(), face.resolve(), output.resolve(), worker.resolve())
    )
    if output.suffix.lower() != ".mp4" or output.exists() or output in (video, face):
        raise InputError("output must be a new MP4 path distinct from both inputs")
    if not output.parent.is_dir():
        raise InputError("output parent directory must already exist")
    raw_source = probe(video, "source video")
    probe(face, "reference image")
    if timeout is None:
        timeout = max(3600, int(raw_source["duration_seconds"] * WORKER_SECONDS_PER_VIDEO_SECOND))
    with tempfile.TemporaryDirectory(prefix=".face-replace-", dir=output.parent) as scratch_name:
        scratch = Path(scratch_name)
        os.chmod(scratch, 0o700)
        prepared = scratch / "source-720p.mp4"
        print(f"{INFO_PREFIX}converting source to 720p MP4", file=sys.stderr, flush=True)
        _convert_to_720p(video, prepared, timeout)
        source = probe(prepared, "source video")
        intermediate = scratch / "swapped.mp4"
        command = [
            sys.executable,
            str(worker),
            "--video",
            str(prepared),
            "--face",
            str(face),
            "--output",
            str(intermediate),
            "--upstream",
            str(worker.parent.parent / "upstream-facefusion"),
            "--preview",
            str(output.with_name(f"{output.stem}-character.jpg")),
        ]
        if character_time is not None:
            command += ["--character-time", str(character_time)]
        _run(command, timeout)
        if not intermediate.is_file():
            raise PipelineError("backend produced no output")
        candidate = scratch / "labeled.mp4"
        # Keep the backend's video and the prepared AAC audio from the source.
        _run(
            [
                "ffmpeg",
                "-nostdin",
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
        result = probe(candidate, "generated output")
        if abs(result["duration_seconds"] - source["duration_seconds"]) > max(
            0.25, float(1 / Fraction(source["frame_rate"]))
        ):
            raise PipelineError("generated output duration differs from source")
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
            raise PipelineError(
                "cannot atomically publish output on destination filesystem"
            ) from exc
        return {"status": "created", "output": str(output), "media": result}
