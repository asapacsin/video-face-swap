"""Safe orchestration and output finalization for the FaceFusion worker."""

import os
import signal
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

from face_replace.__main__ import InputError, probe


class PipelineError(RuntimeError):
    pass


def _run(command: list[str], timeout: int) -> subprocess.CompletedProcess:
    process = None
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        stdout, stderr = process.communicate(timeout=timeout)
        if process.returncode:
            raise PipelineError("media processing failed")
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    except subprocess.TimeoutExpired as exc:
        if process is not None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        raise PipelineError("media processing timed out") from exc
    except KeyboardInterrupt:
        if process is not None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        raise
    except (subprocess.SubprocessError, OSError) as exc:
        raise PipelineError("media processing failed") from exc


def run_pipeline(video: Path, face: Path, output: Path, worker: Path, timeout: int = 3600) -> dict:
    """Run worker in private scratch space, label/remux, verify, then atomically publish."""
    video, face, output, worker = map(
        Path, (video.resolve(), face.resolve(), output.resolve(), worker.resolve())
    )
    if output.suffix.lower() != ".mp4" or output.exists() or output in (video, face):
        raise InputError("output must be a new MP4 path distinct from both inputs")
    if not output.parent.is_dir():
        raise InputError("output parent directory must already exist")
    source = probe(video, "source video")
    probe(face, "reference image")
    with tempfile.TemporaryDirectory(prefix=".face-replace-", dir=output.parent) as scratch_name:
        scratch = Path(scratch_name)
        os.chmod(scratch, 0o700)
        intermediate = scratch / "swapped.mp4"
        command = [
            sys.executable,
            str(worker),
            "--video",
            str(video),
            "--face",
            str(face),
            "--output",
            str(intermediate),
            "--upstream",
            str(worker.parent.parent / "upstream-facefusion"),
        ]
        _run(command, timeout)
        if not intermediate.is_file():
            raise PipelineError("backend produced no output")
        candidate = scratch / "labeled.mp4"
        # Keep the backend's video and source audio; transcode audio to AAC for MP4.
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
                str(video),
                "-map",
                "0:v:0",
                "-map",
                "1:a?",
                "-vf",
                "drawtext=text='AI-ALTERED':x=12:y=12:fontsize='min(24,w/14)':fontcolor=white:box=1:boxcolor=black@0.65",
                "-metadata",
                "comment=AI-ALTERED face replacement",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
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
