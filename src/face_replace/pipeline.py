"""Safe orchestration and output finalization for the FaceFusion worker."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from face_replace.__main__ import InputError, probe


class PipelineError(RuntimeError):
    pass


def _run(command: list[str], timeout: int) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=True)
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
    with tempfile.TemporaryDirectory(prefix="face-replace-") as scratch_name:
        scratch = Path(scratch_name)
        os.chmod(scratch, 0o700)
        intermediate = scratch / "swapped.mp4"
        command = [
            "python",
            str(worker),
            "--video",
            str(video),
            "--face",
            str(face),
            "--output",
            str(intermediate),
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
                "-i",
                str(intermediate),
                "-i",
                str(video),
                "-map",
                "0:v:0",
                "-map",
                "1:a?",
                "-vf",
                "drawtext=text='AI-ALTERED':x=12:y=12:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.65",
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
            0.25, 1 / float(source["frame_rate"])
        ):
            raise PipelineError("generated output duration differs from source")
        if result["frame_rate"] != source["frame_rate"] or (
            source["audio_streams"] and not result["audio_streams"]
        ):
            raise PipelineError("generated output timing or audio validation failed")
        # Hard-link publication is atomic and fails if destination appeared meanwhile.
        try:
            os.link(candidate, output)
        except FileExistsError as exc:
            raise InputError("output appeared during processing; refusing to overwrite") from exc
        except OSError:
            # Same filesystem is expected (scratch uses system temp); copy via exclusive create.
            try:
                with candidate.open("rb") as src, output.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
            except FileExistsError as exc:
                raise InputError(
                    "output appeared during processing; refusing to overwrite"
                ) from exc
        return {"status": "created", "output": str(output), "media": result}
