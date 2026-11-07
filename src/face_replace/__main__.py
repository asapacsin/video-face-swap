"""Inspect local media without importing an inference backend or downloading models."""

import argparse
import json
import math
import shutil
import signal
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

from face_replace import FACEFUSION_REVISION

# Long enough to read through a full-length movie file on a slow disk.
PROBE_TIMEOUT_SECONDS = 600


class InputError(ValueError):
    pass


def probe(path: Path, kind: str) -> dict:
    if not path.is_file():
        raise InputError(f"{kind} must be an existing local regular file")
    if not shutil.which("ffprobe"):
        raise InputError("ffprobe is required; install FFmpeg before continuing")
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-show_streams",
                "-count_packets",
                "-show_format",
                "-of",
                "json",
                str(path.resolve()),
            ],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=PROBE_TIMEOUT_SECONDS,
            check=True,
        )
        data = json.loads(result.stdout)
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        # Backend stderr can contain personal paths or embedded metadata.
        raise InputError(f"{kind} could not be inspected as readable media") from exc
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video or video.get("width", 0) <= 0 or video.get("height", 0) <= 0:
        raise InputError(f"{kind} has no usable visual stream")
    if kind == "reference image":
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise InputError("reference image must be JPEG, PNG or WebP")
        if video.get("codec_name") not in {"mjpeg", "png", "webp"}:
            raise InputError("reference image must contain a supported still image")
        return {"width": video["width"], "height": video["height"]}
    try:
        duration = float(data.get("format", {}).get("duration", 0))
        fps = Fraction(video.get("avg_frame_rate", "0/1"))
        if not math.isfinite(duration) or duration <= 0 or fps <= 0:
            raise ValueError
    except (ValueError, ZeroDivisionError) as exc:
        raise InputError("source video must have positive duration and frame rate") from exc
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    return {
        "width": video["width"],
        "height": video["height"],
        "frame_rate": str(fps),
        "duration_seconds": duration,
        "audio_streams": sum(s.get("codec_type") == "audio" for s in streams),
        "video_frames": int(video.get("nb_read_packets", 0)),
        "video_start": _start_time(video),
        "audio_start": _start_time(audio),
    }


def _start_time(stream: dict | None) -> float | None:
    """Stream start in seconds, or None when the stream or its start time is absent."""
    try:
        value = float(stream["start_time"])
    except (TypeError, KeyError, ValueError):
        return None
    return value if math.isfinite(value) else None


def clock_seconds(value: str) -> float:
    """Accept seconds or [HH:]MM:SS for picking the target character."""
    try:
        seconds = 0.0
        for part in value.split(":"):
            seconds = seconds * 60 + float(part)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use seconds or HH:MM:SS") from exc
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError("use seconds or HH:MM:SS")
    return seconds


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare a consensual adult, non-explicit video job.")
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--face", required=True, type=Path)
    p.add_argument(
        "--output", type=Path, help="New .mp4 path; defaults to <video>-swapped-720p.mp4 beside it"
    )
    p.add_argument(
        "--character-time",
        type=clock_seconds,
        help="Replace the largest face at this movie time (seconds or HH:MM:SS); "
        "defaults to the face seen most often",
    )
    p.add_argument(
        "--consent-confirmed",
        action="store_true",
        help="I attest every depicted person whose identity is used is an adult and consents; "
        "media is non-explicit, authorized, and not for impersonation, deception or harassment.",
    )
    p.add_argument(
        "--content-rating",
        choices=["unknown", "non-explicit", "explicit"],
        default="unknown",
        help="Operator assessment; never substitutes for safety checks",
    )
    p.add_argument(
        "--scan-only",
        action="store_true",
        help="Convert and screen the movie, list its characters, then stop before swapping",
    )
    p.add_argument(
        "--character",
        type=int,
        default=0,
        help="Number of the scanned character to replace; 0 is the face seen most often",
    )
    p.add_argument(
        "--confirmed-job",
        help="Job the character number was confirmed in; the run stops if the inputs changed since",
    )
    p.add_argument("--dry-run", action="store_true", help="Inspect only; no inference or output")
    return p


def stop_workers_on_signals() -> None:
    """The worker has its own session, so a stopped CLI must end it rather than leave it billing."""

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    for name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), interrupt)


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if not args.consent_confirmed:
            raise InputError("affirmative --consent-confirmed attestation is required")
        if args.content_rating == "explicit":
            raise InputError("known explicit content is refused")
        default_output = args.video.with_name(f"{args.video.stem}-swapped-720p.mp4")
        output = (args.output or default_output).resolve()
        if output in {args.video.resolve(), args.face.resolve()}:
            raise InputError("output must not overwrite either input")
        if output.suffix.lower() != ".mp4":
            raise InputError("output must end in .mp4")
        if output.exists():
            raise InputError("output already exists; choose a new path")
        if not output.parent.is_dir():
            raise InputError("output parent directory must already exist")
        print(
            "Inspecting local media; inference is disabled."
            if args.dry_run
            else "Validating media and starting the pinned FaceFusion worker.",
            file=sys.stderr,
        )
        source = probe(args.video, "source video")
        reference = probe(args.face, "reference image")
        if not args.dry_run:
            from face_replace.pipeline import PipelineError, run_pipeline

            stop_workers_on_signals()
            try:
                project_root = Path(__file__).resolve().parents[2]
                worker = project_root / "scripts" / "facefusion_worker.py"
                result = run_pipeline(
                    args.video,
                    args.face,
                    output,
                    worker,
                    character_time=args.character_time,
                    character=args.character,
                    scan_only=args.scan_only,
                    confirmed_job=args.confirmed_job,
                )
            except PipelineError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 3
            except OSError:
                # The exception's text can hold private paths, so it is not shown.
                print(
                    "error: could not read or write the job's files; "
                    "check the output folder is writable and the disk has space",
                    file=sys.stderr,
                )
                return 3
            except KeyboardInterrupt:
                print("error: stopped; run the same command again to resume", file=sys.stderr)
                return 3
            print(json.dumps(result, indent=2))
            return 0
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "backend": "FaceFusion",
                    "revision": FACEFUSION_REVISION,
                    "execution_target": "local or rented NVIDIA GPU (not contacted)",
                    "source": source,
                    "reference": reference,
                    "safety_screening": "NOT_RUN; dry-run is not content approval",
                    "planned_output": {
                        "container": "mp4",
                        "max_height": 720,
                        "visible_label": "AI-ALTERED",
                        "audio": "preserve source audio",
                        "metadata": "AI-altered",
                    },
                    "output_written": False,
                    "gpu_inference": "NOT_VERIFIED",
                },
                indent=2,
            )
        )
        return 0
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
