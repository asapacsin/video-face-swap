"""Inspect local media without importing an inference backend or downloading models."""

import argparse
import json
import math
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

from face_replace import FACEFUSION_REVISION


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
                "-show_format",
                "-of",
                "json",
                str(path.resolve()),
            ],
            capture_output=True,
            text=True,
            timeout=30,
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
    return {
        "width": video["width"],
        "height": video["height"],
        "frame_rate": str(fps),
        "duration_seconds": duration,
        "audio_streams": sum(s.get("codec_type") == "audio" for s in streams),
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare a consensual adult, non-explicit video job.")
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--face", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
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
    p.add_argument("--dry-run", action="store_true", help="Inspect only; no inference or output")
    p.add_argument("--worker", type=Path, help="Pinned FaceFusion worker script for GPU execution")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if not args.consent_confirmed:
            raise InputError("affirmative --consent-confirmed attestation is required")
        if args.content_rating == "explicit":
            raise InputError("known explicit content is refused")
        output = args.output.resolve()
        if output in {args.video.resolve(), args.face.resolve()}:
            raise InputError("output must not overwrite either input")
        if output.suffix.lower() != ".mp4":
            raise InputError("output must end in .mp4")
        if output.exists():
            raise InputError("output already exists; choose a new path")
        if not output.parent.is_dir():
            raise InputError("output parent directory must already exist")
        print("Inspecting local media; inference is disabled.", file=sys.stderr)
        source = probe(args.video, "source video")
        reference = probe(args.face, "reference image")
        if not args.dry_run and not args.worker:
            print("inference requires --worker; no output written", file=sys.stderr)
            return 3
        if not args.dry_run:
            from face_replace.pipeline import PipelineError, run_pipeline

            try:
                result = run_pipeline(args.video, args.face, output, args.worker)
            except (InputError, PipelineError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 3
            print(json.dumps(result, indent=2))
            return 0
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "backend": "FaceFusion",
                    "revision": FACEFUSION_REVISION,
                    "execution_target": "rented Linux NVIDIA GPU (not contacted)",
                    "source": source,
                    "reference": reference,
                    "safety_screening": "NOT_RUN; dry-run is not content approval",
                    "planned_output": {
                        "container": "mp4",
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
