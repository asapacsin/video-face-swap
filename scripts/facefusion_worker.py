#!/usr/bin/env python3
"""Run pinned FaceFusion in-process with downloads disabled and fail-closed screening."""

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--face", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--upstream", required=True)
    args = parser.parse_args()

    from pathlib import Path

    upstream = Path(args.upstream).resolve()
    sys.path.insert(0, str(upstream))
    import os

    os.chdir(upstream)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=upstream, capture_output=True, text=True, check=True
    ).stdout.strip()
    if revision != "72470819a0373be3388b3929c8f8f311f418fc3c":
        print("pinned FaceFusion revision mismatch", file=sys.stderr)
        return 2
    if (
        subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--"], cwd=upstream, check=False
        ).returncode
        != 0
    ):
        print("FaceFusion checkout has modified tracked files", file=sys.stderr)
        return 2

    from facefusion import content_analyser, core, state_manager
    from facefusion.args import apply_args
    from facefusion.face_creator import get_many_faces
    from facefusion.program import create_program
    from facefusion.video_manager import close_video_reader, get_reader, read_video_frame
    from facefusion.vision import count_video_frame_total

    # Parse only a fixed headless command; no user-supplied upstream flags.
    sys.argv = [
        "facefusion",
        "headless-run",
        "--source-paths",
        args.face,
        "--target-path",
        args.video,
        "--output-path",
        args.output,
        "--processors",
        "face_swapper",
        "--face-swapper-model",
        "hyperswap_1a_256",
        "--face-selector-mode",
        "one",
        "--execution-providers",
        "cuda",
        "--output-video-encoder",
        "libx264",
        "--log-level",
        "error",
        "--jobs-path",
        str(Path(args.output).parent / "jobs"),
        "--temp-path",
        str(Path(args.output).parent / "temp"),
    ]
    program = create_program()
    parsed = vars(program.parse_args())
    parsed["download_providers"] = []
    apply_args(parsed, state_manager.init_item)
    if not core.common_pre_check() or not core.processors_pre_check():
        print("safety/model precheck failed", file=sys.stderr)
        return 2
    from facefusion.jobs import job_manager

    if not job_manager.init_jobs(state_manager.get_item("jobs_path")):
        return 2
    # Upstream prechecks the target video but not the reference image.
    if content_analyser.analyse_image(args.face):
        print("reference image screening refused", file=sys.stderr)
        return 2
    from facefusion.vision import read_image

    reference_frame = read_image(args.face)
    if reference_frame is None or len(get_many_faces([reference_frame])) != 1:
        print("reference must contain exactly one detectable face", file=sys.stderr)
        return 2
    # Screen every decodable target frame with upstream's unchanged classifier.
    total = count_video_frame_total(args.video)
    if total <= 0:
        print("video has no frames", file=sys.stderr)
        return 2
    reader = get_reader(args.video, "face_replace_screen")
    detected_target_face = False
    try:
        for _ in range(total):
            frame = read_video_frame(reader)
            if frame is None or content_analyser.analyse_frame(frame):
                print("video screening failed or refused", file=sys.stderr)
                return 2
            face_total = len(get_many_faces([frame]))
            if face_total > 1:
                print("ambiguous multiple faces in target video", file=sys.stderr)
                return 2
            detected_target_face = detected_target_face or face_total == 1
        if read_video_frame(reader) is not None:
            print("video frame count changed during screening", file=sys.stderr)
            return 2
    finally:
        close_video_reader(reader)
    if not detected_target_face:
        print("no target face detected", file=sys.stderr)
        return 2
    # Normal headless workflow keeps its own standard safety checks as well.
    return int(core.process_headless(parsed))


if __name__ == "__main__":
    raise SystemExit(main())
