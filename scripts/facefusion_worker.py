#!/usr/bin/env python3
"""Run pinned FaceFusion in-process with downloads disabled and fail-closed screening."""

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--face", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from facefusion import content_analyser, core, state_manager
    from facefusion.args import apply_args
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
    ]
    program = create_program()
    parsed = vars(program.parse_args())
    parsed["download_providers"] = []
    apply_args(parsed, state_manager.init_item)
    if not core.common_pre_check() or not core.processors_pre_check():
        print("safety/model precheck failed", file=sys.stderr)
        return 2
    # Upstream prechecks the target video but not the reference image.
    if content_analyser.analyse_image(args.face):
        print("reference image screening refused", file=sys.stderr)
        return 2
    # Screen every decodable target frame with upstream's unchanged classifier.
    total = count_video_frame_total(args.video)
    if total <= 0:
        print("video has no frames", file=sys.stderr)
        return 2
    reader = get_reader(args.video, "face_replace_screen")
    try:
        for _ in range(total):
            frame = read_video_frame(reader)
            if frame is None or content_analyser.analyse_frame(frame):
                print("video screening failed or refused", file=sys.stderr)
                return 2
    finally:
        close_video_reader(reader)
    # Normal headless workflow keeps its own standard safety checks as well.
    return int(core.process_headless(parsed))


if __name__ == "__main__":
    raise SystemExit(main())
