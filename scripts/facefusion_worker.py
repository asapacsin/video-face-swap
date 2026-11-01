#!/usr/bin/env python3
"""Run pinned FaceFusion in-process with downloads disabled and fail-closed screening."""

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REVISION = "72470819a0373be3388b3929c8f8f311f418fc3c"
EXECUTION_THREADS = "4"
SCREENING_THREADS = 4
# Identity sampling is expensive and a character stays on screen; screening is still every frame.
FACE_SAMPLE_SECONDS = 2
# Same scale and default as upstream's reference-face matching.
FACE_DISTANCE = 0.3


def info(message: str) -> None:
    print(f"face-replace: {message}", file=sys.stderr, flush=True)


def refuse(message: str) -> int:
    print(f"face-replace-error: {message}", file=sys.stderr, flush=True)
    return 2


def clock(seconds: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def gpu_convolution_works(onnxruntime, numpy, upstream: Path) -> bool:
    """FaceFusion lets ONNX Runtime drop to CPU silently, turning a movie into a weeks-long job."""
    model = upstream / ".assets" / "models" / "yoloface_8n.onnx"
    if not model.is_file():
        return True  # The model precheck reports missing weights with a clearer message.
    options = onnxruntime.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    try:
        session = onnxruntime.InferenceSession(
            str(model), sess_options=options, providers=["CUDAExecutionProvider"]
        )
        session.disable_fallback()
        probe = numpy.zeros((1, 3, 640, 640), numpy.float32)
        session.run(None, {session.get_inputs()[0].name: probe})
        return True
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--face", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--preview", required=True)
    parser.add_argument("--character-time", type=float)
    args = parser.parse_args()

    upstream = Path(args.upstream).resolve()
    sys.path.insert(0, str(upstream))
    os.chdir(upstream)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=upstream, capture_output=True, text=True, check=True
    ).stdout.strip()
    if revision != REVISION:
        return refuse("pinned FaceFusion revision mismatch")
    if (
        subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--"], cwd=upstream, check=False
        ).returncode
        != 0
    ):
        return refuse("FaceFusion checkout has modified tracked files")

    if os.name == "nt":
        # cuDNN loads its sub-libraries lazily by name, so preloading alone is not enough.
        library_bins = (Path(sys.prefix) / "Lib" / "site-packages" / "nvidia").glob("*/bin")
        for library_bin in library_bins:
            os.environ["PATH"] = f"{library_bin}{os.pathsep}{os.environ['PATH']}"
            os.add_dll_directory(str(library_bin))

    import numpy
    import onnxruntime

    if os.name == "nt":
        onnxruntime.preload_dlls(directory="")
    if not gpu_convolution_works(onnxruntime, numpy, upstream):
        return refuse("GPU convolution failed; refusing to fall back to CPU")

    import cv2
    from facefusion import content_analyser, core, logger, state_manager
    from facefusion.args import apply_args
    from facefusion.face_creator import get_many_faces
    from facefusion.face_selector import compare_faces, sort_and_filter_faces
    from facefusion.program import create_program
    from facefusion.video_manager import close_video_reader, get_reader, read_video_frame
    from facefusion.vision import count_video_frame_total, detect_video_fps, read_image, write_image

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
        "reference",
        "--reference-face-distance",
        str(FACE_DISTANCE),
        "--execution-providers",
        "cuda",
        "--execution-thread-count",
        EXECUTION_THREADS,
        "--video-memory-strategy",
        "strict",
        "--output-video-encoder",
        "libx264",
        "--output-video-quality",
        "90",
        "--log-level",
        "info",
        "--jobs-path",
        str(Path(args.output).parent / "jobs"),
        "--temp-path",
        str(Path(args.output).parent / "temp"),
    ]
    program = create_program()
    parsed = vars(program.parse_args())
    parsed["download_providers"] = []
    apply_args(parsed, state_manager.init_item)
    logger.init("info")
    if not core.common_pre_check() or not core.processors_pre_check():
        return refuse("safety/model precheck failed; run scripts/fetch_models.py")
    from facefusion.jobs import job_manager

    if not job_manager.init_jobs(state_manager.get_item("jobs_path")):
        return refuse("could not initialize private job storage")
    # Upstream prechecks the target video but not the reference image.
    if content_analyser.analyse_image(args.face):
        return refuse("reference image screening refused")
    reference_frame = read_image(args.face)
    source_faces = [] if reference_frame is None else get_many_faces([reference_frame])
    if len(source_faces) != 1:
        return refuse("reference must contain exactly one detectable face")
    # Screen every decodable target frame with upstream's unchanged classifier.
    total = count_video_frame_total(args.video)
    if total <= 0:
        return refuse("video has no frames")
    fps = detect_video_fps(args.video)
    sample_step = max(1, round(fps * FACE_SAMPLE_SECONDS))
    character_frame = None if args.character_time is None else round(args.character_time * fps)
    if character_frame is not None and character_frame >= total:
        return refuse("--character-time is past the end of the video")
    clusters = []
    chosen = None
    started = last_report = time.monotonic()
    reader = get_reader(args.video, "face_replace_screen")
    screening = deque()
    pool = ThreadPoolExecutor(SCREENING_THREADS)

    def first_refused_frame(keep: int):
        while len(screening) > keep:
            screened_number, refused = screening.popleft()
            if refused.result():
                return screened_number
        return None

    try:
        for frame_number in range(total):
            frame = read_video_frame(reader)
            if frame is None:
                return refuse("video could not be fully decoded for screening")
            screening.append((frame_number, pool.submit(content_analyser.analyse_frame, frame)))
            refused_frame = first_refused_frame(SCREENING_THREADS * 2)
            if refused_frame is not None:
                return refuse(f"video screening refused the frame at {clock(refused_frame / fps)}")
            if frame_number == character_frame:
                faces = sort_and_filter_faces(source_faces, get_many_faces([frame]))
                if not faces:
                    return refuse("no face detected at --character-time")
                chosen = {"frame": frame_number, "position": 0, "face": faces[0], "image": frame}
            if character_frame is None and frame_number % sample_step == 0:
                faces = sort_and_filter_faces(source_faces, get_many_faces([frame]))
                for position, face in enumerate(faces):
                    box = face.bounding_box
                    area = float((box[2] - box[0]) * (box[3] - box[1]))
                    cluster = next(
                        (c for c in clusters if compare_faces(face, c["face"], FACE_DISTANCE)),
                        None,
                    )
                    if cluster is None:
                        cluster = {"count": 0, "area": 0.0, "face": face}
                        clusters.append(cluster)
                    cluster["count"] += 1
                    if area > cluster["area"]:
                        # Keep the clearest appearance as the identity this character is matched to.
                        cluster.update(
                            area=area,
                            face=face,
                            frame=frame_number,
                            position=position,
                            image=frame,
                        )
            if time.monotonic() - last_report >= 60:
                last_report = time.monotonic()
                rate = (frame_number + 1) / (last_report - started)
                info(f"screening {(frame_number + 1) / total:.1%} at {rate:.1f} frames/s")
        refused_frame = first_refused_frame(0)
        if refused_frame is not None:
            return refuse(f"video screening refused the frame at {clock(refused_frame / fps)}")
        if read_video_frame(reader) is not None:
            return refuse("video frame count changed during screening")
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        close_video_reader(reader)
    info(f"screening finished at {total / (time.monotonic() - started):.1f} frames/s")
    if character_frame is None:
        if not clusters:
            return refuse("no target face detected")
        chosen = max(clusters, key=lambda c: c["count"])
    info(f"replacing the character shown at {clock(chosen['frame'] / fps)}")
    x1, y1, x2, y2 = (int(value) for value in chosen["face"].bounding_box)
    preview = chosen["image"].copy()
    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
    write_image(args.preview, preview)
    parsed["reference_frame_number"] = chosen["frame"]
    parsed["reference_face_position"] = chosen["position"]
    apply_args(parsed, state_manager.init_item)
    # Free the screening models so the 6 GB card only holds what swapping needs.
    content_analyser.clear_inference_pool()
    # Normal headless workflow keeps its own standard safety checks as well.
    if core.process_headless(parsed):
        return refuse("FaceFusion could not process the video")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
