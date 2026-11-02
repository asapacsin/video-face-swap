#!/usr/bin/env python3
"""Run pinned FaceFusion in-process with downloads disabled and fail-closed screening.

Modes: scan screens every frame and finds characters, pick finds the face at one frame, and swap
replaces one character within one frame range.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REVISION = "72470819a0373be3388b3929c8f8f311f418fc3c"
EXECUTION_THREADS = "4"
# Four parallel screens hit CUDA error 700 on both the RTX 4050 and the rented GPU.
SCREENING_THREADS = 1
# Identity sampling is expensive and a character stays on screen; screening is still every frame.
FACE_SAMPLE_SECONDS = 2
CHECKPOINT_SECONDS = 300
MAX_CANDIDATES = 5
# Same scale and default as upstream's reference-face matching.
FACE_DISTANCE = 0.3


def info(message: str) -> None:
    print(f"face-replace: {message}", file=sys.stderr, flush=True)


def refuse(message: str) -> int:
    print(f"face-replace-error: {message}", file=sys.stderr, flush=True)
    return 2


def stop(message: str) -> int:
    """A problem with this machine or setup, not the media: the job's progress stays resumable."""
    print(f"face-replace-error: {message}", file=sys.stderr, flush=True)
    return 4


def clock(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def save_json(path: Path, value: dict) -> None:
    partial = path.with_name(path.name + ".part")
    partial.write_text(json.dumps(value), encoding="utf-8")
    partial.replace(path)


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


def forbid_cpu_fallback(session_class) -> None:
    """Refuse a session that loads on the CPU and turn off its fallback.

    Graph nodes placed on the CPU and a fallback during a job are not detected here or tested.
    """
    original = session_class.__init__

    def guarded(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if self.get_providers()[0] != "CUDAExecutionProvider":
            # Upstream swallows this exception, so the reason has to be reported here.
            stop("a model did not load on the GPU; refusing to fall back to CPU")
            raise RuntimeError("model did not load on the GPU")
        self.disable_fallback()

    session_class.__init__ = guarded


def free_screening_after_analysis(content_analyser) -> None:
    """Screening is unchanged, but its models leave VRAM before the swap.

    Upstream screens again inside its own run and otherwise keeps them loaded until it ends.
    """
    original = content_analyser.analyse_video

    def analyse_then_free(*args):
        refused = original(*args)
        content_analyser.clear_inference_pool()
        return refused

    content_analyser.analyse_video = analyse_then_free


def same_person(embedding, other, numpy) -> bool:
    return float(numpy.interp(1 - numpy.dot(embedding, other), [0, 2], [0, 1])) < FACE_DISTANCE


def scan(args, ff, numpy) -> int:
    """Screen all frames from the last checkpoint and rank the characters by appearances."""
    state_path = Path(args.state)
    state = {"next_frame": 0, "clusters": []}
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    clusters = state["clusters"]
    total = ff.count_video_frame_total(args.video)
    if total <= 0:
        return refuse("video has no frames")
    fps = ff.detect_video_fps(args.video)
    sample_step = max(1, round(fps * FACE_SAMPLE_SECONDS))
    checkpoint_step = max(1, round(fps * CHECKPOINT_SECONDS))
    start = state["next_frame"]
    started = last_report = time.monotonic()
    reader = ff.get_reader(args.video, "face_replace_screen")
    if start:
        ff.seek_video_reader(reader, start)
    screening = deque()
    pool = ThreadPoolExecutor(SCREENING_THREADS)

    def first_refused_frame(keep: int):
        while len(screening) > keep:
            screened_number, refused = screening.popleft()
            if refused.result():
                return screened_number
        return None

    try:
        for frame_number in range(start, total):
            frame = ff.read_video_frame(reader)
            if frame is None:
                return refuse("video could not be fully decoded for screening")
            screening.append((frame_number, pool.submit(ff.analyse_frame, frame)))
            refused_frame = first_refused_frame(SCREENING_THREADS * 2)
            if refused_frame is not None:
                return refuse(f"video screening refused the frame at {clock(refused_frame / fps)}")
            if frame_number % sample_step == 0:
                for position, face in enumerate(ff.target_faces(frame)):
                    x1, y1, x2, y2 = (float(value) for value in face.bounding_box)
                    area = (x2 - x1) * (y2 - y1)
                    embedding = numpy.asarray(face.embedding_norm)
                    cluster = next(
                        (
                            c
                            for c in clusters
                            if same_person(embedding, numpy.asarray(c["embedding"]), numpy)
                        ),
                        None,
                    )
                    if cluster is None:
                        cluster = {"count": 0, "area": 0.0}
                        clusters.append(cluster)
                    cluster["count"] += 1
                    if area > cluster["area"]:
                        # Keep the clearest appearance as the identity this character is matched to.
                        cluster.update(
                            area=area,
                            frame=frame_number,
                            position=position,
                            box=[x1, y1, x2, y2],
                            embedding=embedding.tolist(),
                        )
            if (frame_number + 1) % checkpoint_step == 0:
                refused_frame = first_refused_frame(0)
                if refused_frame is not None:
                    return refuse(
                        f"video screening refused the frame at {clock(refused_frame / fps)}"
                    )
                state["next_frame"] = frame_number + 1
                save_json(state_path, state)
            if time.monotonic() - last_report >= 60:
                last_report = time.monotonic()
                rate = (frame_number + 1 - start) / (last_report - started)
                left = (total - frame_number - 1) / max(rate, 0.001)
                info(
                    f"scanning {(frame_number + 1) / total:.1%} at {rate:.1f} frames/s, "
                    f"about {clock(left)} left"
                )
        refused_frame = first_refused_frame(0)
        if refused_frame is not None:
            return refuse(f"video screening refused the frame at {clock(refused_frame / fps)}")
        if ff.read_video_frame(reader) is not None:
            return refuse("video frame count changed during screening")
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        ff.close_video_reader(reader)
    if not clusters:
        return refuse("no target face detected")
    elapsed = max(time.monotonic() - started, 0.001)
    info(f"scan finished at {(total - start) / elapsed:.1f} frames/s")
    ranked = sorted(clusters, key=lambda c: c["count"], reverse=True)[:MAX_CANDIDATES]
    candidates = []
    for index, cluster in enumerate(ranked):
        preview = str(Path(args.state).with_name(f"candidate-{index}.png"))
        ff.save_preview(cluster["frame"], cluster["box"], preview)
        candidates.append(
            {
                "index": index,
                "frame": cluster["frame"],
                "position": cluster["position"],
                "appearances": cluster["count"],
                "time": clock(cluster["frame"] / fps),
                "preview": preview,
            }
        )
    # Identity embeddings are needed only to resume the scan, so they end with it.
    state.pop("clusters")
    state.update(next_frame=total, done=True, candidates=candidates)
    save_json(state_path, state)
    return 0


def swap(args, analyse_video, process_headless, parsed) -> int:
    # Upstream's result cannot tell a refusal from a crash, so its screening runs here first.
    if analyse_video(args.video, args.start_frame, args.end_frame):
        return refuse("video screening refused this segment")
    if process_headless(parsed):
        return stop("FaceFusion could not process the video")
    return 0


def pick(args, ff) -> int:
    """Choose the largest face at one frame the owner named."""
    frame = ff.read_frame(args.video, args.reference_frame)
    faces = [] if frame is None else ff.target_faces(frame)
    if not faces:
        # A wrong time is not a refusal of the media, so the job's work stays.
        return stop("no face detected at the chosen character time")
    box = [float(value) for value in faces[0].bounding_box]
    preview = str(Path(args.state).with_name(f"pick-{args.reference_frame}.png"))
    ff.save_preview(args.reference_frame, box, preview)
    save_json(
        Path(args.state),
        {"frame": args.reference_frame, "position": 0, "preview": preview},
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("scan", "pick", "swap"))
    parser.add_argument("--video", required=True)
    parser.add_argument("--face", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--state")
    parser.add_argument("--output")
    parser.add_argument("--reference-frame", type=int, default=0)
    parser.add_argument("--reference-position", type=int, default=0)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    args = parser.parse_args()

    upstream = Path(args.upstream).resolve()
    sys.path.insert(0, str(upstream))
    os.chdir(upstream)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=upstream, capture_output=True, text=True, check=True
    ).stdout.strip()
    if revision != REVISION:
        return stop("pinned FaceFusion revision mismatch")
    if (
        subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--"], cwd=upstream, check=False
        ).returncode
        != 0
    ):
        return stop("FaceFusion checkout has modified tracked files")

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
        return stop("GPU convolution failed; refusing to fall back to CPU")
    forbid_cpu_fallback(onnxruntime.InferenceSession)

    import cv2
    from facefusion import content_analyser, core, logger, state_manager
    from facefusion.args import apply_args
    from facefusion.face_creator import get_many_faces
    from facefusion.face_selector import sort_and_filter_faces
    from facefusion.program import create_program
    from facefusion.video_manager import (
        close_video_reader,
        get_reader,
        read_video_frame,
        seek_video_reader,
    )
    from facefusion.vision import (
        count_video_frame_total,
        detect_video_fps,
        read_image,
        write_image,
    )
    from facefusion.vision import read_video_frame as read_frame

    # Parse only a fixed headless command; no user-supplied upstream flags.
    sys.argv = [
        "facefusion",
        "headless-run",
        "--source-paths",
        args.face,
        "--target-path",
        args.video,
        "--output-path",
        args.output or str(Path(args.work) / "unused.mp4"),
        "--processors",
        "face_swapper",
        "--face-swapper-model",
        "hyperswap_1a_256",
        "--face-selector-mode",
        "reference",
        "--reference-face-distance",
        str(FACE_DISTANCE),
        "--reference-frame-number",
        str(args.reference_frame),
        "--reference-face-position",
        str(args.reference_position),
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
        str(Path(args.work) / "jobs"),
        "--temp-path",
        str(Path(args.work) / "temp"),
    ]
    if args.mode == "swap":
        sys.argv += ["--trim-frame-start", str(args.start_frame)]
        sys.argv += ["--trim-frame-end", str(args.end_frame)]
    program = create_program()
    parsed = vars(program.parse_args())
    parsed["download_providers"] = []
    apply_args(parsed, state_manager.init_item)
    logger.init("info")
    if not core.common_pre_check() or not core.processors_pre_check():
        return stop("safety/model precheck failed; run scripts/fetch_models.py")
    from facefusion.jobs import job_manager

    if not job_manager.init_jobs(state_manager.get_item("jobs_path")):
        return stop("could not initialize private job storage")
    # Upstream prechecks the target video but not the reference image.
    if content_analyser.analyse_image(args.face):
        return refuse("reference image screening refused")
    reference_frame = read_image(args.face)
    source_faces = [] if reference_frame is None else get_many_faces([reference_frame])
    if len(source_faces) != 1:
        return refuse("reference must contain exactly one detectable face")
    if args.mode == "swap":
        free_screening_after_analysis(content_analyser)
        # Normal headless workflow keeps its own standard safety checks as well.
        return swap(args, content_analyser.analyse_video, core.process_headless, parsed)

    def save_preview(frame_number: int, box: list, path: str) -> None:
        image = read_frame(args.video, frame_number)
        x1, y1, x2, y2 = (int(value) for value in box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        write_image(path, image)

    ff = argparse.Namespace(
        analyse_frame=content_analyser.analyse_frame,
        close_video_reader=close_video_reader,
        count_video_frame_total=count_video_frame_total,
        detect_video_fps=detect_video_fps,
        get_reader=get_reader,
        read_frame=read_frame,
        read_video_frame=read_video_frame,
        save_preview=save_preview,
        seek_video_reader=seek_video_reader,
        target_faces=lambda frame: sort_and_filter_faces(source_faces, get_many_faces([frame])),
    )
    return scan(args, ff, numpy) if args.mode == "scan" else pick(args, ff)


if __name__ == "__main__":
    raise SystemExit(main())
