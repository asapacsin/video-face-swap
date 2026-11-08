#!/usr/bin/env python3
"""FaceFusion-compatible worker using local GHOST ONNX models without content screening."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from face_replace.models import FACEFUSION_REVISION, MODEL_SHA256

MAX_CANDIDATES = 5
FACE_SAMPLE_SECONDS = 2
CHECKPOINT_SECONDS = 300
MODEL_NAMES = tuple(MODEL_SHA256)
SCREENING_MODULES = ("facefusion.content_analyser", "facefusion.face_classifier")
AUDITED_CUDA_MODELS = []


class WorkerSetupError(RuntimeError):
    """A safe, actionable worker startup failure without private path details."""


def format_timestamp(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def matches_selected_target(similarity):
    # FaceFusion's reference distance is (1 - cosine_similarity) / 2 < 0.3.
    return similarity > 0.4


def face_area(face):
    x1, y1, x2, y2 = face.bounding_box
    return (x2 - x1) * (y2 - y1)


def sort_faces_largest_first(faces):
    return sorted(faces, key=face_area, reverse=True)


def validate_ghost_embedding(embedding, numpy):
    embedding = numpy.asarray(embedding, dtype=numpy.float32)
    if not numpy.isfinite(embedding).all() or numpy.linalg.norm(embedding) == 0:
        raise ValueError("GHOST identity conversion returned an invalid embedding")
    return embedding


def write_video_frame(writer, frame_bytes):
    try:
        writer.write(frame_bytes)
    except BrokenPipeError:
        return False
    return True


def info(message):
    print(f"face-replace: {message}", file=sys.stderr, flush=True)


def fail(message, code=4):
    print(f"face-replace-error: {message}", file=sys.stderr, flush=True)
    return code


def assert_screening_modules_unloaded():
    loaded = [
        module
        for module in sys.modules
        if any(module == prefix or module.startswith(prefix + ".") for prefix in SCREENING_MODULES)
    ]
    if loaded:
        raise WorkerSetupError("a disabled FaceFusion content-screening module was imported")


def save_json(path, value):
    path = Path(path)
    partial = path.with_name(path.name + ".part")
    partial.write_text(json.dumps(value), encoding="utf-8")
    partial.replace(path)


def require_cuda_session(session, model_name):
    session.disable_fallback()
    providers = session.get_providers()
    if not providers or providers[0] != "CUDAExecutionProvider":
        raise WorkerSetupError(f"{model_name} did not load on CUDA; CPU fallback is disabled")
    return session


def install_model_load_audit(ort, model_dir):
    """Allow only the pinned model files through ONNX Runtime's session factory."""
    model_dir = Path(model_dir).resolve()
    allowed = {f"{name}.onnx" for name in MODEL_NAMES}
    original = ort.InferenceSession
    loaded = []

    def audited_inference_session(model_path, *args, **kwargs):
        requested_path = Path(str(model_path)).resolve()
        model_name = requested_path.name
        if model_name not in allowed or requested_path.parent != model_dir:
            raise WorkerSetupError("an unapproved inference model load was requested")
        session = require_cuda_session(original(model_path, *args, **kwargs), model_name[:-5])
        loaded.append(model_name)
        AUDITED_CUDA_MODELS.append(model_name)
        return session

    ort.InferenceSession = audited_inference_session
    return loaded


def load_runtime(upstream, work):
    import cv2
    import numpy
    import onnxruntime as ort

    if os.name == "nt":
        try:
            for folder in (Path(sys.prefix) / "Lib/site-packages/nvidia").glob("*/bin"):
                os.add_dll_directory(str(folder))
                os.environ["PATH"] = str(folder) + os.pathsep + os.environ["PATH"]
            ort.preload_dlls(directory="")
        except Exception as exc:
            raise WorkerSetupError(
                "CUDA libraries could not be loaded; rerun Windows GPU setup"
            ) from exc
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=upstream, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise WorkerSetupError("could not verify the pinned FaceFusion checkout") from exc
    if revision != FACEFUSION_REVISION:
        raise WorkerSetupError("FaceFusion checkout is not at the pinned revision")
    try:
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--"], cwd=upstream, check=False
        ).returncode
        untracked_source = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", "facefusion"],
            cwd=upstream,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise WorkerSetupError("could not verify the pinned FaceFusion checkout") from exc
    if dirty or untracked_source:
        raise WorkerSetupError("FaceFusion checkout has modified or untracked source files")

    model_dir = upstream / ".assets/models"
    names = MODEL_NAMES
    AUDITED_CUDA_MODELS.clear()
    loaded_models = install_model_load_audit(ort, model_dir)
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    sessions = {}
    for name in names:
        model = model_dir / f"{name}.onnx"
        if not model.is_file():
            raise WorkerSetupError(f"required local model is missing: {name}")
        with model.open("rb") as model_file:
            digest = hashlib.file_digest(model_file, "sha256").hexdigest()
        if digest != MODEL_SHA256[name]:
            raise WorkerSetupError(f"required local model checksum does not match: {name}")
        try:
            session = require_cuda_session(
                ort.InferenceSession(
                    str(model), sess_options=options, providers=["CUDAExecutionProvider"]
                ),
                name,
            )
        except WorkerSetupError:
            raise
        except Exception as exc:
            raise WorkerSetupError(f"{name} could not be loaded on CUDA") from exc
        sessions[name] = session
    if loaded_models != [f"{name}.onnx" for name in MODEL_NAMES]:
        raise WorkerSetupError("pinned GPU model load audit did not match the required model set")

    sys.path.insert(0, str(upstream))
    os.chdir(upstream)
    from facefusion import (
        face_detector,
        face_landmarker,
        face_recognizer,
        inference_manager,
        state_manager,
    )
    from facefusion.face_helper import estimate_face_angle
    from facefusion.types import Face

    assert_screening_modules_unloaded()

    state_manager.STATE_SET["cli"] = {
        "execution_device_ids": [0],
        "execution_providers": ["cuda"],
        "download_providers": [],
        "face_detector_model": "yolo_face",
        "face_detector_size": "640x640",
        "face_detector_score": 0.5,
        "face_detector_margin": [0, 0, 0, 0],
        "face_detector_angles": [0],
        "face_landmarker_model": "2dfan4",
        "face_landmarker_score": 0,
        "face_recognizer_model": "arcface",
        "face_tracker_score": 0,
        "face_selector_mode": "one",
        "face_selector_order": "large-small",
        "face_selector_gender": "none",
        "face_selector_race": "none",
        "face_selector_age_start": 0,
        "face_selector_age_end": 0,
        "face_mask_types": ["box"],
        "face_mask_blur": 0.3,
        "face_mask_padding": [0, 0, 0, 0],
        "face_mask_areas": [
            "skin",
            "left-eyebrow",
            "right-eyebrow",
            "left-eye",
            "right-eye",
            "nose",
            "mouth",
            "upper-lip",
            "lower-lip",
        ],
        "face_swapper_model": "ghost_1_256",
        "face_swapper_pixel_boost": "256x256",
        "face_swapper_weight": 0.5,
    }

    def put(module, model_names, mapping):
        context = inference_manager.get_inference_context(module, model_names, 0, ["cuda"])
        inference_manager.INFERENCE_POOL_SET["cli"][context] = mapping

    put("facefusion.face_detector", ["yolo_face"], {"yolo_face": sessions["yoloface_8n"]})
    put(
        "facefusion.face_landmarker",
        ["2dfan4", "fan_68_5"],
        {"fan_68_5": sessions["fan_68_5"]},
    )
    put(
        "facefusion.face_recognizer",
        ["arcface"],
        {"face_recognizer": sessions["arcface_w600k_r50"]},
    )
    put(
        "facefusion.processors.modules.face_swapper.core",
        ["ghost_1_256"],
        {
            "face_swapper": sessions["ghost_1_256"],
            "embedding_converter": sessions["crossface_ghost"],
        },
    )

    from facefusion.face_helper import apply_nms, get_nms_threshold

    def faces(frame):
        assert_screening_modules_unloaded()
        boxes, scores, landmarks = face_detector.detect_faces(frame)
        keep = apply_nms(boxes, scores, 0.5, get_nms_threshold("yolo_face", [0]))
        result = []
        for index in keep:
            landmark5 = landmarks[index]
            embedding, normalized = face_recognizer.calculate_face_embedding(frame, landmark5)
            landmark68 = face_landmarker.estimate_face_landmark_68_5(landmark5)
            result.append(
                Face(
                    "detect",
                    boxes[index],
                    {"detector": scores[index], "landmarker": 0.0},
                    {"5": landmark5, "5/68": landmark5, "68": landmark68, "68/5": landmark68},
                    estimate_face_angle(landmark68),
                    embedding,
                    normalized,
                    None,
                    None,
                    None,
                )
            )
        return sort_faces_largest_first(result)

    return cv2, numpy, ort, sessions, faces


def _main(argv=None, runtime_loader=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("scan", "pick", "swap"))
    for name in ("video", "face", "work", "upstream"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--state")
    parser.add_argument("--output")
    parser.add_argument("--reference-frame", type=int, default=0)
    parser.add_argument("--reference-position", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--frame-rate")
    parser.add_argument("--frame-count", type=int)
    args = parser.parse_args(argv)
    upstream, work = Path(args.upstream).resolve(), Path(args.work).resolve()
    if not args.state:
        return fail("worker state file is required")
    if args.frame_count is not None and args.frame_count <= 0:
        return fail("video frame count is invalid")
    if args.mode == "swap":
        if not args.output:
            return fail("swap output path is required")
        if args.start_frame < 0 or args.end_frame is None or args.end_frame <= args.start_frame:
            return fail("swap frame range is invalid")
        if args.frame_count is not None and args.end_frame > args.frame_count:
            return fail("swap frame range exceeds the source video", 2)
        try:
            frame_rate = Fraction(args.frame_rate)
        except (TypeError, ValueError, ZeroDivisionError):
            return fail("swap frame rate is invalid")
        if frame_rate <= 0:
            return fail("swap frame rate is invalid")
    try:
        loader = runtime_loader or load_runtime
        cv2, numpy, ort, sessions, get_faces = loader(upstream, work)
    except WorkerSetupError as exc:
        return fail(str(exc), 4)
    except Exception:
        return fail("GHOST worker setup failed; check the pinned runtime and CUDA setup", 4)
    from facefusion.face_helper import paste_back, warp_face_by_face_landmark_5
    from facefusion.face_masker import create_box_mask
    from facefusion.video_manager import (
        close_video_reader,
        get_reader,
        read_video_frame,
        seek_video_reader,
    )
    from facefusion.vision import (
        count_video_frame_total,
        detect_video_fps,
        read_static_image,
        write_image,
    )

    def frame_at(frame_number):
        reader = get_reader(args.video, "ghost_worker_pick")
        seek_video_reader(reader, frame_number)
        frame = read_video_frame(reader)
        close_video_reader(reader)
        return frame

    def preview(frame_number, box, path):
        image = frame_at(frame_number)
        if image is None:
            return False
        x1, y1, x2, y2 = (int(x) for x in box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return bool(write_image(path, image))

    state_path = Path(args.state)
    fps = Fraction(args.frame_rate) if args.frame_rate else detect_video_fps(args.video)
    if args.mode in ("scan", "pick"):
        if args.mode == "pick":
            frame_number = args.reference_frame
            frame = frame_at(frame_number)
            found = [] if frame is None else get_faces(frame)
            if not found:
                return fail("no face at requested character time", 4)
            if not 0 <= args.reference_position < len(found):
                return fail("selected character could not be found in the requested frame", 4)
            face = found[args.reference_position]
            box = [float(x) for x in face.bounding_box]
            image_path = state_path.with_name(f"pick-{frame_number}.png")
            if not preview(frame_number, box, str(image_path)):
                return fail("could not write the selected face preview", 4)
            save_json(
                state_path,
                {
                    "frame": frame_number,
                    "position": args.reference_position,
                    "preview": str(image_path),
                },
            )
            return 0
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {"next_frame": 0, "clusters": []}
        )
        clusters = state["clusters"]
        total = args.frame_count or count_video_frame_total(args.video)
        if total <= 0:
            return fail("video frame count is invalid")
        step = max(1, round(fps * FACE_SAMPLE_SECONDS))
        checkpoint = max(1, round(fps * CHECKPOINT_SECONDS))
        reader = get_reader(args.video, "ghost_worker_scan")
        if state["next_frame"]:
            seek_video_reader(reader, state["next_frame"])
        try:
            for frame_number in range(state["next_frame"], total):
                frame = read_video_frame(reader)
                if frame is None:
                    return fail("video decode ended early", 4)
                if frame_number % step == 0:
                    for position, face in enumerate(get_faces(frame)):
                        box = [float(x) for x in face.bounding_box]
                        area = (box[2] - box[0]) * (box[3] - box[1])
                        cluster = next(
                            (
                                c
                                for c in clusters
                                if float(numpy.dot(face.embedding_norm, c["embedding"])) > 0.7
                            ),
                            None,
                        )
                        if cluster is None:
                            cluster = {
                                "count": 0,
                                "area": 0,
                                "embedding": face.embedding_norm.tolist(),
                            }
                            clusters.append(cluster)
                        cluster["count"] += 1
                        if area > cluster["area"]:
                            cluster.update(
                                area=area, frame=frame_number, position=position, box=box
                            )
                if (frame_number + 1) % checkpoint == 0:
                    state["next_frame"] = frame_number + 1
                    save_json(state_path, state)
            if read_video_frame(reader) is not None:
                return fail("video frame count changed during scan", 2)
        finally:
            close_video_reader(reader)
        ranked = sorted(clusters, key=lambda c: c["count"], reverse=True)[:MAX_CANDIDATES]
        candidates = []
        for index, cluster in enumerate(ranked):
            path = state_path.with_name(f"candidate-{index}.png")
            if not preview(cluster["frame"], cluster["box"], str(path)):
                return fail("could not write a face preview", 4)
            candidates.append(
                {
                    "index": index,
                    "frame": cluster["frame"],
                    "position": cluster["position"],
                    "appearances": cluster["count"],
                    "time": format_timestamp(cluster["frame"] / fps),
                    "preview": str(path),
                }
            )
        save_json(state_path, {"next_frame": total, "done": True, "candidates": candidates})
        return 0 if candidates else fail("no target face detected", 2)

    target_reference_frame = frame_at(args.reference_frame)
    target_reference_faces = (
        [] if target_reference_frame is None else get_faces(target_reference_frame)
    )
    if not 0 <= args.reference_position < len(target_reference_faces):
        return fail("selected character could not be found in the reference frame", 2)
    selected_target = target_reference_faces[args.reference_position]
    selected_target_embedding = numpy.asarray(selected_target.embedding_norm, dtype=numpy.float32)

    source_frame = read_static_image(args.face)
    source_faces = [] if source_frame is None else get_faces(source_frame)
    if len(source_faces) != 1:
        return fail("reference photo must contain exactly one detectable face", 2)
    source_face = source_faces[0]
    source_embedding = numpy.asarray(source_face.embedding, dtype=numpy.float32).reshape(1, 512)
    converted_embedding = sessions["crossface_ghost"].run(None, {"input": source_embedding})[0]
    try:
        source_embedding = validate_ghost_embedding(converted_embedding, numpy)
    except ValueError as exc:
        return fail(str(exc))

    height, width = target_reference_frame.shape[:2]
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return fail("ffmpeg executable is unavailable")
    command = [
        ffmpeg,
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(frame_rate),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        args.output,
    ]
    reader = get_reader(args.video, "ghost_worker_swap")
    seek_video_reader(reader, args.start_frame)
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    processed = 0
    matched_frames = 0
    try:
        for frame_number in range(args.start_frame, args.end_frame):
            frame = read_video_frame(reader)
            if frame is None:
                return fail("video decode ended before the swap range", 4)
            matched_this_frame = False
            for face in get_faces(frame):
                similarity = float(
                    numpy.dot(selected_target_embedding, numpy.asarray(face.embedding_norm))
                )
                if not matches_selected_target(similarity):
                    continue
                matched_this_frame = True
                crop, matrix = warp_face_by_face_landmark_5(
                    frame, face.landmark_set["5/68"], "arcface_112_v1", (256, 256)
                )
                target = crop[:, :, ::-1].astype(numpy.float32) / 127.5 - 1.0
                target = target.transpose(2, 0, 1)[None, ...]
                generated = (
                    sessions["ghost_1_256"]
                    .run(None, {"target": target, "source": source_embedding})[0][0]
                    .transpose(1, 2, 0)
                )
                generated = ((generated + 1.0) * 127.5).clip(0, 255).astype(numpy.uint8)
                generated = generated[:, :, ::-1]
                mask = create_box_mask(crop, 0.3, [0, 0, 0, 0])
                frame = paste_back(frame, generated, mask, matrix)
            if matched_this_frame:
                matched_frames += 1
            if process.stdin is None:
                return fail("video encoder input is unavailable")
            if not write_video_frame(process.stdin, frame.tobytes()):
                return fail("video encoder stopped accepting frames", 4)
            processed += 1
        if process.stdin is not None:
            try:
                process.stdin.close()
            except BrokenPipeError:
                return fail("video encoder stopped accepting frames", 4)
        result = process.wait()
        if result:
            return fail(f"video encoder failed with exit code {result}", 4)
    finally:
        close_video_reader(reader)
        if process.poll() is None:
            process.kill()
            process.wait()
    if processed != args.end_frame - args.start_frame:
        return fail("swapped frame count differs from requested range", 4)
    save_json(
        state_path,
        {"done": True, "processed_frames": processed, "matched_frames": matched_frames},
    )
    return 0


def main(argv=None, runtime_loader=None):
    AUDITED_CUDA_MODELS.clear()
    try:
        result = _main(argv=argv, runtime_loader=runtime_loader)
        if AUDITED_CUDA_MODELS:
            assert_screening_modules_unloaded()
            info("audited CUDA inference sessions: " + ", ".join(AUDITED_CUDA_MODELS))
            info("content-screening modules: not imported")
        return result
    except WorkerSetupError as exc:
        return fail(str(exc), 4)
    except (OSError, subprocess.SubprocessError):
        return fail("worker could not read media or start the video encoder", 4)
    except (KeyError, ValueError, TypeError, IndexError):
        return fail("worker received invalid media or model data", 4)
    except Exception:
        return fail("GPU inference failed; check the local runtime and CUDA setup", 4)


if __name__ == "__main__":
    raise SystemExit(main())
