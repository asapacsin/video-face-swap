import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy

from face_replace.models import (
    BACKEND_REVISION,
    FACEFUSION_REVISION,
    MODEL_RELEASES,
    MODEL_SHA256,
)

WORKER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ghost_worker.py"
SPEC = importlib.util.spec_from_file_location("ghost_worker", WORKER_PATH)
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


class GhostWorkerContractTest(unittest.TestCase):
    class Frame:
        def __init__(self, index):
            self.index = index

    class Reader:
        def __init__(self, frames):
            self.frames = frames
            self.index = 0

        def read(self):
            if self.index >= len(self.frames):
                return None
            frame = self.frames[self.index]
            self.index += 1
            return frame

    class EncoderInput:
        def __init__(self, broken=False):
            self.broken = broken
            self.frames = []

        def write(self, value):
            if self.broken:
                raise BrokenPipeError
            self.frames.append(value)

        def close(self):
            pass

    class EncoderProcess:
        def __init__(self, broken=False, return_code=0):
            self.stdin = GhostWorkerContractTest.EncoderInput(broken)
            self.return_code = return_code
            self.killed = False

        def wait(self):
            return self.return_code

        def poll(self):
            return self.return_code if not self.killed else -9

        def kill(self):
            self.killed = True

    def test_loads_all_required_gpu_models_including_landmark_refiner(self):
        self.assertEqual(
            WORKER.MODEL_NAMES,
            (
                "yoloface_8n",
                "fan_68_5",
                "arcface_w600k_r50",
                "ghost_1_256",
                "crossface_ghost",
            ),
        )

    def test_backend_revision_is_derived_from_pinned_source_and_model_hashes(self):
        identity = "|".join(
            [FACEFUSION_REVISION] + [f"{name}:{MODEL_SHA256[name]}" for name in MODEL_RELEASES]
        )
        self.assertEqual(
            BACKEND_REVISION, f"ghost-{hashlib.sha256(identity.encode()).hexdigest()[:16]}"
        )

    def test_target_match_uses_facefusion_reference_distance(self):
        self.assertFalse(WORKER.matches_selected_target(0.4))
        self.assertFalse(WORKER.matches_selected_target(0.399))
        self.assertTrue(WORKER.matches_selected_target(0.401))

    def test_runtime_rejects_cpu_only_sessions_and_disables_provider_fallback(self):
        class Session:
            def __init__(self, providers):
                self.providers = providers
                self.disabled = False

            def disable_fallback(self):
                self.disabled = True

            def get_providers(self):
                return self.providers

        cpu_session = Session(["CPUExecutionProvider"])
        with self.assertRaisesRegex(RuntimeError, "did not load on CUDA"):
            WORKER.require_cuda_session(cpu_session, "test-model")
        self.assertTrue(cpu_session.disabled)
        cuda_session = Session(["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.assertIs(WORKER.require_cuda_session(cuda_session, "test-model"), cuda_session)
        self.assertTrue(cuda_session.disabled)

    def test_candidate_time_uses_clock_format(self):
        self.assertEqual(WORKER.format_timestamp(59.9), "00:00:59")
        self.assertEqual(WORKER.format_timestamp(3661.1), "01:01:01")

    def test_faces_are_sorted_by_area_largest_first(self):
        small = SimpleNamespace(bounding_box=(0, 0, 10, 10))
        large = SimpleNamespace(bounding_box=(0, 0, 20, 20))
        medium = SimpleNamespace(bounding_box=(0, 0, 15, 15))
        self.assertEqual(
            WORKER.sort_faces_largest_first([small, large, medium]), [large, medium, small]
        )

    def test_crossface_embedding_keeps_original_scale(self):
        import numpy

        converted = numpy.asarray([[3.0, 4.0]], dtype=numpy.float32)
        self.assertTrue(
            numpy.array_equal(WORKER.validate_ghost_embedding(converted, numpy), converted)
        )

    def test_encoder_broken_pipe_is_reported(self):
        class ClosedWriter:
            def write(self, _):
                raise BrokenPipeError

        self.assertFalse(WORKER.write_video_frame(ClosedWriter(), b"frame"))

    def test_active_model_allowlist_excludes_screening_weights(self):
        self.assertEqual(
            WORKER.SCREENING_MODULES,
            ("facefusion.content_analyser", "facefusion.face_classifier"),
        )
        self.assertNotIn("nsfw_", " ".join(WORKER.MODEL_NAMES))

    def test_runtime_guard_rejects_imported_screening_modules(self):
        module = types.ModuleType("facefusion.content_analyser")
        with mock.patch.dict(sys.modules, {"facefusion.content_analyser": module}):
            with self.assertRaisesRegex(
                WORKER.WorkerSetupError, "disabled FaceFusion content-screening module"
            ):
                WORKER.assert_screening_modules_unloaded()

    def test_model_load_audit_rejects_any_unpinned_inference_session(self):
        created = []
        provider = ["CUDAExecutionProvider"]

        class FakeSession:
            def disable_fallback(self):
                pass

            def get_providers(self):
                return provider

        ort = SimpleNamespace(
            InferenceSession=lambda path, **_: created.append(Path(path).name) or FakeSession()
        )
        with tempfile.TemporaryDirectory() as temp:
            loaded = WORKER.install_model_load_audit(ort, temp)
            ort.InferenceSession(Path(temp) / "ghost_1_256.onnx")
        self.assertEqual(loaded, ["ghost_1_256.onnx"])
        provider[:] = ["CPUExecutionProvider"]
        with self.assertRaisesRegex(WORKER.WorkerSetupError, "did not load on CUDA"):
            ort.InferenceSession(Path(temp) / "fan_68_5.onnx")
        provider[:] = ["CUDAExecutionProvider"]
        with self.assertRaisesRegex(WORKER.WorkerSetupError, "unapproved inference model"):
            ort.InferenceSession(Path(temp) / "nsfw_1.onnx")
        with self.assertRaisesRegex(WORKER.WorkerSetupError, "unapproved inference model"):
            ort.InferenceSession(Path(temp) / "unrelated.onnx")
        with self.assertRaisesRegex(WORKER.WorkerSetupError, "unapproved inference model"):
            ort.InferenceSession(Path(temp).parent / "ghost_1_256.onnx")
        self.assertEqual(created, ["ghost_1_256.onnx", "fan_68_5.onnx"])

    def test_runtime_loads_only_pinned_models_and_never_opens_screening_weights(self):
        import numpy

        fake_ort = types.ModuleType("onnxruntime")
        fake_ort.SessionOptions = type(
            "SessionOptions",
            (),
            {"add_session_config_entry": lambda self, *_: None},
        )
        opened = []

        class FakeSession:
            def __init__(self, path, **_):
                opened.append(Path(path).name)

            def disable_fallback(self):
                pass

            def get_providers(self):
                return ["CUDAExecutionProvider"]

        fake_ort.InferenceSession = FakeSession
        fake_ort.preload_dlls = lambda **_: None

        facefusion = types.ModuleType("facefusion")
        facefusion.__path__ = []
        for name in ("face_detector", "face_landmarker", "face_recognizer"):
            module = types.ModuleType(f"facefusion.{name}")
            setattr(facefusion, name, module)
        facefusion.face_detector.detect_faces = lambda _: ([], [], [])
        inference_manager = types.ModuleType("facefusion.inference_manager")
        inference_manager.INFERENCE_POOL_SET = {"cli": {}}
        inference_manager.get_inference_context = lambda *args: tuple(map(str, args))
        facefusion.inference_manager = inference_manager
        state_manager = types.ModuleType("facefusion.state_manager")
        state_manager.STATE_SET = {}
        facefusion.state_manager = state_manager
        face_helper = types.ModuleType("facefusion.face_helper")
        face_helper.estimate_face_angle = lambda _: None
        face_helper.apply_nms = lambda *args: []
        face_helper.get_nms_threshold = lambda *_: 0.5
        face_types = types.ModuleType("facefusion.types")
        face_types.Face = object

        modules = {
            "onnxruntime": fake_ort,
            "facefusion": facefusion,
            "facefusion.inference_manager": inference_manager,
            "facefusion.state_manager": state_manager,
            "facefusion.face_helper": face_helper,
            "facefusion.types": face_types,
        }
        modules.update(
            {
                f"facefusion.{name}": getattr(facefusion, name)
                for name in (
                    "face_detector",
                    "face_landmarker",
                    "face_recognizer",
                )
            }
        )
        cwd = Path.cwd()
        original_sys_path = list(sys.path)
        with tempfile.TemporaryDirectory() as temp:
            upstream = Path(temp) / "upstream-facefusion"
            model_dir = upstream / ".assets" / "models"
            model_dir.mkdir(parents=True)
            for name in WORKER.MODEL_NAMES:
                (model_dir / f"{name}.onnx").write_bytes(b"mock model")

            has_untracked_source = False

            def fake_run(args, **_):
                if args[:2] == ["git", "status"]:
                    return SimpleNamespace(
                        stdout=("?? facefusion/injected.py\n" if has_untracked_source else ""),
                        returncode=0,
                    )
                return SimpleNamespace(
                    stdout="72470819a0373be3388b3929c8f8f311f418fc3c\n", returncode=0
                )

            def fake_digest(file, _):
                name = Path(file.name).stem
                return SimpleNamespace(hexdigest=lambda: WORKER.MODEL_SHA256[name])

            original_open = Path.open

            def guarded_open(path, *args, **kwargs):
                if "nsfw_" in str(path).lower():
                    raise AssertionError("screening weight was opened")
                return original_open(path, *args, **kwargs)

            try:
                with (
                    mock.patch.dict(sys.modules, modules),
                    mock.patch.object(WORKER.subprocess, "run", side_effect=fake_run),
                    mock.patch.object(WORKER.hashlib, "file_digest", side_effect=fake_digest),
                    mock.patch.object(Path, "open", guarded_open),
                ):
                    _, _, _, sessions, find_faces = WORKER.load_runtime(upstream, Path(temp))
                    self.assertEqual(find_faces(numpy.zeros((16, 16, 3), dtype=numpy.uint8)), [])
                    self.assertEqual(state_manager.STATE_SET["cli"]["download_providers"], [])
                    for prefix in WORKER.SCREENING_MODULES:
                        self.assertFalse(
                            any(
                                name == prefix or name.startswith(prefix + ".")
                                for name in sys.modules
                            ),
                            f"{prefix} appeared in sys.modules",
                        )
                    has_untracked_source = True
                    with self.assertRaisesRegex(
                        WORKER.WorkerSetupError, "modified or untracked source files"
                    ):
                        WORKER.load_runtime(upstream, Path(temp))
            finally:
                os.chdir(cwd)
                sys.path[:] = original_sys_path

        self.assertEqual(opened, [f"{name}.onnx" for name in WORKER.MODEL_NAMES])
        self.assertTrue(
            all(s.get_providers()[0] == "CUDAExecutionProvider" for s in sessions.values())
        )

    def build_worker_apis(
        self, frames, total=None, source_frame=None, written=None, preview_writes=True
    ):
        video_manager = types.ModuleType("facefusion.video_manager")

        def get_reader(_video, _name):
            return self.Reader(frames)

        video_manager.get_reader = get_reader
        video_manager.seek_video_reader = lambda reader, frame: setattr(reader, "index", frame)
        video_manager.read_video_frame = lambda reader: reader.read()
        video_manager.close_video_reader = lambda _reader: None

        vision = types.ModuleType("facefusion.vision")
        vision.count_video_frame_total = lambda _video: len(frames) if total is None else total
        vision.detect_video_fps = lambda _video: 1.0
        vision.read_static_image = lambda _image: source_frame
        written = [] if written is None else written
        vision.write_image = lambda path, image: (written.append(Path(path).name) or preview_writes)

        face_helper = types.ModuleType("facefusion.face_helper")
        face_helper.paste_back = lambda frame, *_: frame
        face_helper.warp_face_by_face_landmark_5 = lambda *_: (
            numpy.zeros((256, 256, 3), dtype=numpy.uint8),
            "matrix",
        )
        face_helper.apply_nms = lambda *_: []
        face_helper.get_nms_threshold = lambda *_: 0.5
        face_masker = types.ModuleType("facefusion.face_masker")
        face_masker.create_box_mask = lambda *_: None
        package = types.ModuleType("facefusion")
        package.__path__ = []
        return {
            "facefusion": package,
            "facefusion.video_manager": video_manager,
            "facefusion.vision": vision,
            "facefusion.face_helper": face_helper,
            "facefusion.face_masker": face_masker,
        }, written

    def make_face(self, embedding, size=20):
        vector = numpy.zeros((512,), dtype=numpy.float32)
        vector[embedding] = 1.0
        return SimpleNamespace(
            bounding_box=(0, 0, size, size),
            embedding=vector,
            embedding_norm=vector,
            landmark_set={"5/68": numpy.zeros((68, 2), dtype=numpy.float32)},
        )

    def run_main(
        self,
        argv,
        frames,
        get_faces,
        source_frame=None,
        sessions=None,
        broken_encoder=False,
        total=None,
        preview_writes=True,
    ):
        import numpy

        if sessions is None:
            sessions = {}
        cv2 = SimpleNamespace(rectangle=lambda image, *_: image)
        ort = SimpleNamespace()
        apis, previews = self.build_worker_apis(
            frames,
            total=total,
            source_frame=source_frame,
            preview_writes=preview_writes,
        )

        def loader(_upstream, _work):
            return cv2, numpy, ort, sessions, get_faces

        encoder = self.EncoderProcess(broken=broken_encoder)
        popen = mock.Mock(return_value=encoder)
        old_cwd = Path.cwd()
        try:
            with (
                mock.patch.dict(sys.modules, apis),
                mock.patch.object(WORKER.subprocess, "Popen", popen),
                mock.patch("shutil.which", return_value="ffmpeg"),
                mock.patch.object(WORKER, "load_runtime", side_effect=loader),
            ):
                result = WORKER.main(argv, runtime_loader=loader)
                self.last_encoder_command = popen.call_args.args[0] if popen.called else None
        finally:
            os.chdir(old_cwd)
        return result, encoder, previews

    def worker_argv(self, temp, mode, frames, **extra):
        state = Path(temp) / f"{mode}.json"
        argv = [
            "--mode",
            mode,
            "--video",
            "synthetic.mp4",
            "--face",
            "synthetic.png",
            "--work",
            str(temp),
            "--upstream",
            str(temp),
            "--state",
            str(state),
            "--frame-count",
            str(len(frames)),
        ]
        for key, value in extra.items():
            argv.extend(["--" + key.replace("_", "-"), str(value)])
        return argv, state

    def test_main_scan_ranks_candidates_and_records_largest_face_position(self):
        frames = [self.Frame(i) for i in range(5)]
        big, small = self.make_face(0, 40), self.make_face(1, 20)
        faces = {0: [small, big], 2: [small, big], 4: [big]}

        def get_faces(frame):
            return WORKER.sort_faces_largest_first(faces.get(frame.index, []))

        with tempfile.TemporaryDirectory() as temp:
            argv, state_path = self.worker_argv(temp, "scan", frames)
            code, _, previews = self.run_main(argv, frames, get_faces)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            saved_state = state_path.read_text(encoding="utf-8")
        self.assertEqual(code, 0)
        self.assertEqual(
            [(item["appearances"], item["position"]) for item in state["candidates"]],
            [(3, 0), (2, 1)],
        )
        self.assertNotIn("embedding", saved_state)
        self.assertEqual(len(previews), 2)

    def test_main_scan_does_not_publish_a_candidate_when_preview_write_fails(self):
        frames = [self.Frame(i) for i in range(3)]
        face = self.make_face(0, 40)
        with tempfile.TemporaryDirectory() as temp:
            argv, state_path = self.worker_argv(temp, "scan", frames)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code, _, _ = self.run_main(argv, frames, lambda _: [face], preview_writes=False)
            self.assertEqual(code, 4)
            self.assertFalse(state_path.exists())
            self.assertIn("could not write a face preview", stderr.getvalue())

    def test_unexpected_inference_error_has_no_traceback_or_private_path(self):
        frames = [self.Frame(i) for i in range(3)]
        with tempfile.TemporaryDirectory() as temp:
            argv, _ = self.worker_argv(temp, "scan", frames)

            def fail_inference(_):
                raise RuntimeError(f"failure in {temp}\\private")

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code, _, _ = self.run_main(argv, frames, fail_inference)
            self.assertEqual(code, 4)
            self.assertIn("GPU inference failed", stderr.getvalue())
            self.assertNotIn(temp, stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())

    def test_main_scan_resumes_from_checkpoint_and_rejects_early_decode(self):
        frames = [self.Frame(i) for i in range(5)]
        face = self.make_face(0, 40)

        def get_faces(frame):
            return [face] if frame.index in (0, 2, 4) else []

        with tempfile.TemporaryDirectory() as temp:
            argv, state_path = self.worker_argv(temp, "scan", frames)
            state_path.write_text(
                json.dumps(
                    {
                        "next_frame": 2,
                        "clusters": [
                            {
                                "count": 1,
                                "area": 1600,
                                "frame": 0,
                                "position": 0,
                                "box": [0, 0, 40, 40],
                                "embedding": face.embedding_norm.tolist(),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            code, _, _ = self.run_main(argv, frames, get_faces)
            self.assertEqual(code, 0)
            self.assertEqual(
                json.loads(state_path.read_text(encoding="utf-8"))["candidates"][0]["appearances"],
                3,
            )

        with tempfile.TemporaryDirectory() as temp:
            argv, state_path = self.worker_argv(temp, "scan", frames)
            code, _, _ = self.run_main(argv, frames[:2], get_faces, total=5)
            self.assertEqual(code, 4)
            self.assertFalse(state_path.exists())

    def test_main_scan_rejects_extra_frames(self):
        frames = [self.Frame(i) for i in range(3)]
        face = self.make_face(0, 40)
        with tempfile.TemporaryDirectory() as temp:
            argv, _ = self.worker_argv(temp, "scan", frames)
            argv.extend(["--frame-count", "2"])
            code, _, _ = self.run_main(argv, frames, lambda _: [face], total=3)
            self.assertEqual(code, 2)

    def test_invalid_swap_configuration_fails_before_runtime_load(self):
        frame = self.Frame(0)
        with tempfile.TemporaryDirectory() as temp:
            argv, _ = self.worker_argv(temp, "swap", [frame], end_frame=1, frame_rate="not-a-rate")

            def unexpected_loader(*_):
                self.fail("invalid worker configuration must fail before CUDA model loading")

            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(WORKER.main(argv, runtime_loader=unexpected_loader), 4)

    def test_cuda_setup_failure_is_resumable_and_reports_the_specific_model(self):
        with tempfile.TemporaryDirectory() as temp:
            argv, _ = self.worker_argv(temp, "scan", [self.Frame(i) for i in range(3)])

            def cpu_only_loader(*_):
                raise WORKER.WorkerSetupError(
                    "ghost_1_256 did not load on CUDA; CPU fallback is disabled"
                )

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = WORKER.main(argv, runtime_loader=cpu_only_loader)
            self.assertEqual(code, 4)
            self.assertIn("ghost_1_256 did not load on CUDA", stderr.getvalue())

    def test_main_pick_saves_the_requested_face_position(self):
        frames = [self.Frame(i) for i in range(3)]
        large, small = self.make_face(0, 40), self.make_face(1, 20)

        def get_faces(_frame):
            return WORKER.sort_faces_largest_first([small, large])

        with tempfile.TemporaryDirectory() as temp:
            argv, state = self.worker_argv(
                temp, "pick", frames, reference_frame=1, reference_position=1
            )
            code, _, _ = self.run_main(argv, frames, get_faces)
            selection = json.loads(state.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual((selection["frame"], selection["position"]), (1, 1))

    def test_main_pick_without_a_face_is_resumable(self):
        frames = [self.Frame(i) for i in range(3)]
        with tempfile.TemporaryDirectory() as temp:
            argv, _ = self.worker_argv(temp, "pick", frames, reference_frame=1)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code, _, _ = self.run_main(argv, frames, lambda _: [])
            self.assertEqual(code, 4)
            self.assertIn("no face at requested character time", stderr.getvalue())

    def test_main_pick_missing_selected_position_is_resumable(self):
        frames = [self.Frame(i) for i in range(3)]
        face = self.make_face(0)
        with tempfile.TemporaryDirectory() as temp:
            argv, _ = self.worker_argv(
                temp, "pick", frames, reference_frame=1, reference_position=1
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code, _, _ = self.run_main(argv, frames, lambda _: [face])
        self.assertEqual(code, 4)
        self.assertIn("selected character could not be found", stderr.getvalue())

    def test_main_swap_skips_non_matching_faces(self):
        import numpy

        frame = numpy.zeros((256, 256, 3), dtype=numpy.uint8)
        frames = [frame]
        source = object()
        target_face, other_face = self.make_face(0), self.make_face(2)
        source_face = self.make_face(1)
        calls = {"target": 0, "swap": 0}

        class Converter:
            def run(self, *_):
                return [source_face.embedding.reshape(1, -1)]

        class Swapper:
            def run(self, _, inputs):
                calls["swap"] += 1
                return [inputs["target"]]

        def get_faces(image):
            if image is source:
                return [source_face]
            calls["target"] += 1
            return [target_face] if calls["target"] == 1 else [other_face]

        sessions = {"crossface_ghost": Converter(), "ghost_1_256": Swapper()}
        with tempfile.TemporaryDirectory() as temp:
            argv, state = self.worker_argv(
                temp,
                "swap",
                frames,
                output=str(Path(temp) / "segment.mp4"),
                end_frame=1,
                frame_rate="24",
            )
            code, _, _ = self.run_main(
                argv, frames, get_faces, source_frame=source, sessions=sessions
            )
            saved = json.loads(state.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(saved["matched_frames"], 0)
        self.assertEqual(calls["swap"], 0)

    def test_main_swap_checks_matches_frame_count_and_encoder_failure(self):
        import numpy

        frame = numpy.zeros((256, 256, 3), dtype=numpy.uint8)
        frames = [frame]
        source = object()
        target_face = self.make_face(0)
        source_face = self.make_face(1)

        class Converter:
            def run(self, *_):
                return [source_face.embedding.reshape(1, -1)]

        class Swapper:
            def run(self, _, inputs):
                return [inputs["target"]]

        sessions = {"crossface_ghost": Converter(), "ghost_1_256": Swapper()}

        def get_faces(image):
            return [source_face] if image is source else [target_face]

        with tempfile.TemporaryDirectory() as temp:
            argv, state = self.worker_argv(
                temp,
                "swap",
                frames,
                output=str(Path(temp) / "segment.mp4"),
                end_frame=1,
                frame_rate="24000/1001",
            )
            code, encoder, _ = self.run_main(
                argv, frames, get_faces, source_frame=source, sessions=sessions
            )
            self.assertEqual(code, 0)
            self.assertEqual(
                self.last_encoder_command[self.last_encoder_command.index("-r") + 1],
                "24000/1001",
            )
            self.assertEqual(len(encoder.stdin.frames), 1)
            self.assertEqual(
                json.loads(state.read_text(encoding="utf-8")),
                {"done": True, "processed_frames": 1, "matched_frames": 1},
            )

        with tempfile.TemporaryDirectory() as temp:
            argv, state = self.worker_argv(
                temp,
                "swap",
                frames,
                output=str(Path(temp) / "segment.mp4"),
                end_frame=1,
                frame_rate="24000/1001",
            )
            code, _, _ = self.run_main(
                argv,
                frames,
                lambda image: [source_face] if image is source else [target_face],
                source_frame=source,
                sessions=sessions,
                broken_encoder=True,
            )
            self.assertEqual(code, 4)
            self.assertFalse(state.exists())


if __name__ == "__main__":
    unittest.main()
