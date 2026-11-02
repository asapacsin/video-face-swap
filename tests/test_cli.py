import argparse
import contextlib
import importlib.util
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from face_replace.__main__ import InputError, clock_seconds, main, probe
from face_replace.pipeline import PipelineError, run_pipeline


def load_worker_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "facefusion_worker.py"
    spec = importlib.util.spec_from_file_location("facefusion_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FAKE_WORKER = """
import json, subprocess, sys
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
if args["--mode"] == "scan":
    open(args["--state"] + ".png", "wb").write(b"preview")
    candidate = {"index": 0, "frame": 0, "position": 0, "appearances": 1, "time": "00:00:00",
                 "preview": args["--state"] + ".png"}
    json.dump({"done": True, "candidates": [candidate]}, open(args["--state"], "w"))
if args["--mode"] == "pick":
    open(args["--state"] + ".png", "wb").write(b"picked")
    json.dump({"frame": int(args["--reference-frame"]), "position": 0,
               "preview": args["--state"] + ".png"}, open(args["--state"], "w"))
if args["--mode"] == "swap":
    start, end = args["--start-frame"], args["--end-frame"]
    trim = f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", args["--video"], "-vf", trim, "-an",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", args["--output"]], check=True)
    open(args["--work"] + "/swaps.log", "a").write(start + "\\n")
"""


class CLITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.video = cls.root / "synthetic.mp4"
        cls.face = cls.root / "synthetic.png"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:s=64x64:r=24",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000",
                "-t",
                "1",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(cls.video),
            ],
            check=True,
            timeout=30,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=green:s=64x64",
                "-frames:v",
                "1",
                "-threads",
                "1",
                str(cls.face),
            ],
            check=True,
            timeout=30,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def args(self):
        return [
            "--video",
            str(self.video),
            "--face",
            str(self.face),
            "--output",
            str(self.root / "result.mp4"),
            "--consent-confirmed",
            "--dry-run",
        ]

    def run_cli(self, args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(args)
        return code, out.getvalue(), err.getvalue()

    def test_dry_run_real_media_no_writes(self):
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        code, output, _ = self.run_cli(self.args())
        self.assertEqual(code, 0)
        data = json.loads(output)
        self.assertEqual(data["source"]["audio_streams"], 1)
        self.assertEqual(data["source"]["frame_rate"], "24")
        self.assertEqual(data["source"]["width"], 64)
        self.assertFalse(data["output_written"])
        self.assertEqual(data["gpu_inference"], "NOT_VERIFIED")
        self.assertNotIn(str(self.root), output)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})

    def test_consent_required_before_probe(self):
        with patch("face_replace.__main__.probe") as mock:
            code, _, error = self.run_cli([x for x in self.args() if x != "--consent-confirmed"])
            self.assertEqual(code, 2)
            self.assertIn("attestation", error)
            mock.assert_not_called()

    def test_known_explicit_refused_before_probe(self):
        with patch("face_replace.__main__.probe") as mock:
            self.assertEqual(self.run_cli(self.args() + ["--content-rating", "explicit"])[0], 2)
            mock.assert_not_called()

    def test_missing_video(self):
        args = self.args()
        args[1] = str(self.root / "missing.mp4")
        self.assertEqual(self.run_cli(args)[0], 2)

    def test_no_input_overwrite(self):
        args = self.args()
        args[5] = str(self.video)
        self.assertEqual(self.run_cli(args)[0], 2)

    def test_output_extension(self):
        args = self.args()
        args[5] = str(self.root / "bad.avi")
        self.assertEqual(self.run_cli(args)[0], 2)

    def test_existing_output(self):
        args = self.args()
        existing = self.root / "existing.mp4"
        try:
            existing.write_bytes(b"preserve me")
            args[5] = str(existing)
            self.assertEqual(self.run_cli(args)[0], 2)
            self.assertEqual(existing.read_bytes(), b"preserve me")
        finally:
            existing.unlink()

    def test_corrupt_media(self):
        bad = self.root / "bad.png"
        try:
            bad.write_text("not media")
            with self.assertRaises(InputError):
                probe(bad, "reference image")
        finally:
            bad.unlink()

    def test_missing_ffprobe(self):
        with patch("face_replace.__main__.shutil.which", return_value=None):
            with self.assertRaisesRegex(InputError, "ffprobe"):
                probe(self.video, "source video")

    def test_probe_timeout(self):
        with patch(
            "face_replace.__main__.subprocess.run",
            side_effect=subprocess.TimeoutExpired("ffprobe", 30),
        ):
            with self.assertRaises(InputError):
                probe(self.video, "source video")

    def test_pipeline_failure_returns_3_with_message(self):
        with patch(
            "face_replace.pipeline.run_pipeline", side_effect=PipelineError("no target face")
        ):
            code, _, error = self.run_cli([x for x in self.args() if x != "--dry-run"])
        self.assertEqual(code, 3)
        self.assertIn("no target face", error)

    def test_default_output_is_beside_video(self):
        args = [x for x in self.args() if x != "--dry-run"]
        del args[4:6]
        with patch("face_replace.pipeline.run_pipeline", return_value={}) as mock:
            self.assertEqual(self.run_cli(args)[0], 0)
        self.assertEqual(
            mock.call_args.args[2], (self.root / "synthetic-swapped-720p.mp4").resolve()
        )

    def test_character_time_formats(self):
        self.assertEqual(clock_seconds("90.5"), 90.5)
        self.assertEqual(clock_seconds("01:02:03"), 3723.0)
        with self.assertRaises(argparse.ArgumentTypeError):
            clock_seconds("soon")

    def test_help_entrypoint(self):
        result = subprocess.run(
            [sys.executable, "-m", "face_replace", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--consent-confirmed", result.stdout)

    def test_pipeline_with_fake_worker_labels_and_retains_audio(self):
        output = self.root / "pipeline-result.mp4"
        worker = self.root / "fake_worker.py"
        worker.write_text(FAKE_WORKER)
        try:
            result = run_pipeline(self.video, self.face, output, worker)
            self.assertEqual(result["status"], "created")
            self.assertTrue(output.is_file())
            media = probe(output, "source video")
            self.assertEqual(media["audio_streams"], 1)
            self.assertEqual(media["frame_rate"], "24")
            metadata = subprocess.run(
                ["ffprobe", "-v", "error", "-show_format", "-of", "json", str(output)],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            self.assertIn("AI-ALTERED", metadata.stdout)
            frame = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(output), "-frames:v", "1", "-f", "null", "-"],
                capture_output=True,
                timeout=10,
            )
            self.assertEqual(frame.returncode, 0)
            self.assertGreater(output.stat().st_size, 0)
        finally:
            output.unlink(missing_ok=True)
            worker.unlink(missing_ok=True)

    def fake_worker(self):
        worker = self.root / "fake_worker.py"
        worker.write_text(FAKE_WORKER)
        return worker

    def test_pipeline_converts_mkv_movie_to_720p_mp4(self):
        movie = self.root / "movie.mkv"
        output = self.root / "movie-result.mp4"
        worker = self.fake_worker()
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:s=1920x1080:r=24",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000",
                "-t",
                "1",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                str(movie),
            ],
            check=True,
            timeout=60,
        )
        try:
            media = run_pipeline(movie, self.face, output, worker)["media"]
            self.assertEqual((media["width"], media["height"]), (1280, 720))
            self.assertEqual(media["audio_streams"], 1)
            # The overlay box must really be drawn, not only recorded as metadata.
            frame = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(output),
                    "-frames:v",
                    "1",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-",
                ],
                capture_output=True,
                check=True,
                timeout=30,
            ).stdout
            corner = frame[(20 * 1280 + 20) * 3 + 2]
            far = frame[(600 * 1280 + 900) * 3 + 2]
            self.assertGreater(far, 150)
            self.assertLess(corner, 150)
        finally:
            for path in (movie, output, worker):
                path.unlink(missing_ok=True)

    def test_pipeline_never_upscales(self):
        output = self.root / "small-result.mp4"
        worker = self.fake_worker()
        try:
            media = run_pipeline(self.video, self.face, output, worker)["media"]
            self.assertEqual((media["width"], media["height"]), (64, 64))
        finally:
            output.unlink(missing_ok=True)
            worker.unlink(missing_ok=True)

    def test_variable_frame_rate_source_is_accepted(self):
        movie = self.root / "vfr.mp4"
        output = self.root / "vfr-result.mp4"
        worker = self.fake_worker()
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:s=320x240:r=30",
                "-t",
                "2",
                "-vf",
                "select=not(eq(n\\,10))",
                "-fps_mode",
                "vfr",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(movie),
            ],
            check=True,
            timeout=30,
        )
        try:
            self.assertEqual(run_pipeline(movie, self.face, output, worker)["status"], "created")
        finally:
            for path in (movie, output, worker):
                path.unlink(missing_ok=True)

    def test_worker_refusal_message_reaches_user(self):
        output = self.root / "refused-result.mp4"
        worker = self.root / "refusing_worker.py"
        worker.write_text(
            "import sys\n"
            "print('face-replace-error: video screening refused', file=sys.stderr)\n"
            "raise SystemExit(2)\n"
        )
        try:
            with self.assertRaisesRegex(PipelineError, "video screening refused"):
                run_pipeline(self.video, self.face, output, worker)
            self.assertFalse(output.exists())
        finally:
            worker.unlink(missing_ok=True)

    def test_interrupted_job_resumes_without_redoing_segments(self):
        output = self.root / "resume-result.mp4"
        flaky = self.root / "flaky_worker.py"
        flag = self.root / "fail-once"
        attempts = self.root / "swap-attempts.log"
        flag.write_text("x")
        flaky.write_text(
            "import os, sys\n"
            f"flag = {str(flag)!r}\n"
            "at = sys.argv.index('--start-frame') + 1 if '--start-frame' in sys.argv else 0\n"
            "if sys.argv[at] == '12' and os.path.exists(flag):\n"
            "    print('face-replace-error: simulated stop', file=sys.stderr)\n"
            "    raise SystemExit(4)\n"
            "if at:\n"
            f"    open({str(attempts)!r}, 'a').write(sys.argv[at] + '\\n')\n" + FAKE_WORKER
        )
        try:
            with self.assertRaises(PipelineError):
                run_pipeline(self.video, self.face, output, flaky, segment_seconds=0.5)
            self.assertFalse(output.exists())
            job = next(self.root.glob(".face-replace-job-*"))
            self.assertEqual(attempts.read_text().split(), ["0"])
            flag.unlink()
            result = run_pipeline(self.video, self.face, output, flaky, segment_seconds=0.5)
            self.assertEqual(result["media"]["video_frames"], 24)
            self.assertEqual(result["media"]["audio_streams"], 1)
            # Segment 0 ran only once, so resuming reused it.
            self.assertEqual(attempts.read_text().split(), ["0", "12"])
            self.assertFalse(job.exists())
        finally:
            output.unlink(missing_ok=True)
            output.with_name("resume-result-character.png").unlink(missing_ok=True)
            flaky.unlink(missing_ok=True)
            attempts.unlink(missing_ok=True)

    def test_changed_inputs_or_code_do_not_reuse_a_job(self):
        output = self.root / "changed-result.mp4"
        worker = self.fake_worker()
        other_face = self.root / "other-face.png"
        other_face.write_bytes(self.face.read_bytes() + b"x")
        other_worker = self.root / "other_worker.py"
        other_worker.write_text(worker.read_text() + "# new version\n")
        before = set(self.root.glob(".face-replace-job-*"))
        try:
            run_pipeline(self.video, self.face, output, worker, scan_only=True)
            first = set(self.root.glob(".face-replace-job-*")) - before
            self.assertEqual(len(first), 1)
            run_pipeline(self.video, other_face, output, worker, scan_only=True)
            run_pipeline(self.video, self.face, output, other_worker, scan_only=True)
            self.assertEqual(len(set(self.root.glob(".face-replace-job-*")) - before), 3)
            # A changed movie is a new job, and a character confirmed in the old one is refused.
            first_name = first.copy().pop().name
            stat = self.video.stat()
            os.utime(self.video, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
            try:
                with self.assertRaisesRegex(InputError, "changed after"):
                    run_pipeline(self.video, self.face, output, worker, confirmed_job=first_name)
                run_pipeline(self.video, self.face, output, worker, scan_only=True)
            finally:
                os.utime(self.video, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            self.assertEqual(len(set(self.root.glob(".face-replace-job-*")) - before), 4)
        finally:
            for job in set(self.root.glob(".face-replace-job-*")) - before:
                shutil.rmtree(job)
            for path in (worker, other_face, other_worker):
                path.unlink(missing_ok=True)

    def test_screening_refusal_removes_its_media_and_publishes_nothing(self):
        output = self.root / "refused-media.mp4"
        worker = self.root / "screening_worker.py"
        worker.write_text(
            "import sys\n"
            "print('face-replace-error: video screening refused the frame', file=sys.stderr)\n"
            "raise SystemExit(2)\n"
        )
        before = set(self.root.glob(".face-replace-job-*"))
        try:
            with self.assertRaisesRegex(PipelineError, "screening refused"):
                run_pipeline(self.video, self.face, output, worker)
            self.assertEqual(set(self.root.glob(".face-replace-job-*")), before)
            self.assertFalse(output.with_name("refused-media-character.png").exists())
        finally:
            worker.unlink(missing_ok=True)

    def test_other_final_refusals_remove_media_and_setup_problems_keep_it(self):
        output = self.root / "refusal-kinds.mp4"
        worker = self.root / "kinds_worker.py"
        before = set(self.root.glob(".face-replace-job-*"))
        script = (
            "import sys\n"
            "print('face-replace-error: no target face detected', file=sys.stderr)\n"
            "raise SystemExit({})\n"
        )
        try:
            worker.write_text(script.format(4))
            with self.assertRaisesRegex(PipelineError, "no target face") as caught:
                run_pipeline(self.video, self.face, output, worker)
            self.assertFalse(caught.exception.refused)
            self.assertEqual(len(set(self.root.glob(".face-replace-job-*")) - before), 1)
            worker.write_text(script.format(2))
            with self.assertRaisesRegex(PipelineError, "no target face"):
                run_pipeline(self.video, self.face, output, worker)
            self.assertEqual(len(set(self.root.glob(".face-replace-job-*")) - before), 1)
        finally:
            for job in set(self.root.glob(".face-replace-job-*")) - before:
                shutil.rmtree(job)
            worker.unlink(missing_ok=True)

    def test_raw_worker_output_is_neither_shown_nor_stored(self):
        output = self.root / "quiet-result.mp4"
        worker = self.root / "noisy_worker.py"
        worker.write_text(
            "import sys\nprint('/private/secret.mp4', file=sys.stderr)\nraise SystemExit(1)\n"
        )
        before = set(self.root.glob(".face-replace-job-*"))
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err), self.assertRaises(PipelineError):
                run_pipeline(self.video, self.face, output, worker)
            self.assertNotIn("secret", err.getvalue())
            for job in set(self.root.glob(".face-replace-job-*")) - before:
                for path in job.rglob("*"):
                    if path.suffix != ".mp4" and path.is_file():
                        self.assertNotIn(b"secret", path.read_bytes())
        finally:
            for job in set(self.root.glob(".face-replace-job-*")) - before:
                shutil.rmtree(job)
            worker.unlink(missing_ok=True)

    def test_a_second_run_of_the_same_job_is_refused(self):
        from face_replace.pipeline import _lock_job, job_directory

        output = self.root / "locked-result.mp4"
        worker = self.fake_worker()
        job = job_directory(
            self.video.resolve(), self.face.resolve(), output.resolve(), worker.resolve()
        )
        job.mkdir()
        lock = _lock_job(job)
        try:
            with self.assertRaisesRegex(InputError, "already running"):
                run_pipeline(self.video, self.face, output, worker)
        finally:
            lock.close()
            shutil.rmtree(job)
            worker.unlink(missing_ok=True)

    def test_named_time_picks_that_frame_and_passes_it_to_the_swap(self):
        output = self.root / "timed-result.mp4"
        calls = self.root / "timed-calls.log"
        worker = self.root / "timed_worker.py"
        worker.write_text(
            "import sys\n"
            f"open({str(calls)!r}, 'a').write(' '.join(sys.argv[1:]) + '\\n')\n" + FAKE_WORKER
        )
        preview = output.with_name("timed-result-character.png")
        try:
            with self.assertRaisesRegex(InputError, "past the end"):
                run_pipeline(self.video, self.face, output, worker, character_time=5.0)
            run_pipeline(self.video, self.face, output, worker, character_time=0.5)
            lines = calls.read_text().splitlines()
            picks = [line for line in lines if "--mode pick" in line]
            swaps = [line for line in lines if "--mode swap" in line]
            self.assertEqual(len(picks), 1)
            self.assertIn("--reference-frame 12", picks[0])
            self.assertIn("--reference-frame 12", swaps[0])
            self.assertIn("--reference-position 0", swaps[0])
            self.assertEqual(preview.read_bytes(), b"picked")
        finally:
            for job in self.root.glob(".face-replace-job-*"):
                shutil.rmtree(job)
            for path in (output, preview, calls, worker):
                path.unlink(missing_ok=True)

    def test_named_time_with_no_face_keeps_the_job(self):
        output = self.root / "noface-result.mp4"
        worker = self.root / "noface_worker.py"
        worker.write_text(
            "import sys\n"
            "if 'pick' in sys.argv:\n"
            "    print('face-replace-error: no face detected', file=sys.stderr)\n"
            "    raise SystemExit(4)\n" + FAKE_WORKER
        )
        before = set(self.root.glob(".face-replace-job-*"))
        try:
            with self.assertRaisesRegex(PipelineError, "no face detected"):
                run_pipeline(self.video, self.face, output, worker, character_time=0.5)
            jobs = set(self.root.glob(".face-replace-job-*")) - before
            self.assertEqual(len(jobs), 1)
            self.assertTrue((jobs.pop() / "scan.json").is_file())
        finally:
            for job in set(self.root.glob(".face-replace-job-*")) - before:
                shutil.rmtree(job)
            worker.unlink(missing_ok=True)

    def test_failed_job_removal_is_reported(self):
        output = self.root / "stuck-result.mp4"
        worker = self.root / "stuck_worker.py"
        worker.write_text(
            "import sys\n"
            "print('face-replace-error: video screening refused', file=sys.stderr)\n"
            "raise SystemExit(2)\n"
        )
        before = set(self.root.glob(".face-replace-job-*"))
        err = io.StringIO()
        try:
            with (
                patch("face_replace.pipeline.shutil.rmtree", side_effect=OSError),
                contextlib.redirect_stderr(err),
                self.assertRaises(PipelineError),
            ):
                run_pipeline(self.video, self.face, output, worker)
            self.assertIn("could not remove", err.getvalue())
            self.assertNotIn(str(self.root), err.getvalue())
        finally:
            for job in set(self.root.glob(".face-replace-job-*")) - before:
                shutil.rmtree(job)
            worker.unlink(missing_ok=True)

    def test_character_preview_never_overwrites_and_waits_for_the_result(self):
        output = self.root / "preview-result.mp4"
        preview = self.root / "preview-result-character.png"
        preview.write_bytes(b"keep")
        worker = self.fake_worker()
        try:
            run_pipeline(self.video, self.face, output, worker)
            self.assertEqual(preview.read_bytes(), b"keep")
        finally:
            for path in (output, preview, worker):
                path.unlink(missing_ok=True)

    def test_scan_prints_no_absolute_path(self):
        output = self.root / "private-result.mp4"
        worker = self.fake_worker()
        before = set(self.root.glob(".face-replace-job-*"))
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                run_pipeline(self.video, self.face, output, worker, scan_only=True)
            self.assertIn("face-replace-candidates", err.getvalue())
            self.assertNotIn(str(self.root), err.getvalue())
        finally:
            for job in set(self.root.glob(".face-replace-job-*")) - before:
                shutil.rmtree(job)
            worker.unlink(missing_ok=True)

    def test_scan_only_lists_characters_without_swapping(self):
        output = self.root / "scan-result.mp4"
        worker = self.fake_worker()
        try:
            result = run_pipeline(self.video, self.face, output, worker, scan_only=True)
            self.assertEqual(result["status"], "scanned")
            self.assertEqual(result["candidates"][0]["index"], 0)
            self.assertFalse(output.exists())
            with self.assertRaises(InputError):
                run_pipeline(self.video, self.face, output, worker, character=9)
        finally:
            for job in self.root.glob(".face-replace-job-*"):
                shutil.rmtree(job)
            worker.unlink(missing_ok=True)

    def test_too_little_disk_is_refused_before_work(self):
        output = self.root / "disk-result.mp4"
        worker = self.fake_worker()
        usage = shutil.disk_usage(self.root)._replace(free=1000)
        try:
            with patch("face_replace.pipeline.shutil.disk_usage", return_value=usage):
                with self.assertRaisesRegex(InputError, "disk space"):
                    run_pipeline(self.video, self.face, output, worker)
            # A scan's job folder must not skip the check before the larger swap.
            run_pipeline(self.video, self.face, output, worker, scan_only=True)
            with patch("face_replace.pipeline.shutil.disk_usage", return_value=usage):
                with self.assertRaisesRegex(InputError, "disk space"):
                    run_pipeline(self.video, self.face, output, worker)
        finally:
            for job in self.root.glob(".face-replace-job-*"):
                shutil.rmtree(job)
            worker.unlink(missing_ok=True)

    def test_cpu_fallback_is_forbidden_at_load(self):
        class Session:
            providers = ["CPUExecutionProvider"]
            disabled = False

            def __init__(self):
                pass

            def get_providers(self):
                return self.providers

            def disable_fallback(self):
                type(self).disabled = True

        load_worker_module().forbid_cpu_fallback(Session)
        with self.assertRaises(RuntimeError):
            Session()
        Session.providers = ["CUDAExecutionProvider"]
        Session()
        self.assertTrue(Session.disabled)

    def test_gpu_failures_reach_the_owner_with_their_reason(self):
        from face_replace.pipeline import _run

        script = self.root / "gpu_failure.py"
        worker = load_worker_module()
        try:
            # Upstream catches the load failure and exits, as this script does.
            script.write_text(
                "import importlib.util, sys\n"
                f"spec = importlib.util.spec_from_file_location('w', {str(worker.__file__)!r})\n"
                "w = importlib.util.module_from_spec(spec)\n"
                "spec.loader.exec_module(w)\n"
                "class Session:\n"
                "    def __init__(self): pass\n"
                "    def get_providers(self): return ['CPUExecutionProvider']\n"
                "w.forbid_cpu_fallback(Session)\n"
                "try:\n"
                "    Session()\n"
                "except RuntimeError:\n"
                "    raise SystemExit(1)\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PipelineError, "did not load on the GPU") as caught:
                _run([sys.executable, str(script)], 60)
            self.assertFalse(caught.exception.refused)
            script.write_text(
                "import sys\n"
                "print('RuntimeError: CUDA failure 700: Illegal address', file=sys.stderr)\n"
                "raise SystemExit(1)\n"
            )
            with self.assertRaisesRegex(PipelineError, "CUDA error") as caught:
                _run([sys.executable, str(script)], 60)
            self.assertFalse(caught.exception.refused)
        finally:
            script.unlink(missing_ok=True)

    def test_file_errors_are_reported_without_private_paths(self):
        err = io.StringIO()
        args = ["--video", str(self.video), "--face", str(self.face), "--consent-confirmed"]
        args += ["--output", str(self.root / "unwritable.mp4")]
        with (
            patch("face_replace.pipeline.run_pipeline", side_effect=OSError("/private/job")),
            patch("face_replace.__main__.stop_workers_on_signals"),
            contextlib.redirect_stderr(err),
        ):
            self.assertEqual(main(args), 3)
        self.assertIn("could not read or write", err.getvalue())
        self.assertNotIn("/private/job", err.getvalue())

    def test_stop_signals_end_the_worker_through_the_interrupt_path(self):
        from face_replace.__main__ import stop_workers_on_signals

        handlers = {}
        with patch("signal.signal", lambda number, handler: handlers.update({number: handler})):
            stop_workers_on_signals()
        self.assertIn(signal.SIGTERM, handlers)
        with self.assertRaises(KeyboardInterrupt):
            handlers[signal.SIGTERM](signal.SIGTERM, None)

    def test_pipeline_worker_failure_publishes_no_output(self):
        output = self.root / "failed-result.mp4"
        worker = self.root / "failing_worker.py"
        worker.write_text("raise SystemExit(1)\n")
        try:
            with self.assertRaises(PipelineError):
                run_pipeline(self.video, self.face, output, worker)
            self.assertFalse(output.exists())
        finally:
            worker.unlink(missing_ok=True)


class Face:
    bounding_box = (0, 0, 10, 10)
    embedding_norm = (1.0, 0.0)


class WorkerScanTest(unittest.TestCase):
    """The worker's scan is the only every-frame screening, so it is tested without a GPU."""

    def setUp(self):
        self.worker = load_worker_module()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name) / "scan.json"

    def run_scan(self, frames=6, total=None, refused=(), fps=0.01):
        numpy = __import__("numpy")
        frames_left = iter(range(frames))
        self.screened = []
        self.seeks = []

        def analyse(frame):
            self.screened.append(frame)
            return frame in refused

        ff = argparse.Namespace(
            analyse_frame=analyse,
            close_video_reader=lambda reader: None,
            count_video_frame_total=lambda video: frames if total is None else total,
            detect_video_fps=lambda video: fps,
            get_reader=lambda video, name: "reader",
            read_video_frame=lambda reader: next(frames_left, None),
            save_preview=lambda frame, box, path: None,
            seek_video_reader=self.seek(frames_left),
            target_faces=lambda frame: [Face()],
        )
        args = argparse.Namespace(state=str(self.state), video="movie")
        with contextlib.redirect_stderr(io.StringIO()) as err:
            code = self.worker.scan(args, ff, numpy)
        return code, err.getvalue()

    def seek(self, frames_left):
        def seek(reader, frame):
            self.seeks.append(frame)
            for _ in range(frame):
                next(frames_left)

        return seek

    def test_every_frame_is_screened_and_embeddings_end_with_the_scan(self):
        code, _ = self.run_scan()
        self.assertEqual(code, 0)
        self.assertEqual(self.screened, list(range(6)))
        state = json.loads(self.state.read_text())
        self.assertTrue(state["done"])
        self.assertNotIn("embedding", self.state.read_text())
        self.assertEqual(state["candidates"][0]["appearances"], 6)

    def test_a_refused_frame_stops_the_job_as_a_final_refusal(self):
        code, err = self.run_scan(refused={5})
        self.assertEqual(code, 2)
        self.assertIn("video screening refused", err)
        self.assertFalse(json.loads(self.state.read_text()).get("done"))

    def test_a_checkpoint_never_covers_an_unscreened_or_refused_frame(self):
        # With a checkpoint every 3 frames, the refusal in frame 2 comes before the first save.
        code, _ = self.run_scan(refused={2})
        self.assertEqual(code, 2)
        self.assertFalse(self.state.exists())

    def test_resume_seeks_to_the_checkpoint_and_screens_the_rest(self):
        self.state.write_text(
            json.dumps(
                {
                    "next_frame": 3,
                    "clusters": [
                        {
                            "count": 2,
                            "area": 1.0,
                            "frame": 0,
                            "position": 0,
                            "box": [0, 0, 1, 1],
                            "embedding": [1.0, 0.0],
                        }
                    ],
                }
            )
        )
        code, _ = self.run_scan()
        self.assertEqual(code, 0)
        self.assertEqual(self.seeks, [3])
        self.assertEqual(self.screened, [3, 4, 5])
        self.assertEqual(json.loads(self.state.read_text())["candidates"][0]["appearances"], 5)

    def test_short_decode_and_extra_frames_are_refused(self):
        code, err = self.run_scan(frames=4, total=6)
        self.assertEqual(code, 2)
        self.assertIn("could not be fully decoded", err)
        self.state.unlink(missing_ok=True)
        code, err = self.run_scan(frames=7, total=6)
        self.assertEqual(code, 2)
        self.assertIn("frame count changed", err)

    def test_swap_refusal_is_final_and_a_crash_can_resume(self):
        args = argparse.Namespace(video="movie", start_frame=10, end_frame=20)
        calls = []

        def analyse(video, start, end):
            calls.append((video, start, end))
            return True

        with contextlib.redirect_stderr(io.StringIO()) as err:
            code = self.worker.swap(args, analyse, lambda parsed: self.fail("swapped"), {})
        self.assertEqual(code, 2)
        self.assertEqual(calls, [("movie", 10, 20)])
        self.assertIn("screening refused", err.getvalue())
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.worker.swap(args, lambda *a: False, lambda p: 1, {}), 4)
            self.assertEqual(self.worker.swap(args, lambda *a: False, lambda p: 0, {}), 0)

    def test_screening_models_are_freed_without_changing_the_result(self):
        for refused in (True, False):
            calls = []
            analyser = argparse.Namespace(
                analyse_video=lambda *args, r=refused: calls.append(args) or r,
                clear_inference_pool=lambda: calls.append("cleared"),
            )
            self.worker.free_screening_after_analysis(analyser)
            self.assertEqual(analyser.analyse_video("movie", 1, 2), refused)
            self.assertEqual(calls, [("movie", 1, 2), "cleared"])

    def test_pick_uses_the_largest_face_or_stops_without_refusing(self):
        saved = []
        ff = argparse.Namespace(
            read_frame=lambda video, frame: "frame",
            target_faces=lambda frame: [Face()],
            save_preview=lambda frame, box, path: saved.append(frame),
        )
        args = argparse.Namespace(video="movie", reference_frame=7, state=str(self.state))
        self.assertEqual(self.worker.pick(args, ff), 0)
        state = json.loads(self.state.read_text())
        self.assertEqual((state["frame"], state["position"], saved), (7, 0, [7]))
        ff.target_faces = lambda frame: []
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.worker.pick(args, ff), 4)

    def test_clock_does_not_wrap_at_24_hours(self):
        self.assertEqual(self.worker.clock(25 * 3600 + 61), "25:01:01")


if __name__ == "__main__":
    unittest.main()
