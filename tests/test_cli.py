import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from face_replace.__main__ import InputError, main, probe
from face_replace.pipeline import PipelineError, run_pipeline


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

    def test_inference_closed(self):
        self.assertEqual(self.run_cli([x for x in self.args() if x != "--dry-run"])[0], 3)

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
        worker.write_text(
            "import shutil, sys\n"
            "args = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n"
            "shutil.copyfile(args['--video'], args['--output'])\n"
        )
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


if __name__ == "__main__":
    unittest.main()
