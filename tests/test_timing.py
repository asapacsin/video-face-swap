import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_cli import FAKE_WORKER

from face_replace.__main__ import probe
from face_replace.pipeline import PipelineError, _timing_matches, run_pipeline

FRAME = 1 / 25


def media(duration=10.0, video_start=0.0, audio_start=0.0, audio_streams=1):
    return {
        "duration_seconds": duration,
        "video_start": video_start,
        "audio_start": audio_start,
        "audio_streams": audio_streams,
    }


class TimingTest(unittest.TestCase):
    def test_identical_timing_matches(self):
        self.assertTrue(_timing_matches(media(), media(), FRAME))

    def test_duration_limit_is_one_frame(self):
        self.assertTrue(_timing_matches(media(duration=10.0 + FRAME), media(), FRAME))
        self.assertFalse(_timing_matches(media(duration=10.0 + FRAME * 1.01), media(), FRAME))

    def test_audio_offset_limit_is_one_frame(self):
        self.assertTrue(_timing_matches(media(audio_start=FRAME), media(), FRAME))
        self.assertFalse(_timing_matches(media(audio_start=FRAME * 1.01), media(), FRAME))

    def test_source_offset_is_preserved_not_forced_to_zero(self):
        source = media(video_start=0.0, audio_start=0.2)
        self.assertTrue(_timing_matches(media(video_start=1.0, audio_start=1.2), source, FRAME))
        self.assertFalse(_timing_matches(media(audio_start=0.0), source, FRAME))

    def test_any_missing_start_time_fails_when_the_source_has_audio(self):
        for key in ("audio_start", "video_start"):
            missing = media(**{key: None})
            with self.subTest(side="output", key=key):
                self.assertFalse(_timing_matches(missing, media(), FRAME))
            with self.subTest(side="source", key=key):
                self.assertFalse(_timing_matches(media(), missing, FRAME))

    def test_silence_is_decided_by_stream_count_not_start_time(self):
        silent = media(audio_start=None, audio_streams=0)
        self.assertTrue(_timing_matches(media(audio_start=None, audio_streams=0), silent, FRAME))
        self.assertFalse(_timing_matches(media(duration=11.0), silent, FRAME))


def ffprobe_output(streams):
    data = {"streams": streams, "format": {"duration": "10.0"}}
    return subprocess.CompletedProcess([], 0, stdout=json.dumps(data), stderr="")


VIDEO = {"codec_type": "video", "width": 64, "height": 64, "avg_frame_rate": "25/1"}


class ProbeStartTimeTest(unittest.TestCase):
    def probe_streams(self, streams):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as file:
            with patch(
                "face_replace.__main__.subprocess.run", return_value=ffprobe_output(streams)
            ):
                return probe(Path(file.name), "source video")

    def test_start_times_are_recorded_from_the_first_audio_stream(self):
        result = self.probe_streams(
            [
                {**VIDEO, "start_time": "0.040000"},
                {"codec_type": "audio", "start_time": "0.021333"},
                {"codec_type": "audio", "start_time": "5.0"},
            ]
        )
        self.assertAlmostEqual(result["video_start"], 0.04)
        self.assertAlmostEqual(result["audio_start"], 0.021333)
        self.assertEqual(result["audio_streams"], 2)

    def test_absent_or_unreadable_start_times_are_none(self):
        for value in (None, "N/A", "nan", "inf"):
            stream = (
                {"codec_type": "audio"}
                if value is None
                else {"codec_type": "audio", "start_time": value}
            )
            with self.subTest(value=value):
                result = self.probe_streams([{**VIDEO, "start_time": value or "N/A"}, stream])
                self.assertIsNone(result["audio_start"])
                self.assertIsNone(result["video_start"])

    def test_no_audio_stream_gives_no_audio_start(self):
        result = self.probe_streams([{**VIDEO, "start_time": "0"}])
        self.assertIsNone(result["audio_start"])
        self.assertEqual(result["audio_streams"], 0)


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is not installed")
class PublishRefusalTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video, self.face = self.root / "in.mp4", self.root / "face.png"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=64x64:r=24",
             "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(self.video)],
            check=True, timeout=30,
        )  # fmt: skip
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=green:s=64x64",
             "-frames:v", "1", str(self.face)],
            check=True, timeout=30,
        )  # fmt: skip
        self.worker = self.root / "fake_worker.py"
        self.worker.write_text(FAKE_WORKER)

    def tearDown(self):
        self.temp.cleanup()

    def run_with_output_changed(self, **change):
        def altered_probe(path, kind):
            result = probe(path, kind)
            return {**result, **change} if kind == "generated output" else result

        output = self.root / "result.mp4"
        with patch("face_replace.pipeline.probe", side_effect=altered_probe):
            with self.assertRaisesRegex(PipelineError, "duration or audio sync"):
                run_pipeline(self.video, self.face, output, self.worker)
        self.assertFalse(output.exists())
        published = {p.name for p in self.root.iterdir() if not p.name.startswith(".")}
        self.assertEqual(published, {"in.mp4", "face.png", "fake_worker.py"})

    def test_shifted_audio_publishes_nothing(self):
        self.run_with_output_changed(audio_start=0.5)

    def test_duration_off_by_more_than_one_frame_publishes_nothing(self):
        self.run_with_output_changed(duration_seconds=2.0)

    def test_missing_audio_start_publishes_nothing(self):
        self.run_with_output_changed(audio_start=None)


if __name__ == "__main__":
    unittest.main()
