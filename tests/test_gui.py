import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("tkinter"), "tkinter is not installed")
class GuiHelpersTest(unittest.TestCase):
    def test_default_output_never_reuses_an_existing_file(self):
        from face_replace.gui import default_output

        with tempfile.TemporaryDirectory() as temp:
            movie = Path(temp) / "movie.mkv"
            self.assertEqual(default_output(movie).name, "movie-swapped-720p.mp4")
            (Path(temp) / "movie-swapped-720p.mp4").write_bytes(b"keep")
            self.assertEqual(default_output(movie).name, "movie-swapped-720p-2.mp4")

    def test_command_carries_inputs_and_optional_character_time(self):
        from face_replace.gui import build_command

        command = build_command(Path("m.mkv"), Path("p.jpg"), Path("o.mp4"))
        self.assertIn("--consent-confirmed", command)
        self.assertNotIn("--character-time", command)
        self.assertEqual(command[command.index("--video") + 1], "m.mkv")
        timed = build_command(Path("m.mkv"), Path("p.jpg"), Path("o.mp4"), "00:12:34")
        self.assertEqual(timed[timed.index("--character-time") + 1], "00:12:34")

    def test_only_status_and_error_lines_are_shown(self):
        from face_replace.gui import display_line

        self.assertEqual(display_line("face-replace: screening 4.0%\n"), "screening 4.0%")
        self.assertEqual(display_line("error: no target face\n"), "error: no target face")
        self.assertIsNone(display_line('  "output": "D:\\\\private\\\\result.mp4",\n'))

    def test_elapsed_format(self):
        from face_replace.gui import format_elapsed

        self.assertEqual(format_elapsed(3723.9), "01:02:03")


if __name__ == "__main__":
    unittest.main()
