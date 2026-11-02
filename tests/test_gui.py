import importlib.util
import sys
import tempfile
import time
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

    def test_scan_and_character_options_reach_the_command(self):
        from face_replace.gui import build_command

        scan = build_command(Path("m"), Path("p"), Path("o"), scan_only=True)
        self.assertIn("--scan-only", scan)
        swap = build_command(Path("m"), Path("p"), Path("o"), character=2)
        self.assertEqual(swap[swap.index("--character") + 1], "2")
        self.assertNotIn("--scan-only", swap)

    def test_candidates_and_percent_parsing(self):
        from face_replace.gui import parse_candidates, parse_percent

        self.assertEqual(
            parse_candidates('face-replace-candidates: [{"index": 0}]\n'), [{"index": 0}]
        )
        self.assertIsNone(parse_candidates("face-replace: scanning\n"))
        self.assertEqual(parse_percent("swapped segment 1/8, 12.5%, 01:00:00 left"), 12.5)
        self.assertIsNone(parse_percent("converting"))
        # An upstream bar restarts per stage and must not move the overall bar.
        self.assertIsNone(parse_percent("processing: 12%|=  | 6/50"))

    def test_elapsed_format(self):
        from face_replace.gui import format_elapsed

        self.assertEqual(format_elapsed(3723.9), "01:02:03")

    def test_swap_needs_a_confirmed_character_and_uses_it(self):
        import tkinter as tk
        from unittest import mock

        from face_replace import gui

        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest("no display")
        self.addCleanup(root.destroy)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            preview = folder / "preview.png"
            tk.PhotoImage(master=root, width=8, height=8).write(str(preview), format="png")
            candidates = [
                {"preview": str(preview), "appearances": 9, "time": "00:00:01"},
                {"preview": str(preview), "appearances": 4, "time": "00:00:07"},
            ]
            job = folder / "job.py"
            job.write_text(
                "import json, sys\n"
                f"print('face-replace-candidates: ' + json.dumps({candidates!r}))\n"
                "print('face-replace: swapped segment 1/2, 50.0%, 00:01:00 left')\n"
            )
            movie = folder / "m.mp4"
            photo = folder / "p.jpg"
            movie.write_bytes(b"m")
            photo.write_bytes(b"p")
            commands = []

            def fake_command(*args, **kwargs):
                commands.append(gui_build(*args, **kwargs))
                return [sys.executable, str(job)]

            gui_build = gui.build_command
            with (
                mock.patch.object(gui, "build_command", fake_command),
                mock.patch.object(gui.messagebox, "showinfo"),
            ):
                app = gui.App(root)
                app.movie.set(str(movie))
                app.photo.set(str(photo))
                app.output.set(str(folder / "o.mp4"))
                app.consent.set(True)
                app.refresh_buttons()
                self.assertTrue(app.start_button.instate(["disabled"]))

                def run(action):
                    action()
                    deadline = time.monotonic() + 30
                    while app.process is not None and time.monotonic() < deadline:
                        root.update()
                        time.sleep(0.01)
                    self.assertIsNone(app.process)

                def dialog_button(dialog, text):
                    return next(
                        w
                        for w in dialog.winfo_children()
                        if w.winfo_class() == "TButton" and w.cget("text") == text
                    )

                run(app.scan)
                dialog = next(w for w in root.winfo_children() if isinstance(w, tk.Toplevel))
                # Closing the dialog is not a confirmation.
                dialog.destroy()
                app.refresh_buttons()
                self.assertTrue(app.start_button.instate(["disabled"]))

                run(app.scan)
                dialog = next(w for w in root.winfo_children() if isinstance(w, tk.Toplevel))
                self.assertIn("1 of 2", dialog.winfo_children()[1].cget("text"))
                dialog_button(dialog, "Next character").invoke()
                dialog_button(dialog, "Replace this one").invoke()
                self.assertFalse(app.start_button.instate(["disabled"]))

                run(app.start)
                self.assertEqual(commands[-1][commands[-1].index("--character") + 1], "1")
                # The number is tied to the scan it was confirmed in.
                self.assertEqual(
                    commands[-1][commands[-1].index("--confirmed-job") + 1],
                    gui.job_directory(
                        movie.resolve(), photo.resolve(), (folder / "o.mp4").resolve(), gui.WORKER
                    ).name,
                )
                self.assertNotIn("--scan-only", commands[-1])
                self.assertEqual(app.progress["value"], 100)
                self.assertIn("left", app.log.get("1.0", "end"))
                self.assertTrue(app.start_button.instate(["disabled"]))

                # Any changed input, even typed, needs a new confirmation.
                app.consent.set(True)
                run(app.scan)
                dialog = next(w for w in root.winfo_children() if isinstance(w, tk.Toplevel))
                self.assertEqual(dialog.grab_current(), dialog)
                dialog_button(dialog, "Replace this one").invoke()
                self.assertFalse(app.start_button.instate(["disabled"]))
                app.photo.set(str(photo) + " ")
                self.assertTrue(app.start_button.instate(["disabled"]))
                # Consent was for the old files, so even a named time cannot start or scan.
                self.assertFalse(app.consent.get())
                app.character_time.set("00:00:01")
                self.assertTrue(app.start_button.instate(["disabled"]))
                self.assertTrue(app.scan_button.instate(["disabled"]))

    def test_a_job_error_reaches_the_owner_with_a_notice_about_kept_media(self):
        import tkinter as tk
        from unittest import mock

        from face_replace import gui

        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest("no display")
        self.addCleanup(root.destroy)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            job_dir = folder / ".face-replace-job-kept"
            job = folder / "job.py"
            job.write_text(
                "print('face-replace-job: .face-replace-job-kept', flush=True)\n"
                "print('error: not enough free disk space: need about 9 GB', flush=True)\n"
                "raise SystemExit(2)\n"
            )
            movie = folder / "m.mp4"
            photo = folder / "p.jpg"
            movie.write_bytes(b"m")
            photo.write_bytes(b"p")
            with (
                mock.patch.object(gui, "build_command", lambda *a, **k: [sys.executable, str(job)]),
                mock.patch.object(gui.messagebox, "showerror") as showerror,
            ):
                app = gui.App(root)
                app.movie.set(str(movie))
                app.photo.set(str(photo))
                app.output.set(str(folder / "o.mp4"))
                for kept in (False, True):
                    if kept:
                        job_dir.mkdir()
                    app.consent.set(True)
                    app.run_job(scan_only=True)
                    deadline = time.monotonic() + 30
                    while app.process is not None and time.monotonic() < deadline:
                        root.update()
                        time.sleep(0.01)
                    message = showerror.call_args.args[1]
                    self.assertIn("not enough free disk space", message)
                    self.assertEqual("private media" in message, kept)
                    self.assertFalse(app.consent.get())

    def test_cancel_or_close_removing_progress_deletes_a_scanned_job(self):
        import tkinter as tk
        from unittest import mock

        from face_replace import gui

        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest("no display")
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            job_dir = folder / ".face-replace-job-abc"
            other = folder / ".face-replace-job-other"
            other.mkdir()
            job = folder / "job.py"
            job.write_text(
                "import time\n"
                "print('face-replace-job: .face-replace-job-abc', flush=True)\n"
                "time.sleep(60)\n"
            )
            movie = folder / "m.mp4"
            photo = folder / "p.jpg"
            movie.write_bytes(b"m")
            photo.write_bytes(b"p")
            with (
                mock.patch.object(gui, "build_command", lambda *a, **k: [sys.executable, str(job)]),
                mock.patch.object(gui.messagebox, "askyesno", return_value=False),
            ):
                app = gui.App(root)
                app.movie.set(str(movie))
                app.photo.set(str(photo))
                app.output.set(str(folder / "o.mp4"))
                app.consent.set(True)

                def start_job():
                    # The folder exists before the run, as it does after a scan.
                    job_dir.mkdir()
                    (job_dir / "source-720p.mp4").write_bytes(b"media")
                    app.consent.set(True)
                    app.run_job(scan_only=False)
                    deadline = time.monotonic() + 30
                    while app.job_name != job_dir.name and time.monotonic() < deadline:
                        root.update()
                        time.sleep(0.01)
                    self.assertEqual(app.job_name, job_dir.name)

                start_job()
                app.cancel()
                deadline = time.monotonic() + 30
                while app.process is not None and time.monotonic() < deadline:
                    root.update()
                    time.sleep(0.01)
                self.assertFalse(job_dir.exists())
                self.assertIn("removed", app.status.get())
                # A cancelled job needs consent again before it runs.
                self.assertFalse(app.consent.get())

                # A folder that cannot be deleted is reported, not called removed.
                start_job()
                app.cancel()
                with mock.patch.object(gui.shutil, "rmtree", side_effect=OSError):
                    deadline = time.monotonic() + 30
                    while app.process is not None and time.monotonic() < deadline:
                        root.update()
                        time.sleep(0.01)
                self.assertIn("could not be deleted", app.status.get())
                gui.shutil.rmtree(job_dir)

                # Cancelling a resumed job before it reports its folder still removes that folder.
                resumed = gui.job_directory(
                    movie.resolve(), photo.resolve(), (folder / "o.mp4").resolve(), gui.WORKER
                )
                resumed.mkdir()
                (resumed / "source-720p.mp4").write_bytes(b"media")
                job.write_text("import time\ntime.sleep(60)\n")
                app.consent.set(True)
                app.run_job(scan_only=False)
                app.cancel()
                deadline = time.monotonic() + 30
                while app.process is not None and time.monotonic() < deadline:
                    root.update()
                    time.sleep(0.01)
                self.assertFalse(resumed.exists())
                self.assertIn("removed", app.status.get())
                job.write_text(
                    "import time\n"
                    "print('face-replace-job: .face-replace-job-abc', flush=True)\n"
                    "time.sleep(60)\n"
                )

                start_job()
                # Inputs are locked while a job runs, so its results cannot go to other files.
                self.assertTrue(app.input_widgets)
                self.assertTrue(all(w.instate(["disabled"]) for w in app.input_widgets))
                # Closing asks to stop the job, then whether to keep its progress.
                gui.messagebox.askyesno.side_effect = [True, False]
                app.close()
                self.assertFalse(job_dir.exists())
                self.assertTrue(other.exists())


if __name__ == "__main__":
    unittest.main()
