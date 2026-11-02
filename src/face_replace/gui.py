"""Small desktop window around the face_replace command line."""

import argparse
import ctypes
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from face_replace.__main__ import clock_seconds
from face_replace.pipeline import job_directory

MOVIE_TYPES = [("Movies", "*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.ts"), ("All files", "*.*")]
PHOTO_TYPES = [("Photos", "*.jpg *.jpeg *.png *.webp")]
CONSENT_TEXT = (
    "I confirm everyone whose identity is used is an adult and has agreed to this use. "
    "The movie and photo are non-explicit and I am allowed to use them. "
    "The result is not for impersonation, deception or harassment."
)
WORKER = Path(__file__).resolve().parents[2] / "scripts" / "facefusion_worker.py"
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def default_output(video: Path) -> Path:
    """Suggest a new file beside the movie; the pipeline never overwrites one."""
    candidate = video.with_name(f"{video.stem}-swapped-720p.mp4")
    number = 2
    while candidate.exists():
        candidate = video.with_name(f"{video.stem}-swapped-720p-{number}.mp4")
        number += 1
    return candidate


def build_command(
    video: Path,
    face: Path,
    output: Path,
    character_time: str = "",
    character: int = 0,
    scan_only: bool = False,
    confirmed_job: str | None = None,
) -> list[str]:
    # pythonw has no console, so the job itself runs under the matching python.exe.
    python = Path(sys.executable)
    if python.name.lower() == "pythonw.exe":
        python = python.with_name("python.exe")
    command = [str(python), "-m", "face_replace"]
    command += ["--video", str(video), "--face", str(face), "--output", str(output)]
    command += ["--consent-confirmed"]
    if character_time:
        command += ["--character-time", character_time]
    else:
        if character:
            command += ["--character", str(character)]
        if confirmed_job:
            command += ["--confirmed-job", confirmed_job]
    if scan_only:
        command += ["--scan-only"]
    return command


def parse_candidates(line: str) -> list[dict] | None:
    prefix = "face-replace-candidates: "
    return json.loads(line.strip().removeprefix(prefix)) if line.startswith(prefix) else None


def parse_percent(text: str) -> float | None:
    # Upstream bars restart for each stage and segment, so only whole-job progress moves the bar.
    match = re.match(r"swapped segment .*?(\d+(?:\.\d+)?)%", text)
    return float(match.group(1)) if match else None


def parse_job(line: str) -> str | None:
    prefix = "face-replace-job: "
    return line.strip().removeprefix(prefix) if line.startswith(prefix) else None


def display_line(line: str) -> str | None:
    """Keep the job's status and error lines; hide the final JSON summary."""
    line = line.strip()
    if line.startswith("face-replace: "):
        return line.removeprefix("face-replace: ")
    if line.startswith("error: "):
        return line
    return None


def format_elapsed(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.process = None
        self.cancelled = False
        self.started = 0.0
        self.last_error = None
        self.result = None
        self.pending_output = None
        self.scratch_before = set()
        self.scanning = False
        self.candidates = []
        self.character = 0
        self.confirmed = False
        self.confirmed_job = None
        self.job_name = None
        self.lines = queue.Queue()
        self.movie = tk.StringVar()
        self.photo = tk.StringVar()
        self.output = tk.StringVar()
        self.character_time = tk.StringVar()
        self.consent = tk.BooleanVar()
        self.status = tk.StringVar(value="Choose a movie and a photo.")
        self.input_widgets = []

        root.title("Face Swap (720p)")
        root.geometry("680x600")
        root.minsize(620, 480)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        root.protocol("WM_DELETE_WINDOW", self.close)

        frame = ttk.Frame(root, padding=12)
        frame.grid(sticky="nsew")
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(7, weight=1)

        self.add_file_row(frame, 0, "Movie", self.movie, self.choose_movie)
        self.add_file_row(frame, 1, "Photo", self.photo, self.choose_photo)
        self.add_file_row(frame, 2, "Save as", self.output, self.choose_output)

        ttk.Label(frame, text="Character time").grid(row=3, column=0, sticky="w", pady=4)
        time_entry = ttk.Entry(frame, textvariable=self.character_time, width=12)
        time_entry.grid(row=3, column=1, sticky="w", padx=8)
        self.input_widgets.append(time_entry)
        self.character_time.trace_add("write", lambda *_: self.refresh_buttons())
        ttk.Label(
            frame,
            text="Optional, e.g. 00:12:34. Replaces the largest face at that moment. "
            "Leave empty to replace the face seen most often.",
            wraplength=560,
            foreground="#555555",
        ).grid(row=4, column=0, columnspan=3, sticky="w")

        ttk.Style().configure("TCheckbutton", wraplength=560)
        ttk.Checkbutton(
            frame, text=CONSENT_TEXT, variable=self.consent, command=self.refresh_buttons
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(10, 6))

        buttons = ttk.Frame(frame)
        buttons.grid(row=6, column=0, columnspan=3, sticky="w", pady=6)
        self.scan_button = ttk.Button(buttons, text="Scan", command=self.scan)
        self.start_button = ttk.Button(buttons, text="Start swap", command=self.start)
        self.cancel_button = ttk.Button(buttons, text="Cancel", command=self.cancel)
        self.open_button = ttk.Button(buttons, text="Open result", command=self.open_result)
        self.folder_button = ttk.Button(buttons, text="Open folder", command=self.open_folder)
        for column, button in enumerate(
            (
                self.scan_button,
                self.start_button,
                self.cancel_button,
                self.open_button,
                self.folder_button,
            )
        ):
            button.grid(row=0, column=column, padx=(0, 6))

        self.log = tk.Text(frame, height=10, wrap="word", state="disabled")
        self.log.grid(row=7, column=0, columnspan=3, sticky="nsew", pady=6)
        self.progress = ttk.Progressbar(frame, maximum=100)
        self.progress.grid(row=8, column=0, columnspan=3, sticky="ew")
        ttk.Label(frame, textvariable=self.status).grid(row=9, column=0, columnspan=3, sticky="w")

        self.refresh_buttons()
        self.root.after(200, self.poll)

    def add_file_row(self, frame, row, label, variable, command) -> None:
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
        entry = ttk.Entry(frame, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", padx=8)
        # A confirmation covers these exact inputs, however they were changed.
        variable.trace_add("write", lambda *_: self.inputs_changed())
        browse = ttk.Button(frame, text="Browse…", command=command)
        browse.grid(row=row, column=2)
        self.input_widgets += [entry, browse]

    def choose_movie(self) -> None:
        path = filedialog.askopenfilename(title="Choose the movie", filetypes=MOVIE_TYPES)
        if path:
            self.movie.set(path)
            self.output.set(str(default_output(Path(path))))

    def choose_photo(self) -> None:
        path = filedialog.askopenfilename(title="Choose the photo", filetypes=PHOTO_TYPES)
        if path:
            self.photo.set(path)

    def inputs_changed(self) -> None:
        # Consent was given for specific files, so new files need it again.
        self.consent.set(False)
        self.forget_character()

    def forget_character(self) -> None:
        self.candidates = []
        self.character = 0
        self.confirmed = False
        self.confirmed_job = None
        self.refresh_buttons()

    def choose_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save the result as",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4")],
            confirmoverwrite=False,
        )
        if path:
            self.output.set(path)

    def refresh_buttons(self) -> None:
        running = self.process is not None
        # A running job's results belong to the inputs it started with.
        for widget in self.input_widgets:
            widget.state(["disabled" if running else "!disabled"])
        ready = all(v.get().strip() for v in (self.movie, self.photo, self.output))
        can_run = ready and self.consent.get() and not running
        self.scan_button.state(["!disabled" if can_run else "disabled"])
        # A long swap starts only after a character was confirmed or named by time.
        chosen = self.confirmed or self.character_time.get().strip()
        self.start_button.state(["!disabled" if can_run and chosen else "disabled"])
        self.cancel_button.state(["!disabled" if running else "disabled"])
        has_result = self.result is not None and not running
        self.open_button.state(["!disabled" if has_result else "disabled"])
        self.folder_button.state(["!disabled" if has_result else "disabled"])

    def append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def scan(self) -> None:
        self.run_job(scan_only=True)

    def start(self) -> None:
        self.run_job(scan_only=False)

    def run_job(self, scan_only: bool) -> None:
        video = Path(self.movie.get().strip())
        face = Path(self.photo.get().strip())
        output = Path(self.output.get().strip())
        character_time = self.character_time.get().strip()
        if not self.consent.get():
            return
        if not video.is_file() or not face.is_file():
            messagebox.showerror("Face Swap", "Choose an existing movie and photo.")
            return
        if character_time:
            try:
                clock_seconds(character_time)
            except argparse.ArgumentTypeError:
                messagebox.showerror("Face Swap", "Character time must look like 00:12:34.")
                return
        try:
            # Known before the job reports it, so a very early cancel can still remove the folder.
            job_name = job_directory(video.resolve(), face.resolve(), output.resolve(), WORKER).name
        except OSError:
            messagebox.showerror("Face Swap", "The movie or photo could not be read.")
            return
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.cancelled = False
        self.scanning = scan_only
        if scan_only:
            self.forget_character()
        self.progress["value"] = 0
        self.last_error = None
        self.result = None
        self.job_name = job_name
        self.pending_output = output
        self.scratch_before = set(output.parent.glob(".face-replace-*"))
        self.started = time.monotonic()
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            self.process = subprocess.Popen(
                build_command(
                    video,
                    face,
                    output,
                    character_time,
                    self.character,
                    scan_only,
                    self.confirmed_job,
                ),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                errors="replace",
                env={**os.environ, "PYTHONUTF8": "1"},
                creationflags=flags,
            )
        except OSError:
            self.process = None
            messagebox.showerror("Face Swap", "The job could not be started.")
            return
        self.keep_awake(True)
        threading.Thread(target=self.read_output, args=(self.process,), daemon=True).start()
        self.refresh_buttons()

    def read_output(self, process: subprocess.Popen) -> None:
        for line in process.stdout:
            self.job_name = parse_job(line) or self.job_name
            self.lines.put(line)
        process.stdout.close()
        self.lines.put(None)

    def poll(self) -> None:
        finished = False
        while True:
            try:
                line = self.lines.get_nowait()
            except queue.Empty:
                break
            if line is None:
                finished = True
                break
            candidates = parse_candidates(line)
            if candidates is not None:
                self.candidates = candidates
            shown = display_line(line)
            if shown:
                self.append_log(shown)
                percent = parse_percent(shown)
                if percent is not None:
                    self.progress["value"] = percent
                if shown.startswith("error: "):
                    self.last_error = shown.removeprefix("error: ")
        if self.process is not None:
            self.status.set(f"Running… {format_elapsed(time.monotonic() - self.started)}")
        if finished:
            self.finish()
        self.root.after(200, self.poll)

    def finish(self) -> None:
        code = self.process.wait()
        elapsed = format_elapsed(time.monotonic() - self.started)
        self.process = None
        self.keep_awake(False)
        if self.cancelled:
            self.consent.set(False)
            if messagebox.askyesno("Face Swap", "Keep progress so this job can resume later?"):
                self.status.set("Cancelled. Press Scan or Start swap to resume.")
            else:
                removed = self.remove_scratch()
                self.forget_character()
                self.status.set(
                    "Cancelled and progress removed."
                    if removed
                    else "Cancelled, but a .face-replace-job folder could not be deleted. "
                    "Delete it by hand."
                )
        elif code == 0 and self.scanning:
            self.status.set(f"Scan finished in {elapsed}.")
            self.confirm_character()
        elif code == 0:
            self.consent.set(False)
            self.forget_character()
            self.progress["value"] = 100
            self.result = self.pending_output
            self.status.set(f"Done in {elapsed}.")
            messagebox.showinfo("Face Swap", f"Finished in {elapsed}.\n\n{self.result}")
        else:
            self.consent.set(False)
            message = self.last_error or "The job failed."
            if (self.pending_output.parent / str(self.job_name)).exists():
                message += f"\n\nProgress, including private media, stays in {self.job_name}."
            self.status.set("Stopped with an error.")
            messagebox.showerror("Face Swap", message)
        self.refresh_buttons()

    def confirm_character(self) -> None:
        """Show each candidate so the owner confirms who is replaced before the long swap."""
        if not self.candidates:
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("Confirm the character")
        dialog.transient(self.root)
        index = tk.IntVar(value=0)
        picture = ttk.Label(dialog)
        picture.grid(row=0, column=0, columnspan=3, padx=8, pady=8)
        caption = ttk.Label(dialog)
        caption.grid(row=1, column=0, columnspan=3)

        def show() -> None:
            candidate = self.candidates[index.get()]
            image = tk.PhotoImage(
                master=dialog, file=self.pending_output.parent / candidate["preview"]
            )
            scale = max(1, image.width() // 640 + 1)
            picture.image = image.subsample(scale, scale)
            picture.configure(image=picture.image)
            caption.configure(
                text=f"Character {index.get() + 1} of {len(self.candidates)}: seen in "
                f"{candidate['appearances']} samples, shown at {candidate['time']}. "
                "The green box marks the face to replace."
            )

        def next_candidate() -> None:
            index.set((index.get() + 1) % len(self.candidates))
            show()

        def use_this() -> None:
            self.character = index.get()
            self.confirmed = True
            self.confirmed_job = self.job_name
            self.character_time.set("")
            self.status.set(f"Character {index.get() + 1} confirmed. Press Start swap.")
            dialog.destroy()
            self.refresh_buttons()

        ttk.Button(dialog, text="Next character", command=next_candidate).grid(row=2, column=0)
        ttk.Button(dialog, text="Replace this one", command=use_this).grid(row=2, column=1)
        show()
        # Nothing else in the window may change while a character is being confirmed.
        dialog.wait_visibility()
        dialog.grab_set()

    def cancel(self) -> None:
        if self.process is None:
            return
        self.cancelled = True
        self.status.set("Cancelling…")
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(self.process.pid)],
                capture_output=True,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            # The job ends its worker on this signal; a hard kill would leave the worker running.
            self.process.terminate()

    def remove_scratch(self) -> bool:
        # A killed job cannot clean up after itself; only this run's folders are removed.
        parent = self.pending_output.parent
        scratch = set(parent.glob(".face-replace-*")) - self.scratch_before
        if self.job_name:
            scratch.add(parent / self.job_name)
        removed = True
        for folder in scratch:
            try:
                shutil.rmtree(folder)
            except FileNotFoundError:
                pass
            except OSError:
                removed = False
        return removed

    def keep_awake(self, active: bool) -> None:
        # A long job dies if the laptop sleeps part way through.
        if os.name == "nt":
            state = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if active else 0)
            ctypes.windll.kernel32.SetThreadExecutionState(state)

    def open_result(self) -> None:
        if self.result is not None:
            os.startfile(self.result)

    def open_folder(self) -> None:
        if self.result is not None:
            os.startfile(self.result.parent)

    def close(self) -> None:
        if self.process is not None:
            if not messagebox.askyesno("Face Swap", "A job is running. Cancel it and close?"):
                return
            self.cancel()
            self.process.wait()
            if not messagebox.askyesno("Face Swap", "Keep progress so this job can resume later?"):
                if not self.remove_scratch():
                    messagebox.showwarning(
                        "Face Swap",
                        "A .face-replace-job folder could not be deleted. Delete it by hand.",
                    )
            self.keep_awake(False)
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
