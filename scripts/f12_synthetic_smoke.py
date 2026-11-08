#!/usr/bin/env python3
"""Run the approved synthetic-only F12 CLI and Tk GUI GPU smoke checks."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIXTURE_HASHES = {
    "movie.mp4": "33d0e847dd49683978dac2604d36f493c5ae823574f5203220bc637c44edaef4",
    "reference.png": "6deb6a8ca665a1dd761036342eb0cbfe12c8147bff35c9a39396e073310531c0",
    "target.png": "4fde9ce3ec119340e66d512bd25fdebb49c70febf657249ea1937fa1c508bbd0",
}
EXPECTED_MODEL_AUDIT = (
    "audited CUDA inference sessions: yoloface_8n.onnx, fan_68_5.onnx, "
    "arcface_w600k_r50.onnx, ghost_1_256.onnx, crossface_ghost.onnx"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_fixtures(folder: Path) -> dict[str, str]:
    actual = {}
    for filename, expected in FIXTURE_HASHES.items():
        path = folder / filename
        if not path.is_file():
            raise RuntimeError(f"required synthetic fixture is missing: {filename}")
        actual[filename] = sha256(path)
        if actual[filename] != expected:
            raise RuntimeError(
                f"fixture hash does not match the approved synthetic file: {filename}"
            )
    return actual


def audit_lines(text: str) -> list[str]:
    matches = []
    for line in text.splitlines():
        if "audited CUDA inference sessions:" in line:
            safe_line = line.split("face-replace: ", 1)[-1]
            if safe_line != EXPECTED_MODEL_AUDIT:
                raise RuntimeError("worker CUDA model audit did not match the pinned model set")
            matches.append(safe_line)
        elif "content-screening modules: not imported" in line:
            matches.append("content-screening modules: not imported")
        elif line.startswith("face-replace-error:") or line.startswith("error:"):
            raise RuntimeError("synthetic smoke job reported a worker error")
    if EXPECTED_MODEL_AUDIT not in matches:
        raise RuntimeError("worker did not report its completed CUDA model audit")
    if "content-screening modules: not imported" not in matches:
        raise RuntimeError("worker did not report the screening-module audit")
    return sorted(set(matches))


def probe(path: Path, source: Path) -> dict:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format_tags=comment:stream=codec_type,width,height,avg_frame_rate,start_time,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(completed.stdout)
    streams = data["streams"]
    video = next(stream for stream in streams if stream["codec_type"] == "video")
    audio = [stream for stream in streams if stream["codec_type"] == "audio"]
    result = {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "frame_rate": video["avg_frame_rate"],
        "video_frames": int(video["nb_frames"]),
        "video_start": float(video["start_time"]),
        "audio_streams": len(audio),
        "audio_start": float(audio[0]["start_time"]) if audio else None,
        "metadata_comment": data.get("format", {}).get("tags", {}).get("comment"),
        "visible_label": "AI-ALTERED overlay source-comparison pixel check passed",
        "visible_label_pixel_delta": verify_visible_label(path, source),
        "face_region_delta": verify_face_region(path, source),
    }
    expected = {
        "width": 1280,
        "height": 720,
        "frame_rate": "24000/1001",
        "video_frames": 119,
        "video_start": 0.0,
        "audio_streams": 1,
        "audio_start": 0.0,
        "metadata_comment": "AI-ALTERED face replacement",
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise RuntimeError("synthetic output did not meet the recorded F12 media checks")
    return result


def verify_visible_label(path: Path, source: Path) -> dict[str, int]:
    import cv2
    import numpy

    output_reader = cv2.VideoCapture(str(path))
    source_reader = cv2.VideoCapture(str(source))
    try:
        output_reader.set(cv2.CAP_PROP_POS_MSEC, 1000)
        source_reader.set(cv2.CAP_PROP_POS_MSEC, 1000)
        output_ok, output_frame = output_reader.read()
        source_ok, source_frame = source_reader.read()
    finally:
        output_reader.release()
        source_reader.release()
    if not output_ok or output_frame is None or not source_ok or source_frame is None:
        raise RuntimeError("could not decode source/output frames for the visible-label check")
    output_region = output_frame[8:36, 8:170]
    source_region = source_frame[8:36, 8:170]
    new_light_pixels = int(
        (numpy.all(output_region > 220, axis=2) & ~numpy.all(source_region > 220, axis=2)).sum()
    )
    new_dark_pixels = int(
        (numpy.all(output_region < 70, axis=2) & ~numpy.all(source_region < 70, axis=2)).sum()
    )
    if new_light_pixels < 80 or new_dark_pixels < 250:
        raise RuntimeError("the visible label text and backing differ insufficiently from source")
    return {"new_light_pixels": new_light_pixels, "new_dark_pixels": new_dark_pixels}


def verify_face_region(path: Path, source: Path) -> dict[str, float]:
    import cv2
    import numpy

    frames = []
    for filename in (source, path):
        reader = cv2.VideoCapture(str(filename))
        try:
            reader.set(cv2.CAP_PROP_POS_MSEC, 1000)
            success, frame = reader.read()
        finally:
            reader.release()
        if not success or frame is None:
            raise RuntimeError("could not decode source/output frames for the face-region check")
        frames.append(frame)

    source_frame, output_frame = frames
    source_face = source_frame[170:530, 400:880].astype(numpy.int16)
    output_face = output_frame[170:530, 400:880].astype(numpy.int16)
    source_control = source_frame[:150, 250:1000].astype(numpy.int16)
    output_control = output_frame[:150, 250:1000].astype(numpy.int16)
    face_delta = numpy.abs(output_face - source_face).mean(axis=2)
    control_delta = numpy.abs(output_control - source_control).mean(axis=2)
    changed_face_fraction = float((face_delta > 30).mean())
    changed_control_fraction = float((control_delta > 30).mean())
    if changed_face_fraction < 0.03 or changed_face_fraction < changed_control_fraction * 2:
        raise RuntimeError("the synthetic face region did not change beyond encoding noise")
    return {
        "face_pixels_changed_over_30": changed_face_fraction,
        "control_pixels_changed_over_30": changed_control_fraction,
    }


def run_cli(folder: Path, output: Path) -> dict:
    command = [
        sys.executable,
        "-m",
        "face_replace",
        "--video",
        str(folder / "movie.mp4"),
        "--face",
        str(folder / "reference.png"),
        "--output",
        str(output),
        "--consent-confirmed",
        "--content-rating",
        "non-explicit",
        "--character",
        "0",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUTF8": "1"},
        check=False,
    )
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError("synthetic CLI conversion failed; inspect local stderr")
    return {
        "command": [
            "python -m face_replace",
            "--video movie.mp4",
            "--face reference.png",
            "--output " + output.name,
            "--consent-confirmed",
            "--content-rating non-explicit",
            "--character 0",
        ],
        "worker_audit": audit_lines(completed.stderr),
        "media": probe(output, folder / "movie.mp4"),
        "output_sha256": sha256(output),
    }


def run_gui(folder: Path, output: Path) -> dict:
    import tkinter as tk

    from face_replace import gui

    root = tk.Tk()
    app = gui.App(root)
    showinfo = gui.messagebox.showinfo
    showerror = gui.messagebox.showerror
    errors = []
    gui.messagebox.showinfo = lambda *_args, **_kwargs: None
    gui.messagebox.showerror = lambda _title, message: errors.append(message)
    try:
        app.movie.set(str(folder / "movie.mp4"))
        app.photo.set(str(folder / "reference.png"))
        app.output.set(str(output))
        app.consent.set(True)
        app.non_explicit_confirmed.set(True)
        app.scan()
        deadline = time.monotonic() + 180
        while app.process is not None and time.monotonic() < deadline:
            root.update()
            time.sleep(0.02)
        if app.process is not None or not app.candidates:
            raise RuntimeError("synthetic GUI scan did not produce a character candidate")
        dialog = next(w for w in root.winfo_children() if isinstance(w, tk.Toplevel))
        button = next(
            widget
            for widget in dialog.winfo_children()
            if widget.winfo_class() == "TButton" and widget.cget("text") == "Replace this one"
        )
        button.invoke()
        app.consent.set(True)
        app.non_explicit_confirmed.set(True)
        app.start()
        deadline = time.monotonic() + 240
        while app.process is not None and time.monotonic() < deadline:
            root.update()
            time.sleep(0.02)
        if app.process is not None or not output.is_file():
            raise RuntimeError("synthetic GUI conversion did not produce an output")
        if errors:
            raise RuntimeError("synthetic GUI reported a worker error")
        return {
            "actions": ["scan", "select first detected character", "swap"],
            "worker_audit": audit_lines(app.log.get("1.0", "end")),
            "media": probe(output, folder / "movie.mp4"),
            "output_sha256": sha256(output),
        }
    finally:
        gui.messagebox.showinfo = showinfo
        gui.messagebox.showerror = showerror
        app.stop_polling()
        root.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "f12-synth-cuE1kt",
    )
    parser.add_argument("--evidence-out", type=Path)
    args = parser.parse_args()
    folder = args.fixture_dir.resolve()
    fixture_hashes = validate_fixtures(folder)
    with tempfile.TemporaryDirectory(prefix=".f12-synthetic-", dir=folder) as run_folder:
        run_folder = Path(run_folder)
        outputs = {
            "cli": run_folder / "cli.mp4",
            "gui": run_folder / "gui.mp4",
        }
        evidence = {
            "evidence_scope": "synthetic-only; no real-person media",
            "output_hash_note": "H.264 output hashes are run-specific and may differ across runs.",
            "fixture_sha256": fixture_hashes,
            "cli": run_cli(folder, outputs["cli"]),
            "gui": run_gui(folder, outputs["gui"]),
        }
    rendered = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    if args.evidence_out:
        args.evidence_out.write_text(rendered, encoding="utf-8")
        print("synthetic F12 evidence written")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
