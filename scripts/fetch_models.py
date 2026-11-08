#!/usr/bin/env python3
"""Fetch the approved FaceFusion models listed in docs/MODEL_MANIFEST.md, then verify them."""

import hashlib
import json
import shutil
import sys
import time
import urllib.request
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from face_replace.models import MODEL_RELEASES as MODELS
from face_replace.models import MODEL_SHA256 as EXPECTED_SHA256

BASE = "https://github.com/facefusion/facefusion-assets/releases/download"


def fetch(url: str, path: Path) -> None:
    # Resumes from the partial file, so a slow link still finishes; gives up only without progress.
    partial = path.with_name(path.name + ".part")
    stalled = 0
    while True:
        before = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={before}-"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                # A server that ignores the range sends the whole file again.
                with partial.open("ab" if response.status == 206 else "wb") as file:
                    start = file.tell()
                    shutil.copyfileobj(response, file, 1024 * 1024)
                    expected = response.headers.get("Content-Length")
                    # A connection closed cleanly mid-file raises nothing, so check its length.
                    if expected is not None and file.tell() - start != int(expected):
                        raise TimeoutError("download ended early")
            partial.replace(path)
            return
        except OSError as exc:
            if getattr(exc, "code", None) == 416:
                # The range starts at or past the end, so this partial file cannot be continued.
                partial.unlink()
            after = partial.stat().st_size if partial.exists() else 0
            stalled = 0 if after > before else stalled + 1
            if stalled == 5:
                raise
            print(f"retrying {path.name} after {type(exc).__name__}", file=sys.stderr)
            time.sleep(5 * stalled)


def verified(model: Path, expected_crc: Path, expected_sha256: str) -> bool:
    if not model.is_file() or not expected_crc.is_file():
        return False
    data = model.read_bytes()
    return (
        format(zlib.crc32(data), "08x") == expected_crc.read_text().strip()
        and hashlib.sha256(data).hexdigest() == expected_sha256
    )


def main() -> int:
    target = Path(__file__).resolve().parents[1] / "upstream-facefusion" / ".assets" / "models"
    target.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, release in MODELS.items():
        model, expected_crc = target / f"{name}.onnx", target / f"{name}.hash"
        if not verified(model, expected_crc, EXPECTED_SHA256[name]):
            print(f"downloading {name}", file=sys.stderr)
            fetch(f"{BASE}/{release}/{name}.hash", expected_crc)
            fetch(f"{BASE}/{release}/{name}.onnx", model)
        if not verified(model, expected_crc, EXPECTED_SHA256[name]):
            print(f"checksum mismatch for {name}", file=sys.stderr)
            return 1
        data = model.read_bytes()
        report[name] = {
            "release": release,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
