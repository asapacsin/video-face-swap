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

BASE = "https://github.com/facefusion/facefusion-assets/releases/download"
MODELS = {
    "nsfw_1": "models-3.3.0",
    "nsfw_2": "models-3.3.0",
    "nsfw_3": "models-3.3.0",
    "fairface": "models-3.0.0",
    "yoloface_8n": "models-3.0.0",
    "2dfan4": "models-3.0.0",
    "fan_68_5": "models-3.0.0",
    "xseg_1": "models-3.1.0",
    "bisenet_resnet_34": "models-3.0.0",
    "arcface_w600k_r50": "models-3.0.0",
    "kim_vocal_2": "models-3.0.0",
    "hyperswap_1a_256": "models-3.3.0",
}


def fetch(url: str, path: Path) -> None:
    partial = path.with_name(path.name + ".part")
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as file:
                shutil.copyfileobj(response, file, 1024 * 1024)
            partial.replace(path)
            return
        except OSError as exc:
            if attempt == 5:
                raise
            print(f"retrying {path.name} after {type(exc).__name__}", file=sys.stderr)
            time.sleep(5 * attempt)


def verified(model: Path, expected_crc: Path) -> bool:
    if not model.is_file() or not expected_crc.is_file():
        return False
    return format(zlib.crc32(model.read_bytes()), "08x") == expected_crc.read_text().strip()


def main() -> int:
    target = Path(__file__).resolve().parents[1] / "upstream-facefusion" / ".assets" / "models"
    target.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, release in MODELS.items():
        model, expected_crc = target / f"{name}.onnx", target / f"{name}.hash"
        if not verified(model, expected_crc):
            print(f"downloading {name}", file=sys.stderr)
            fetch(f"{BASE}/{release}/{name}.hash", expected_crc)
            fetch(f"{BASE}/{release}/{name}.onnx", model)
        if not verified(model, expected_crc):
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
