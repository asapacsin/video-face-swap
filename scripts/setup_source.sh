#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
revision=72470819a0373be3388b3929c8f8f311f418fc3c
for tool in git python3 ffmpeg ffprobe; do command -v "$tool" >/dev/null; done
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'
if [[ ! -e upstream-facefusion ]]; then
  git clone --no-checkout https://github.com/facefusion/facefusion upstream-facefusion
  git -C upstream-facefusion checkout --detach "$revision"
fi
[[ "$(git -C upstream-facefusion rev-parse HEAD)" == "$revision" ]] || {
  echo 'Upstream revision mismatch; preserve checkout and inspect manually.' >&2; exit 1;
}
[[ -z "$(git -C upstream-facefusion status --porcelain)" ]] || {
  echo 'Upstream is modified; preserve files and inspect manually.' >&2; exit 1;
}
python3 -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check -e '.[dev]'
.venv/bin/python -m face_replace --help
