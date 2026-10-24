#!/usr/bin/env bash
# Prerequisite: an operator-approved Linux CUDA 12.8+ / cuDNN 9 image, Python 3.12.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mode=${1:---verify-only}
case "$mode" in --verify-only|--install) ;; *) echo 'Use --verify-only or --install' >&2; exit 2;; esac
[[ "$(uname -s)" == Linux ]] || { echo 'Linux is required' >&2; exit 1; }
for tool in nvidia-smi ffmpeg ffprobe python3 git; do
  command -v "$tool" >/dev/null || { echo "Missing prerequisite: $tool" >&2; exit 1; }
done
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
python3 -c 'import sys,venv,ssl; assert sys.version_info[:2] == (3,12), "Python 3.12 required"'
[[ "$(git -C upstream-facefusion rev-parse HEAD)" == 72470819a0373be3388b3929c8f8f311f418fc3c ]]
[[ -z "$(git -C upstream-facefusion status --porcelain)" ]]
python3 scripts/verify_gpu.py --preflight
if [[ "$mode" == --install ]]; then
  python3 -m venv .venv-gpu
  .venv-gpu/bin/python -m pip --version
  # Use the pinned official installer; it does not download model weights.
  (
    export PATH="$PWD/.venv-gpu/bin:$PATH"
    cd upstream-facefusion
    python install.py cuda@12 --skip-conda
  )
  .venv-gpu/bin/python -m pip install -e .
fi
[[ -x .venv-gpu/bin/python ]] || {
  echo 'GPU environment is absent; --verify-only never installs it.' >&2; exit 1;
}
.venv-gpu/bin/python - <<'PYTHON'
from importlib.metadata import version
from pathlib import Path
for requirement in Path('upstream-facefusion/requirements.txt').read_text().splitlines():
    if requirement.startswith('onnxruntime'):
        requirement = 'onnxruntime-gpu==1.24.4'
    name, expected = requirement.split('==')
    assert version(name) == expected, f'Installed package mismatch: {name}'
PYTHON
.venv-gpu/bin/python -m pip check
.venv-gpu/bin/python scripts/verify_gpu.py
