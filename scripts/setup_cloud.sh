#!/usr/bin/env bash
# Repeatable setup on an Ubuntu NVIDIA GPU host with CUDA 13. Run as root from the repo root.
# Downloads about 5 GB of packages and 725 MB of model weights.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
revision=72470819a0373be3388b3929c8f8f311f418fc3c

command -v nvidia-smi >/dev/null || { echo 'nvidia-smi is required' >&2; exit 1; }
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq fontconfig fonts-dejavu-core xz-utils tmux curl git python3-pip >/dev/null

# Ubuntu 22.04's FFmpeg is too old for -fps_mode, so use a static build.
if ! grep -q fps_mode <<<"$(ffmpeg -hide_banner -h full 2>/dev/null)"; then
  base=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest
  curl -fsSL "$base/ffmpeg-master-latest-linux64-gpl.tar.xz" -o /tmp/ffmpeg.tar.xz
  tar -xJf /tmp/ffmpeg.tar.xz -C /tmp
  install -m 755 /tmp/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg /usr/local/bin/ffmpeg
  install -m 755 /tmp/ffmpeg-master-latest-linux64-gpl/bin/ffprobe /usr/local/bin/ffprobe
fi

if [[ ! -x .venv-gpu/bin/python ]]; then
  python3 -m pip install -q uv
  python3 -m uv venv --python 3.12 --seed .venv-gpu
fi

if [[ ! -e upstream-facefusion ]]; then
  git clone -q --no-checkout https://github.com/facefusion/facefusion upstream-facefusion
  git -C upstream-facefusion checkout -q --detach "$revision"
fi
[[ "$(git -C upstream-facefusion rev-parse HEAD)" == "$revision" ]]
[[ -z "$(git -C upstream-facefusion status --porcelain)" ]]

export PATH="$PWD/.venv-gpu/bin:$PATH"
(cd upstream-facefusion && python install.py cuda@13 --skip-conda)
# The upstream installer ignores pip's exit status and does not fetch cuDNN.
python -m pip install -q 'onnxruntime-gpu[cuda,cudnn]==1.30.0'
python -m pip install -q -e '.[dev]'
python -m pip check
python scripts/fetch_models.py >/dev/null
python -m face_replace --help >/dev/null
echo 'Setup complete.'
