# Face replacement — CPU pipeline preparation

Status: **CPU pipeline implemented; full project NOT_VERIFIED**. Help and dry-run
work without models. Inference uses a fixed FaceFusion worker, requires the pinned
upstream checkout, approved model files and a compatible GPU. CPU fake-worker tests
exercise finalization, labeling, audio retention and output validation, not face quality.

Backend selected provisionally: FaceFusion at
`72470819a0373be3388b3929c8f8f311f418fc3c`, with its safety code untouched. See
[architecture](docs/ARCHITECTURE.md), [consent policy](docs/PRIVACY_AND_CONSENT.md),
[environment](docs/ENVIRONMENT.md), [review](docs/REVIEW_REPORT.md), and
[remaining work](docs/OPEN_PROBLEMS.md), [model manifest](docs/MODEL_MANIFEST.md),
and [project plan](docs/PROJECT_PLAN.md).

## Reproduce local setup

Prerequisites: Git, Python 3.12 with venv/pip, FFmpeg/ffprobe. Transfer this project
directory or restore its cloud snapshot; no project remote is configured. The
ignored upstream checkout is reproducible from its pin.

```bash
cd /workspace/face-replace
bash scripts/setup_source.sh
.venv/bin/python -m face_replace --help
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/ruff check src tests scripts/verify_gpu.py
.venv/bin/ruff format --check src tests scripts/verify_gpu.py
bash -n scripts/setup_source.sh scripts/setup_remote_gpu.sh
```

Setup can be repeated. It refuses to reset a changed upstream checkout. Weights,
assets, media, virtual environments and nested upstream Git history are ignored.
Review `git status` before staging. Nothing is pushed.

## Dry-run with local files

```bash
mkdir -p outputs
.venv/bin/python -m face_replace \
  --video /absolute/path/input.mp4 --face /absolute/path/authorized-face.jpg \
  --output "$PWD/outputs/result.mp4" --consent-confirmed --dry-run
```

Attestation means all people whose identities are used are consenting adults,
media is non-explicit and authorized, and the use is not impersonation, harassment
or deception. `--content-rating explicit` refuses the request. Dry-run does not
screen content, recognize a face, contact a GPU or write media. It reports dimensions,
duration, frame rate and audio presence without private paths. Exit codes: 0 successful
dry-run/help, 2 invalid input/consent, 3 inference unavailable. Omitting `--dry-run`
currently returns 3 after validation; no output is written.

## Synthetic CPU smoke fixture (not face inference)

```bash
mkdir -p assets outputs
ffmpeg -v error -f lavfi -i color=c=blue:s=64x64:r=24 \
  -f lavfi -i sine=frequency=440:sample_rate=48000 -t 1 \
  -c:v libx264 -pix_fmt yuv420p -c:a aac assets/synthetic.mp4
ffmpeg -v error -f lavfi -i color=c=green:s=64x64 \
  -frames:v 1 -threads 1 assets/synthetic.png
.venv/bin/python -m face_replace --video assets/synthetic.mp4 \
  --face assets/synthetic.png --output outputs/synthetic-result.mp4 \
  --consent-confirmed --dry-run
```

Generated colors and tone contain no identities and filenames identify them as
synthetic. They validate inspection only, not face suitability. Tests create
equivalent fixtures in temporary directories and remove them.

## Remote GPU commands — only after rental approval

Do not rent yet. User supplies host/access securely and approves GPU, tariff,
billing granularity, storage cost, runtime cap and shutdown procedure within about
RMB 200. Prefer Linux x86_64 with Python 3.12, CUDA 12.8+, cuDNN 9, FFmpeg and
NVIDIA drivers. Scripts never change drivers. These remote commands are prepared,
not executed. Replace `GPU_HOST` with the approved SSH alias and use SSH agent/key
configuration. Code sync excludes local inputs, outputs and secrets.

```bash
cd /workspace/face-replace
ssh GPU_HOST 'umask 077; mkdir -p ~/face-replace'
rsync -az --filter=':- .gitignore' --exclude='.git/' ./ GPU_HOST:face-replace/
ssh GPU_HOST 'cd ~/face-replace && bash scripts/setup_source.sh'
# Before installation, verify-only is expected to report missing GPU venv:
ssh GPU_HOST 'cd ~/face-replace && bash scripts/setup_remote_gpu.sh --verify-only'
ssh GPU_HOST 'cd ~/face-replace && bash scripts/setup_remote_gpu.sh --install'
ssh GPU_HOST 'cd ~/face-replace && bash scripts/setup_remote_gpu.sh --verify-only'
```

`--install` invokes pinned upstream `install.py cuda@12 --skip-conda` inside
`.venv-gpu`, then pip checks and an actual ONNX CUDA MatMul test. It installs
packages, never model weights. An existing CUDA/cuDNN image is required. Checks
identify B200/H800, inspect loaded runtime/cuDNN and require actual CUDA kernel
profiling evidence. The nvidia-smi CUDA ceiling alone is insufficient.
Full remote installation and repeatability remain NOT_VERIFIED until run there.

After approving private media storage on the host, transfer the authorized assets
separately (they are excluded from code sync):

```bash
ssh GPU_HOST 'umask 077; mkdir -p ~/face-replace/assets ~/face-replace/outputs'
scp /path/to/licensed-adult-clip.mp4 GPU_HOST:face-replace/assets/input.mp4
scp /path/to/authorized-adult-face.jpg GPU_HOST:face-replace/assets/reference.jpg
ssh GPU_HOST 'cd ~/face-replace && .venv-gpu/bin/python -m face_replace \
  --video assets/input.mp4 --face assets/reference.jpg --output outputs/result.mp4 \
  --consent-confirmed --dry-run'
```

That is the first runnable remote CLI command today. The eventual E2E command is
the same with `--dry-run` removed, **after M2 and exact model/download approvals**;
today it intentionally exits 3. Required E2E assets: short licensed/synthetic adult
non-explicit face video with audio, authorized adult reference, approved face/safety
model weights, verified GPU environment. Output must have visible AI labeling and
retained audio before project PASS. See [benchmark plan](docs/BENCHMARKS.md).
