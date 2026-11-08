# Face replacement — consent-gated 720p movie pipeline

Status: **Synthetic GHOST CLI and GUI conversions passed on the RTX 4050. No real-person media
was used; identity quality, long-video behavior and performance remain NOT_VERIFIED.**
The current worker uses GHOST with locally loaded ONNX models and pinned FaceFusion face-analysis
helpers. See the
[product spec](docs/SPEC.md) and [benchmarks](docs/BENCHMARKS.md).

## Run on an Ubuntu NVIDIA GPU host

From the repo root on an Ubuntu NVIDIA GPU host with CUDA 13 (downloads about 5 GB of packages
and 725 MB of required model weights):

```bash
bash scripts/setup_cloud.sh
```

Scan first to see who can be replaced. This prints the candidates and saves a preview of each:

```bash
.venv-gpu/bin/python -m face_replace --video assets/movie.mp4 --face assets/photo.jpg --output outputs/result.mp4 --consent-confirmed --content-rating non-explicit --scan-only
```

Then run the job with the chosen candidate number, starting at 0:

```bash
.venv-gpu/bin/python -m face_replace --video assets/movie.mp4 --face assets/photo.jpg --output outputs/result.mp4 --consent-confirmed --content-rating non-explicit --character 0
```

If a job stops, run the same command again and it resumes from the last finished 5-minute segment.
Progress is kept in a `.face-replace-job-*` folder beside the output until the job succeeds.

## Use the app on Windows

Synthetic CLI and GUI conversions have passed on the Windows RTX 4050; real-person identity
quality remains unverified. First time only, from PowerShell in this folder (downloads about
3.5 GB of packages and 725 MB of model weights):

```bash
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Models
```

1. Double-click `FaceSwap.bat`, choose the movie and photo, and check both the consent and
   non-explicit confirmations.
2. Press **Scan**, then choose the character to replace in the window that opens.
3. Press **Start swap**. You can instead enter a time when the character's face is the largest on
   screen and press **Start swap** directly.

Any readable movie is converted to a 720p MP4 (never upscaled). The result is saved beside it as
`<movie>-swapped-720p.mp4` with an `AI-ALTERED` label and the first audio track, and
`<movie>-swapped-720p-character.png` boxes the face that was replaced. The local app does not
automatically classify video or reference-photo content.

Backend: GHOST ONNX models with FaceFusion's pinned face detection, landmark and recognition
helpers. The upstream checkout stays unmodified.
See [architecture](docs/ARCHITECTURE.md), [consent policy](docs/PRIVACY_AND_CONSENT.md),
[environment](docs/ENVIRONMENT.md), [reviewer](docs/REVIEWER.md),
[remaining work](docs/OPEN_PROBLEMS.md) and [model manifest](docs/MODEL_MANIFEST.md).

## Consent and exit codes

Attestation means all people whose identities are used are consenting adults, media is non-explicit
and authorized, and the use is not impersonation, harassment or deception. Real CLI jobs require
`--content-rating non-explicit`; `--content-rating explicit` refuses the request. This is an operator
assertion, not an automated content check. `--dry-run` checks the inputs and reports dimensions,
duration, frame rate and audio presence without screening, using a GPU or writing media.

Exit codes: 0 successful dry-run, help or finished job; 2 invalid input or consent; 3 inference failed
or refused. A failed job publishes no output.

## Tests

CPU tests use fake workers and do not prove GPU inference. Install `.[dev]`, then:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```
```bash
python -m ruff check src tests scripts
```

Weights, assets, media, virtual environments and the upstream checkout are ignored by Git.
