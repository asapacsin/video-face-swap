# Face replacement — consent-gated 720p movie pipeline

Status: **Short clips and a 15-second film clip verified on a rented GPU server; full-length movies
and the current code on the RTX 4050 are NOT_VERIFIED**. Inference uses a fixed FaceFusion worker,
and requires the pinned upstream checkout, approved model files and a compatible GPU. See the
[product spec](docs/SPEC.md) and [benchmarks](docs/BENCHMARKS.md).

## Run on the rented server

Phase 1 runs all GPU work here. From the repo root on an Ubuntu NVIDIA server with CUDA 13 (downloads
about 5 GB of packages and 1.47 GB of models):

```bash
bash scripts/setup_cloud.sh
```

Scan first to see who can be replaced. This prints the candidates and saves a preview of each:

```bash
.venv-gpu/bin/python -m face_replace --video assets/movie.mp4 --face assets/photo.jpg --output outputs/result.mp4 --consent-confirmed --scan-only
```

Then run the job with the chosen candidate number, starting at 0:

```bash
.venv-gpu/bin/python -m face_replace --video assets/movie.mp4 --face assets/photo.jpg --output outputs/result.mp4 --consent-confirmed --character 0
```

If a job stops, run the same command again and it resumes from the last finished 5-minute segment.
Progress is kept in a `.face-replace-job-*` folder beside the output until the job succeeds.

## Use the app on Windows

This is phase 2 and has not yet been proven with the current code. First time only, from PowerShell
in this folder (downloads about 5 GB):

```bash
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Models
```

1. Double-click `FaceSwap.bat`, choose the movie and photo, and tick the confirmation.
2. Press **Scan**, then choose the character to replace in the window that opens.
3. Press **Start swap**. You can instead enter a time when the character's face is the largest on
   screen and press **Start swap** directly. This skips only the confirmation window: every frame is
   still screened first, so it takes just as long.

Any readable movie is converted to a 720p MP4 (never upscaled). The result is saved beside it as
`<movie>-swapped-720p.mp4` with an `AI-ALTERED` label and the first audio track, and
`<movie>-swapped-720p-character.png` boxes the face that was replaced. Every frame is screened, and
one refused frame stops the whole job.

Backend: FaceFusion at `72470819a0373be3388b3929c8f8f311f418fc3c`, with its safety code untouched.
See [architecture](docs/ARCHITECTURE.md), [consent policy](docs/PRIVACY_AND_CONSENT.md),
[environment](docs/ENVIRONMENT.md), [reviewer](docs/REVIEWER.md),
[remaining work](docs/OPEN_PROBLEMS.md) and [model manifest](docs/MODEL_MANIFEST.md).

## Consent and exit codes

Attestation means all people whose identities are used are consenting adults, media is non-explicit
and authorized, and the use is not impersonation, harassment or deception. `--content-rating
explicit` refuses the request. `--dry-run` checks the inputs and reports dimensions, duration, frame
rate and audio presence without screening, using a GPU or writing media.

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
