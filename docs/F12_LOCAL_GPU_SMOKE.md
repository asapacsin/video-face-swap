# F12 local GPU smoke check

Date: 2026-11-08. Environment: NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB), driver
610.78, local Windows `.venv-gpu` environment.

The GHOST worker verified the pinned FaceFusion revision and clean tracked checkout, checked all
five model SHA-256 values, and loaded each ONNX session with CPU fallback disabled. The runtime
ONNX Runtime audit allows session creation only for the five pinned model filenames; the face
analysis guard checks that FaceFusion content-screening modules remain unloaded. Synthetic
tensor inputs ran through `ghost_1_256` and `crossface_ghost` using `CUDAExecutionProvider`; their
output shapes were `(1, 3, 256, 256)` and `(1, 512)`, and all output values were finite.

An in-memory OpenCV drawing and a separate three-frame synthetic blank video were also passed
through the actual GPU face-analysis path. The drawing produced zero detections; the blank scan
exited with the expected “no target face detected” result. Neither used personal media or invoked
a content classifier.

## Synthetic end-to-end conversion

On the same RTX 4050, the actual CLI completed a local GPU conversion using an AI-generated
fictional adult face as the video target and a separate AI-generated fictional adult portrait as
the reference. The input was a five-second, 1280x720 synthetic clip with generated tone audio,
119 frames at 24000/1001 fps. The command used `--consent-confirmed --content-rating
non-explicit --character 0`; it exited 0 with the runtime import guard enabled and reported one
retained audio stream.

FFprobe on the result reported 1280x720, 119 video frames, 24000/1001 fps, video and audio start
times of 0, and the `AI-ALTERED face replacement` metadata comment. The visible `AI-ALTERED`
overlay was confirmed by viewing a decoded result frame. The run exercised GPU face detection,
reference selection, GHOST conversion, compositing, audio remux, labeling, and output validation.

The original audited CLI command shape was:

```powershell
$testDir = Join-Path $env:TEMP 'f12-synth-cuE1kt'
.venv-gpu\Scripts\python.exe -m face_replace `
  --video "$testDir\movie.mp4" `
  --face "$testDir\reference.png" `
  --output "$testDir\result-guard.mp4" `
  --consent-confirmed --content-rating non-explicit --character 0
```

Reproduce both audited CUDA runs with the pinned synthetic fixtures using:

```powershell
$fixtureDir = Join-Path $env:TEMP 'f12-synth-cuE1kt'
.venv-gpu\Scripts\python.exe scripts\f12_synthetic_smoke.py `
  --fixture-dir $fixtureDir `
  --evidence-out docs\F12_LOCAL_GPU_EVIDENCE.json
```

The script checks fixture hashes before processing, creates its output in a temporary directory,
captures the exact CLI argument shape and completed worker audits, drives the GUI
scan/selection/swap, and stores sanitized ffprobe summaries and output hashes in
[F12_LOCAL_GPU_EVIDENCE.json](F12_LOCAL_GPU_EVIDENCE.json). It also compares the face region with
the source frame to verify a localized synthetic face change. No media paths are written there.
The F12 contract runs this same script as the `synthetic-gpu-smoke` gate. H.264 output hashes are
run-specific and may change between runs; the contract checks output properties and runtime
audits instead of requiring identical hashes.

SHA-256 for the generated input fixtures:

| Artifact | SHA-256 |
|---|---|
| Synthetic movie | `33d0e847dd49683978dac2604d36f493c5ae823574f5203220bc637c44edaef4` |
| Synthetic target portrait | `4fde9ce3ec119340e66d512bd25fdebb49c70febf657249ea1937fa1c508bbd0` |
| Synthetic reference portrait | `6deb6a8ca665a1dd761036342eb0cbfe12c8147bff35c9a39396e073310531c0` |

The first run exposed a frame-count discrepancy: FaceFusion estimated 120 frames from container
duration while the video stream had 119. The pipeline now passes its FFprobe video-stream count to
the worker, and the end-to-end run then completed. A unit test covers the extra-frame and exact
count behavior.

The reproducible audited CLI run reported only the five pinned GPU models and confirmed CUDA
providers for every inference session through the end of the worker. It also confirmed that
content-screening modules were not imported. A Tkinter driver scanned the synthetic clip, selected
its detected character, and completed a GUI GPU swap. Machine-readable output summaries, audit
messages and hashes for both runs are in [F12_LOCAL_GPU_EVIDENCE.json](F12_LOCAL_GPU_EVIDENCE.json).
I inspected a decoded frame from the audited CLI output and confirmed its visible `AI-ALTERED`
overlay. FFprobe confirmed 1280x720, 119 frames at 24000/1001 fps, one retained audio stream,
video/audio start at zero, and the AI-ALTERED metadata comment for both outputs.

This is synthetic evidence only. No real-person media was used. It does not establish real-media
identity matching or visual quality, the SPEC R4–R11 quality checks, long-video stability or
throughput, retry/resume on GPU, clean-install behavior, or Linux behavior. Those items remain
unverified.
