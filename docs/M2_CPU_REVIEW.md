# M2 CPU pipeline review — 2026-10-24

**Result: PASS for synthetic CPU orchestration and output handling. Overall face
replacement and GPU inference: NOT_VERIFIED.**

Independent reviewer ran all 14 unit/integration tests, Ruff lint/format and reviewed
the worker and pipeline changes. It additionally exercised process timeout cleanup
(including a child process), a `30000/1001` frame rate with silent input, and a
destination collision. Existing destination bytes were preserved and scratch data
was cleaned.

The tests use generated color media and a fake worker. They verify labeling metadata,
decodable output, frame rate/count/dimensions, audio stream count, worker failure,
and atomic no-clobber publication. They do not prove a visible label on real footage,
FaceFusion model downloads are disabled at runtime, safety classifiers execute
correctly, identity replacement works, or CUDA is compatible.

The fixed worker pins the FaceFusion revision, rejects modified tracked upstream
files, empties download providers before normal checks, screens the reference and
every decoded target frame using upstream safety code, rejects ambiguous faces,
initializes private jobs/temp paths and propagates the upstream result. Runtime
verification requires the upstream Python dependencies, reviewed model licenses,
approved local weights, authorized test assets and a compatible GPU. Those are
separately gated; no GPU was rented and no weights were downloaded.
