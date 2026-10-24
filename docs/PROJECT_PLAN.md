# Face replacement MVP — project plan

Date: 2026-10-24. Project: `/workspace/face-replace`.

Build a Python CLI that takes one video and one authorized adult reference image,
replaces one consenting adult's face, and produces a new MP4 with retained audio
and a persistent `AI-ALTERED` label. Use Codex Cloud for development and an approved
rented Linux NVIDIA GPU for inference. Total GPU budget: approximately RMB 200.

## Current baseline

M1 source preparation passed independent review: pinned FaceFusion source,
installable CLI, consent/input validation, media dry-run, remote bootstrap scripts,
12 passing CPU tests and reproducible commands. See `REVIEW_REPORT.md` for evidence.
Actual face inference and the full project remain **NOT_VERIFIED**.

FaceFusion is provisionally selected at
`72470819a0373be3388b3929c8f8f311f418fc3c`. Its source and safety checks remain
unmodified. Model selection and full license review are outstanding. The project
has no remote; source persistence currently depends on this workspace/snapshot.

## MVP boundaries and acceptance

- One local source video, one reference image, one selected identity and one new MP4.
  Initially reject ambiguous multiple-face selection rather than silently choosing.
- Require affirmative consent; reject known explicit content. Screen both inputs
  using the backend's supported safety checks, and refuse inference if screening
  cannot run. Screening does not establish consent or age; operator attestation is
  still required. Preserve all upstream restrictions.
- Keep source frame order, frame rate and approximate resolution. Initially support
  constant frame rate; detect and reject unsupported variable-rate inputs with an
  actionable error until their timing can be preserved and tested.
- Preserve source audio streams where MP4-compatible; document any required AAC
  conversion. Handle silent inputs explicitly. Do not silently drop audio.
- Burn `AI-ALTERED` into every output frame and add alteration metadata where the
  container supports it. Verify readable video, duration, frame rate, dimensions,
  frame count and expected audio before publishing output.
- Never overwrite input or existing output. Use a private temporary directory and
  publish the validated file without clobbering a concurrently created destination.
  Clean temporary media on success, errors and handled interruption; document
  manual cleanup after forced termination or host failure.
- Keep media on the operator's machine or approved GPU host. No default analytics
  or face-processing services. Logs contain stages, timings and generic error codes,
  without media, credentials, private paths or captured sensitive backend output.

Training, web UI, batch processing, live video, arbitrary identity selection and
GPU-specific optimization are outside this MVP. Prohibited uses remain as described
in `PRIVACY_AND_CONSENT.md`.

## Delivery sequence

| Stage | Deliverable | Exit evidence | Dependency |
|---|---|---|---|
| M1 — complete | Source setup, preflight CLI, tests and GPU bootstrap | Existing independent M1 PASS | None |
| M2a — backend readiness | Exact adapter contract, model/license/download manifest and failure policy | Every required model has an origin, version, purpose, rights evidence, download size and integrity verification method; upstream safety entry points mapped | Pinned source inspection |
| M2b — local pipeline | Backend subprocess adapter, safety gates, output finalization, cleanup and progress/errors | Meaningful CPU tests and synthetic media integration checks pass; no automatic weight downloads | M2a technical contract |
| M2c — independent readiness review | Reviewed code, updated setup/run instructions and concrete approval packet | Reviewer PASS for CPU preparation, with GPU and face quality explicitly NOT_VERIFIED | M2a and M2b |
| M3a — approved GPU deployment | Reproducible installation and real CUDA verification on actual host | Hardware/runtime recorded, repeat setup succeeds, numerical CUDA MatMul and profiling evidence pass | Explicit rental approval and secure access |
| M3b — face smoke and benchmark | Approved weights, authorized 3–5 second face clip with audio, real output and measurements | Safety path works, face transformation inspected, media/label/audio checks pass; throughput and peak VRAM recorded | Model download approval and M3a |
| M4 — review and handoff | Independent end-to-end review, reproducibility guide, final status and cleanup | All completion criteria below have actual evidence; otherwise scoped BLOCKED/NOT_VERIFIED | M3b |

### M2a: settle integration and download requirements

Inspect the pinned headless CLI and its model, safety and execution-provider paths.
Identify whether upstream could download weights implicitly; prevent that before
approval without editing or bypassing safety checks. If upstream cannot provide
the required safety or offline execution guarantees, report the integration blocker
and revisit the backend decision before renting hardware.

Create `docs/MODEL_MANIFEST.md` with all face and safety model dependencies, exact
identifiers, official sources, size evidence, licenses/restrictions, checksums or
upstream integrity checks, and intended use. Unknown rights or size remain blockers
to requesting a concrete download approval. Do not download weights during research.

Record exact supported input constraints, target-selection behavior, timeout and
error handling in an M2 task contract before implementation.

### M2b: implement small, testable increments

1. Extract reusable media inspection and add a subprocess backend interface with
   bounded execution, cancellation, safe logs and explicit CUDA configuration.
2. Add both-input safety screening and required-model checks. Denied, missing or
   inconclusive checks must stop before transformation and output publication.
3. Implement FFmpeg labeling, metadata, audio handling, output verification and
   safe final publication using a private temporary workspace.
4. Connect the CLI and document exit codes, progress, supported media and recovery.

Test subprocess contracts with controlled fakes; test actual FFmpeg output using
generated color frames and tone. Cover screening refusal/unavailability, missing
models, backend failure/timeout, interruption, destination collisions, audio and
silent inputs, label filter application, corrupt output and cleanup. Distinguish
synthetic pipeline tests from real face inference in every report. CPU tests cannot
establish model quality, safety classifier accuracy or CUDA compatibility.

### M2c: prepare a concrete approval packet

An independent reviewer inspects changes and runs the applicable test/lint/format
and shell checks. Resolve blocking findings within the bounded retry budget.
Update README, architecture, open problems and review evidence; retain M1 evidence
as historical scope and regenerate fingerprints for the newly reviewed deliverable.

Present exact model download sizes/reasons and rental configuration for approval:
GPU/image, hourly tariff, billing granularity, storage/transfer charges, maximum
session duration, spending cap, shutdown procedure and failure allowance. Obtain
provider/SSH access through secure configuration, never tracked files or chat secrets.

## GPU execution and budget control

Choose the least expensive approved host that meets the pinned backend's needs.
B200 is an expected target, not a condition for success; any alternative needs an
explicit compatible configuration and approval. Prefer an existing CUDA 12.8+
and cuDNN 9 image for B200, with Python 3.12 and FFmpeg. Verify loaded libraries,
actual GPU identity and the existing real-kernel probe before loading face models.
Do not automatically replace system drivers or CUDA.

Proposed allocation within RMB 200: up to RMB 40 for setup/compatibility, RMB 100
for smoke tests and measurements, RMB 40 for one justified correction run, RMB 20
reserve. These are spending envelopes, not provider quotes or spending authority.
Revise them against actual tariff and billing minimums before approval; count all
billable setup, idle time, disk and transfer charges. Set a provider-side limit
where supported and a shutdown deadline before starting.

Run a 3–5 second authorized clip first. Inspect every frame for the label, identity
continuity, obvious artifacts and target selection; verify video and retained audio.
Record cold and warm elapsed times and sampled peak VRAM, with sampling limitations.
Only then consider a longer sample. Estimate five-minute processing from measured
sample time using `300 / sample_seconds * processing_seconds`, explicitly reporting
it as an extrapolation plus setup/transfer/billing overhead. Stop if the projected
remaining spend exceeds the approved cap or quality is inadequate. Stop billing
promptly after retrieving approved results and cleaning remote private media/storage.

## Engineering workflow and effort bounds

Use Planner → Executor → independent Reviewer → stop gate, as defined in `AGENTS.md`.
Before each implementation increment, write a bounded `TASK_CONTRACT.json` with
scope, acceptance, exact checks, evidence location and a time limit. Use 45–90 minute
development increments with at most three correction attempts; GPU increments also
need a hard paid-session deadline derived from the approved tariff. These are work
budgets, not delivery promises. At the limit, report completed scope and the exact
blocker rather than repeating attempts indefinitely.

M2a can proceed with no GPU spending. M2b can proceed once its technical contract is
clear; model rights/download approval gates actual weight use and the GPU trial.
M3 waits for explicit approvals and access. M4 documentation and CPU review can be
prepared earlier, but final acceptance waits for the real end-to-end result.

## Final completion gate

The independent reviewer may report full project PASS only when:

1. A new developer reproduces source setup and remote deployment from README;
   installation repeatability on the real host is demonstrated.
2. Help, dry-run, consent/refusal, invalid-input and pipeline checks pass.
3. Approved models process authorized adult non-explicit face media end to end.
4. The output is readable, visibly labeled, correctly timed, and retains audio;
   face transformation and target quality have been inspected.
5. Actual GPU/model/environment versions, measurements, test results and review
   findings are recorded without invented benchmarks or broad safety claims.
6. Git inspection confirms no credentials, private images, media, model weights
   or nested upstream history are tracked.

If access, licensing, assets or runtime checks block real execution, report
NOT_VERIFIED for that scope with exact missing prerequisites and prepared commands.
No project remote is configured until the user supplies its exact URL. Publishing
the cloud snapshot and verifying a fresh restore remain separate persistence checks.

Immediate next action: M2a — establish the backend/safety contract and exact model
manifest, then implement the CPU-testable pipeline before requesting GPU spending.
