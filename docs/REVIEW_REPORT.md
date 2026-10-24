# Review report — 2026-10-24

**Milestone 1: PASS within source-preparation scope. Overall project: NOT_VERIFIED.**

An independent read-only review agent inspected the deliverable and executed checks
in this environment. It returned PASS for M1, and explicitly NOT_VERIFIED for the
complete face-replacement project. No external paid reviewer CLI was configured.
This is not a claim that the ai-native-engineering controller was installed or run.

## Actual evidence

- 12 tests passed via `.venv/bin/python -m unittest discover -s tests -v`.
  Tests use FFmpeg-generated color video with AAC audio and a generated still image.
  Verified dimensions, frame rate, audio presence, byte-for-byte input preservation,
  no output writes, consent refusal, explicit declaration refusal, corrupt/missing
  media, missing FFprobe, probe timeout, overwrite refusal, help and inference refusal.
- Ruff lint and format checks passed. Both Bash scripts passed `bash -n`.
- `bash scripts/setup_source.sh` completed twice; editable install/help succeeded.
  `.venv/bin/python -m pip check` reported no broken requirements.
- GPU `--verify-only` exited 1 with `Missing prerequisite: nvidia-smi` as expected.
  This demonstrates truthful refusal, not a successful remote/GPU check.
- Upstream detached SHA: `72470819a0373be3388b3929c8f8f311f418fc3c`; clean tree,
  including unmodified safety analysis. No models downloaded.
- Project Git branch `main` is unborn; files are untracked and no remote is configured.
  Git has zero tracked media/weights/credentials because no files have been staged.
  Ignore rules exclude the nested upstream checkout, virtual environments and media.

An initial help test accidentally invoked the global Python instead of the installed
venv. It was corrected to use sys.executable and the complete suite reran successfully.
An initial lint issue was corrected. No assertions were disabled or weakened.
Installer review identified upstream's ignored pip exit status: the wrapper now
checks every pinned direct dependency explicitly, performs CUDA/cuDNN preflight
before installation, and requires actual CUDA MatMul numerical/profiling evidence.
The independent reviewer inspected these changes and found no remaining M1 blocker.

## Not verified / not yet implemented

No GPU was rented. Remote installation, idempotence on the real GPU server, CUDA
kernel execution, full face inference, runtime safety classification, output overlay,
audio retention and throughput are NOT_VERIFIED. M2 output pipeline is not implemented.
No celebrity, private or explicit face data was used. Color fixtures are not evidence
of face-swap quality. Full project PASS requires the user's labeled face-video E2E test.
See README for exact setup/sync/dry-run commands and future E2E command/required assets.

## Provenance and persistence

`REVIEWED_FILES.sha256` fingerprints the delivered source/config/docs (excluding this
report and the manifest itself). Use `sha256sum -c docs/REVIEWED_FILES.sha256` from
the project root to detect later changes. There is no project commit yet.

Cloud configuration `install_script` and `start_skill` were successfully saved as a
draft. They point to `/workspace/face-replace`; no repository remote/membership was
invented. Review/save the environment settings and publish to capture local source
and installed dependencies. Publication and restoration in a fresh task were not run.
The scripts cannot recover this unpushed project without its directory/snapshot.

Next: implement and CPU-test M2's backend/safety/output adapter and prepare a precise
model license/download manifest before requesting GPU rental approval. Budget spent
on cloud GPUs in this task: RMB 0.
