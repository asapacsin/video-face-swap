# Outstanding work

## Current GHOST backend — 2026-11-08

- F12 routes through GHOST and now has a synthetic end-to-end RTX 4050 conversion with an
  AI-generated fictional face, audio, exact frame count, timing, metadata, and visible label.
  Synthetic CLI and GUI scan/swap paths passed. No real-person media was authorized, so
  real-media identity, R4–R11 quality, full-length stability, cross-platform behavior,
  performance, and GPU retry/resume remain unverified.
- The owner has authorized revising F12 review criteria to permit the local backend without
  automated screening, with synthetic-only GPU evidence. An independent review of the revised
  criteria and current tree is still required before this milestone can be called complete.

The entries below document previous milestones and do not describe the active backend.

- **Phase 1 has not passed independent review.** The final review (2026-11-02) returned REWORK, and
  this work was committed unreviewed at the owner's request. Open findings:
  - The unit-test gate hit its 240 s limit during the review. Timed per module, nothing hangs:
    test_cli takes about 180 s, mostly ffprobe, whose startup costs 0.63 s per call with this
    machine's FFmpeg build. The suite fits under 240 s only with little margin under load; raising
    the limit is the owner's decision, since contracts must not be edited to pass.
  - Fixed, unreviewed: a failed character preview after publishing is now a warning, and an
    unreadable swapped segment or result is reported as a processing failure (exit 3).
  - No test starts a real worker and confirms SIGTERM or SIGHUP ends it, so that is unmeasured.
  - Joining many segments with audio is tested only on 2 segments; drift over a full film is
    unmeasured.
  - `.gitignore` is outside the contract's approved paths, and the contract's short `base_commit`
    fails the full-SHA check in `governance.py`.
- **Correction rounds beyond the limit.** AGENTS.md allows three. The owner approved more in chat on
  2026-11-02, which the repository cannot show; the contract was not changed.
- **Stopping with a hard kill.** SIGKILL cannot be caught, so it can still leave a worker running on
  Linux. SIGTERM and SIGHUP should end it, but no test runs a real worker.
- **Nothing is proven on the RTX 4050 since the fixes.** Segments and resume have only run on the
  server, and serial screening has run on the laptop only on a 10.8-second clip. Phase 2 repeats
  stability, VRAM and speed tests on Windows, and
  `scripts/setup_windows.ps1` has never run.
- **Model download from the server is very slow.** GitHub gave about 3 MB/minute, or about 8 hours for
  all models. A new server should have the models copied from the owner's PC.
- **Setup uses a moving FFmpeg build.** `setup_cloud.sh` installs BtbN's latest static build without
  a checksum, because old builds are removed from that release page, so a future run may differ.
- **CPU fallback during a job is not exercised.** Only refusal at start has been run for real.
- **No full movie has run.** By owner decision, phase 1 stops at a 15-second film clip.
- **The window has not been clicked by hand with a real job.** Its Scan, confirm and Start flow is
  tested with a fake job. The file pickers are untested.
- **Temperature pause (P5) is not built.** It belongs to phase 2.
- **Model rights:** review full OpenRAIL-AS terms and each exact model's rights. Only upstream
  license labels have been checked, so output stays personal-use.
- **Screening limit:** one refused frame stops a whole movie, so some films will not process.
- **Server:** the owner confirms the hourly rate, and whether a stopped server keeps its files.
- **GitHub CI and branch rules are not active:** a required status check must be configured
  separately.
