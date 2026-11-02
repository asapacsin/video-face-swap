# Outstanding work

- **Correction rounds beyond the limit.** AGENTS.md allows three. The owner approved more in chat on
  2026-11-02, which the repository cannot show; the contract was not changed.
- **Stopping with a hard kill.** SIGKILL cannot be caught, so it can still leave a worker running on
  Linux. SIGTERM and SIGHUP end it.
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
