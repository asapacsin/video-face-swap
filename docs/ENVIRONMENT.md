# Observed Codex Cloud environment

Inspected 2026-10-24. Initial directory `/workspace`; project `/workspace/face-replace`.
The child directory did not exist and was created. Git initialized on `main` with
no commits and no remote. Existing `/workspace/video-face-swap` was preserved.

| Component | Observed |
|---|---|
| OS | Debian GNU/Linux 13.6 (trixie), Linux x86_64 |
| Git | 2.52.0 |
| Python | 3.12.14 |
| FFmpeg and FFprobe | 7.1.5-0+deb13u1 |
| Disk before installation | 32 GB overlay, about 30 GB available |
| NVIDIA GPU | No nvidia-smi executable or /dev/nvidia devices |
| Development tools | project venv, editable package, Ruff 0.14.2 |
| Git access | FaceFusion clone succeeded through existing HTTPS configuration |
| Upstream | detached 72470819a0373be3388b3929c8f8f311f418fc3c, unmodified |

`bash scripts/setup_source.sh` checks Python 3.12 and FFmpeg, preserves upstream
changes by refusing a mismatch, and installs the wrapper and development tooling.
Wrapper dependencies are standard library + FFprobe; inference dependencies stay
in the separate upstream checkout and GPU virtual environment. Build backend and
Ruff are pinned. Upstream pins direct GPU dependencies; transitive GPU dependencies
are not fully locked. Capture a tested lock after real GPU validation.

Issues: hosted FaceFusion docs returned HTTP 403. Network status did not establish
fully enforced policy; actual successful Git/package requests establish access for
those operations. No credentials were needed. Pip's shared cache was unwritable;
installation succeeded with caching disabled. GPU verify-only correctly exits 1
with `Missing prerequisite: nvidia-smi` here. This is expected, not GPU validation.
No GPU packages, model weights or private media were installed.
