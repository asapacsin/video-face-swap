# Historical benchmark: retired FaceFusion + HyperSwap backend

These runs describe the previous backend and are not performance or quality evidence for GHOST.
The current GPU evidence includes synthetic end-to-end CLI and GUI conversions on the RTX 4050,
documented in [F12_LOCAL_GPU_SMOKE.md](F12_LOCAL_GPU_SMOKE.md). Real-person identity quality,
long-video conversion, and throughput remain NOT_VERIFIED.

All runs use FaceFusion 7247081 and hyperswap_1a_256.

## Rented server (phase 1), measured 2026-11-01

32 GB NVIDIA card behind HAMi, Ubuntu 22.04, CUDA 13, ONNX Runtime 1.30.0, serial screening, 4 swap
threads. Cost uses the estimated ¥3.26/hour, which the owner has yet to confirm.

| Input | Total time | Overall rate | Peak VRAM | Cost |
|---|---|---|---|---|
| 65 s sample, 1364x720, run 1 | 703 s | About 2.3 frames/s | Not sampled | ¥0.64 |
| 65 s sample, run 2 | 704 s | About 2.3 frames/s | About 7.9 GB | ¥0.64 |
| 10 min looped sample, killed after segment 1, then resumed (15,000 frames) | 4203 s, including a 60 s wait before the kill | About 3.6 frames/s | 7757 MiB | ¥3.81 |
| 15 s of Charade (1963), 1280x720, scene changes, at least 3 candidate characters (361 frames) | 332 s, scan and swap | About 1.1 frames/s | 8128 MiB | ¥0.30 |

Screening ran at about 8 frames/s. Short clips look slow overall because model loading and the
finishing steps take a larger share of the time. No driver error occurred in these runs.

**Extrapolation, not a measurement:** at 2.3–3.6 frames/s, a 2-hour, 24 fps movie (172,800 frames)
takes about 13–21 hours on this server, or about ¥43–68. Real films may differ, since time depends
on how many faces appear.

**Disk:** a 2-hour job needs about 43 GB free at its peak. 15 s of the film, encoded on the CPU,
gave 1.45 MB/s at the segments' CRF 5 and 0.4 MB/s at CRF 18. The peak holds the 720p copy, the
segments, their joined copy and the labeled result: 3.7 MB/s, or 6 MB/s after a 50% margin and
rounding up. It is not a measured job size, and grainier films may need more.

## Local RTX 4050 Laptop (6 GB), measured 2026-11-01

Driver 610.78, i5-12450H, Windows 11, upstream's sample media. These predate the segmented job
code, and serial screening has run on this laptop only on the 10.8-second clip.

| Input | Result on 4050 | Peak VRAM |
|---|---|---|
| 10.8 s, 1364x720 output, serial screening | 185 s total; screening about 3.9 frames/s, swapping about 8.6 frames/s | 4183 MiB |
| 10.8 s, same clip, 4 screening threads | 110 s total; screening 11.2 frames/s | Not sampled |
| 10.8 s, 2084x720 two-person MKV with audio, 4 screening threads | 164 s total | 5399 MiB |
| 65 s, same face clip looped, 4 screening threads | Stopped with CUDA error 700. One run stopped at 48.6% of screening and diagnostic runs after about 250 frames; those logs were not kept | Not recorded |

The server's peaks are not 4050 results. Freeing the screening models after each screening pass
lowered the film clip's peak from 8128 MiB to 6093 MiB on the server (334 s, one run each, on a
32 GB card), and the output file was byte-identical to the earlier run. The remaining cause of the
peak was not measured. 6093 MiB is close to the 4050's 6141 MiB, so phase 2 must measure the limit
on the 6 GB card.
