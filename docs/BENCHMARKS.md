# Benchmark status: short clips measured; full-length movies NOT_VERIFIED

Measured 2026-11-01 on a local RTX 4050 Laptop (6 GB, driver 610.78), i5-12450H, Windows 11,
FaceFusion 7247081, hyperswap_1a_256, using upstream's sample media (a face in every frame).

| Input | Result on 4050 | Peak VRAM |
|---|---|---|
| 10.8 s, 1364x720 output, serial screening | 185 s total; screening about 3.9 frames/s, swapping about 8.6 frames/s | 4183 MiB |
| 10.8 s, same clip, 4 screening threads | 110 s total; screening 11.2 frames/s | Not sampled |
| 10.8 s, 2084x720 two-person MKV with audio, 4 screening threads | 164 s total | 5399 MiB |
| 65 s, same face clip looped, 4 screening threads | Stopped at 48.6% of screening; rate fell from about 8.5 to under 1 frame/s | Not recorded |

The 65 s slowdown is unexplained, so longer jobs may be much slower than the short-clip
rates. Extrapolating those rates, a two-hour 24 fps movie takes at least 10-20 hours.

No GPU was rented. Remote throughput and five-minute cost are unknown.

After rental and exact model approvals: record GPU model, driver, loaded CUDA/cuDNN,
ONNX Runtime, FaceFusion SHA, model identifiers/checksums/licenses, synthetic/licensed
asset provenance, dimensions, frame rate, duration, warmup and elapsed processing time.
Measure peak device memory with nvidia-smi sampling; note sampling can miss short peaks.
Record cold and warm timing separately. Estimate five-minute processing time as
300 / sample_duration_seconds * measured_processing_seconds, label it an extrapolation,
and add setup/upload/download/billing overhead using the approved provider's tariff.
Budget ceiling is approximately RMB 200. Agree the host, tariff, billing granularity,
disk charges, duration cap and shutdown procedure before incurring any cost.
