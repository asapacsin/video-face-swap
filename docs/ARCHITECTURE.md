# Decision: local CLI + pinned FaceFusion on a rented NVIDIA GPU

Decision date: 2026-10-24. Scope: milestone 1 preparation. The Python standard-library
CLI validates consent and inspects local files with FFprobe. No backend is imported
in dry-run. Future inference will be a subprocess adapter to the pinned FaceFusion
headless CLI, followed by FFmpeg labeling, audio remuxing and output verification.
Local media stays on the operator's machine or explicitly selected GPU host.

## Candidate comparison (documented capabilities, not measured quality)

| Candidate | License and model constraints | GPU/quality/maintenance evidence | Decision |
|---|---|---|---|
| FaceFusion | Pinned LICENSE.md says OpenRAIL-AS; this short file does not include full use terms. Each selected model needs separate license review. | Maintained headless video pipeline, safety analysis, CUDA 12/13 installers; selected source commit dated 2026-09-30. No quality or B200 measurements here. | Selected provisionally for source integration; best fit to one reference image without training. |
| InsightFace | Official README: MIT code; downloaded pretrained models limited to non-commercial research unless separately licensed. | CUDA/CPU face-analysis toolkit and swap examples; README has 2026-10-03 updates. Requires our own video/safety/labeling integration. Quality unmeasured. | Higher integration cost and model licensing friction. |
| deepfakes/faceswap | GPL-3.0 code (official LICENSE); model/data rights remain separate. | Official workflow extracts, trains and converts; CUDA GPU recommended. README retrieved; current release cadence not independently established. Quality depends on training/data. | Training workflow exceeds single-reference MVP scope and budget. |

Official sources inspected: https://github.com/facefusion/facefusion (pin
72470819a0373be3388b3929c8f8f311f418fc3c),
https://github.com/deepinsight/insightface/blob/master/README.md,
https://github.com/deepfakes/faceswap/blob/master/README.md.
FaceFusion's hosted https://docs.facefusion.io/installation returned HTTP 403 in
this environment. Its pinned `facefusion/installer.py`, requirements and CI are
the available installation reference. Do not claim the hosted docs were read.

## Implementation boundary

M1 implements consent, explicit-content declaration refusal, input validation,
media inspection and a JSON execution plan. No media transformation occurs.
M2 must add actual safety screening of both source and reference through upstream
checks, prohibit processing if safety checks are unavailable, use an approved model,
and call the headless pipeline without disabling checks. Attestation cannot detect
age, authorization or explicit content by itself. The dry-run marks screening NOT_RUN.

M2 output acceptance: one readable MP4, visible persistent `AI-ALTERED` overlay,
metadata identifying AI alteration, matching frame rate/order, approximate resolution,
source audio retained (copy compatible codecs, otherwise explicit AAC transcode),
duration checked against source, atomic final rename, private temporary workspace
cleaned after success and on controlled failure. No raw media or private filenames in logs.
These are requirements, not implemented output behavior in M1.

## GPU decision and risk

Prefer an approved Linux image with Python 3.12, CUDA 12.8+, cuDNN 9 and FFmpeg.
The pinned upstream CUDA-12 installer selects ONNX Runtime GPU 1.24.4.
B200 is Blackwell (compute capability 10.x); H800 is Hopper (9.x). Verify actual
hardware with nvidia-smi. Driver-reported "CUDA Version" is only a compatibility
ceiling; verify the loaded runtime via cudaRuntimeGetVersion. Never replace drivers
or system CUDA automatically. CUDA-13 images are outside this initial script's scope.
Minimum CUDA 12.8 for Blackwell is a prerequisite, not proof all kernels work.
The setup's real ONNX MatMul must have CUDA profiling evidence and correct numeric
output; even that does not prove all face models or safety models work.

Top risks: model usage rights, upstream package availability, binary compatibility,
identity preservation/occlusion quality, false negatives in safety analysis, remote
disk privacy, and rental startup costs. Model license/size approvals and paid GPU
access are unresolved. No model has been chosen or downloaded.

Smallest discriminating experiment after approval: 3–5 second licensed adult or
synthetic face clip with one face and audio, one authorized reference; first kernel
check, then full pipeline, then inspect all frames for label/swap quality and compare
duration/frame rate/audio. Measure elapsed time and peak VRAM on the actual host.
If FaceFusion fails quality or license constraints, reconsider the backend before
spending on longer clips. Swap adapter is a subprocess boundary to keep replacement cheap.
