# FaceFusion model manifest (source inspected; no weights downloaded)

Pinned upstream: `72470819a0373be3388b3929c8f8f311f418fc3c`.
Upstream constructs URLs under `https://github.com/facefusion/facefusion-assets/releases/download/RELEASE/FILE`
or `https://huggingface.co/facefusion/RELEASE/resolve/main/FILE`.
Each ONNX file has a paired `.hash` file. Upstream validates CRC32; obtain and
record SHA-256 and exact sizes before any download approval. The managed network
proxy returned 403 for Hugging Face metadata, so sizes and SHA-256 are unknown.

| Model | Purpose | Release | Upstream metadata license | Size / SHA-256 |
|---|---|---|---|---|
| nsfw_1 | Explicit content screening | models-3.3.0 | Apache-2.0 | Unknown |
| nsfw_2 | Explicit content screening | models-3.3.0 | Apache-2.0 | Unknown |
| nsfw_3 | Explicit content screening | models-3.3.0 | MIT | Unknown |
| fairface | Face age/gender/race classification dependency | models-3.0.0 | CC-BY-4.0 | Unknown |
| yoloface_8n | Face detection | models-3.0.0 | GPL-3.0 | Unknown |
| 2dfan4, fan_68_5 | Face landmarks | models-3.0.0 | MIT; OpenRAIL-M | Unknown |
| xseg_1, bisenet_resnet_34 | Masking | models-3.1.0; models-3.0.0 | GPL-3.0; MIT | Unknown |
| arcface_w600k_r50 | Face recognition | models-3.0.0 | Non-Commercial | Unknown |
| kim_vocal_2 | Voice extraction (unconditionally prechecked by swapper) | models-3.0.0 | Non-Commercial | Unknown |
| hyperswap_1a_256 | Face replacement | models-3.3.0 | ResearchRAIL | Unknown |

Each listed item requires its `.onnx` and `.hash` companion. Licenses above are
upstream metadata labels, not a substitute for reviewing full terms. The upstream
source license is OpenRAIL-AS. Non-commercial and research restrictions may rule
out some intended uses; clarify intended use and review every full license before
selecting a model. No weights are approved or downloaded by this manifest.

The pinned face swapper's `pre_check()` unconditionally checks the safety analyzer,
classifier, detector, landmarker, masker, recognizer and voice extractor. Missing
files fail closed if its download provider list is empty. A worker can set that
list empty before model checks without modifying upstream safety code. The reference
image must be screened explicitly: upstream swapper preprocessing does not do so.
Upstream video screening samples approximately one frame per second; stricter
all-frame screening remains required by this project. Safety classification is
fallible and does not establish consent or age.

Approval packet is incomplete until exact per-file byte sizes, SHA-256, official
download URLs and complete model terms are recorded. Do not download weights or
rent a GPU on the basis of this document alone.
