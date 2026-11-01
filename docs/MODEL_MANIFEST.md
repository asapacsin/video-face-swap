# FaceFusion model manifest

Pinned upstream: `72470819a0373be3388b3929c8f8f311f418fc3c`.
Upstream constructs URLs under `https://github.com/facefusion/facefusion-assets/releases/download/RELEASE/FILE`
or `https://huggingface.co/facefusion/RELEASE/resolve/main/FILE`.
Each ONNX file has a paired `.hash` file. Upstream validates CRC32. The owner approved
downloading these 12 files (1,465,626,594 bytes) from the GitHub release URLs on
2026-11-01; `scripts/fetch_models.py` fetches them and verifies CRC32.
The non-commercial and research licenses below limit the output to personal use.

| Model | Purpose | Release | Upstream metadata license | Size / SHA-256 |
|---|---|---|---|---|
| nsfw_1 | Explicit content screening | models-3.3.0 | Apache-2.0 | 80,414,194 B / `3debfd72763bc931dd308f57d1e9299ae13bb309d14ddbdc17ce8f1fd42f96af` |
| nsfw_2 | Explicit content screening | models-3.3.0 | Apache-2.0 | 22,489,928 B / `5aae1806ce44e3c27fab0a7d44eed581dd25ca71a4387ff809a191c98239c1be` |
| nsfw_3 | Explicit content screening | models-3.3.0 | MIT | 358,188,033 B / `778139fb930240ac39812a1480295bfc5dc2cec8a8e46ee6b453481d600c0fa2` |
| fairface | Face age/gender/race classification dependency | models-3.0.0 | CC-BY-4.0 | 85,170,772 B / `b176b1e4b0739496e7943af7ba4ff306271fecf837bca5ec5d9a48201959db37` |
| yoloface_8n | Face detection | models-3.0.0 | GPL-3.0 | 12,659,761 B / `821cdbb1e65fbbabdde7dd0933f754797a343e56fd962729c61ffcefcd135929` |
| 2dfan4, fan_68_5 | Face landmarks | models-3.0.0 | MIT; OpenRAIL-M | 97,904,803 B / `678c6fa539d52335a31c980feefdf4a6e02d781d83dce00af8a894f114557285`<br>944,321 B / `ea05f7d9d014ae1d9cdc6e9c72643d8f00d198b5db3fdd384148ddff499a613e` |
| xseg_1, bisenet_resnet_34 | Masking | models-3.1.0; models-3.0.0 | GPL-3.0; MIT | 70,324,286 B / `c4d1498b8a03b5fe2a3a5d2ef2a0402ab03bd51edaf5b2d8d5fb764702a97dd3`<br>93,632,546 B / `4a0b8c958a3c938913bd06a8365dbb3c8761afba6ecbf0d14b3b1f77eb230c96` |
| arcface_w600k_r50 | Face recognition | models-3.0.0 | Non-Commercial | 174,388,474 B / `f1f79dc3b0b79a69f94799af1fffebff09fbd78fd96a275fd8f0cbbea23270d1` |
| kim_vocal_2 | Voice extraction (unconditionally prechecked by swapper) | models-3.0.0 | Non-Commercial | 66,766,794 B / `fbde1fd7d35a6b1785add2225ce03255a01ccef6c8a5fb7b2ccd4e0c1e9097f7` |
| hyperswap_1a_256 | Face replacement | models-3.3.0 | ResearchRAIL | 402,742,682 B / `c0e98a8a03a238f461ed3d2570e426b49f46745ee400854a60dceeb70c246add` |

Each listed item requires its `.onnx` and `.hash` companion. Licenses above are
upstream metadata labels, not a substitute for reviewing full terms. The upstream
source license is OpenRAIL-AS. Non-commercial and research restrictions may rule
out some intended uses; clarify intended use and review every full license before
selecting a model.

The pinned face swapper's `pre_check()` unconditionally checks the safety analyzer,
classifier, detector, landmarker, masker, recognizer and voice extractor. Missing
files fail closed if its download provider list is empty. A worker can set that
list empty before model checks without modifying upstream safety code. The reference
image must be screened explicitly: upstream swapper preprocessing does not do so.
Upstream video screening samples approximately one frame per second; stricter
all-frame screening remains required by this project. Safety classification is
fallible and does not establish consent or age.

Complete model terms have not been reviewed; only the upstream metadata labels are recorded.
