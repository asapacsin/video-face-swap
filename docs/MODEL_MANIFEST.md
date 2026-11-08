# Face replacement model manifest

The runtime worker uses FaceFusion's unmodified face geometry and image-analysis helpers with
local ONNX sessions. It does not use FaceFusion's content analyzer, demographic classifier,
headless workflow, or face-swapper preflight. `scripts/fetch_models.py` fetches the required
models and verifies each ONNX file against its paired upstream CRC32 file and a pinned SHA-256
digest. The model files are
stored in the pinned FaceFusion checkout's `.assets/models` directory.

| Model | Purpose | Release | Upstream license metadata | Size / SHA-256 |
|---|---|---|---|---|
| `yoloface_8n` | Face detection | `models-3.0.0` | GPL-3.0 | 12,659,761 B / `821cdbb1e65fbbabdde7dd0933f754797a343e56fd962729c61ffcefcd135929` |
| `fan_68_5` | Face landmarks | `models-3.0.0` | OpenRAIL-M | 944,321 B / `ea05f7d9d014ae1d9cdc6e9c72643d8f00d198b5db3fdd384148ddff499a613e` |
| `arcface_w600k_r50` | Face recognition | `models-3.0.0` | Non-Commercial | 174,388,474 B / `f1f79dc3b0b79a69f94799af1fffebff09fbd78fd96a275fd8f0cbbea23270d1` |
| `ghost_1_256` | GHOST face replacement | `models-3.0.0` | Apache-2.0 | 514,902,950 B / `4eac28021ae80128e9262080cf0da850e3190e41862650c9e805f9a516b3e924` |
| `crossface_ghost` | GHOST identity embedding conversion | `models-3.4.0` | OpenRAIL-M | 22,083,800 B / `9ec5862d9ff1f723a7380ea89baf87cddec9c56670d4db766702657939284957` |

The previous release used FaceFusion content-classification models, demographic classification,
voice extraction, masking models, and HyperSwap. They are no longer required by this worker and
are excluded from a fresh model fetch. Existing cached files can remain on disk. They are never
opened by the GHOST worker.

The labels above come from upstream metadata. The pinned FaceFusion checkout's `LICENSE.md`
identifies its code license as OpenRAIL-AS, but the file in that checkout contains only the
license name and copyright line, not the terms. The GHOST project publishes Apache-2.0 code and
describes the method as research; this does not settle rights for every converted model artifact.
ArcFace metadata is Non-Commercial, and CrossFace metadata is OpenRAIL-M. The owner's approved
scope is personal non-commercial research. These source labels are evidence, not a legal
conclusion that the combined backend is cleared for other uses.

Sources: [pinned FaceFusion license](https://github.com/facefusion/facefusion/blob/72470819a0373be3388b3929c8f8f311f418fc3c/LICENSE.md),
[FaceFusion model license table](https://docs.facefusion.io/introduction/licenses),
[GHOST project](https://github.com/ai-forever/ghost), and the pinned FaceFusion model metadata
at `upstream-facefusion/facefusion/processors/modules/face_swapper/core.py`.
