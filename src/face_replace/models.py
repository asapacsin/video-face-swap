"""Pinned model and backend identities shared by setup and runtime."""

import hashlib

FACEFUSION_REVISION = "72470819a0373be3388b3929c8f8f311f418fc3c"

MODEL_RELEASES = {
    "yoloface_8n": "models-3.0.0",
    "fan_68_5": "models-3.0.0",
    "arcface_w600k_r50": "models-3.0.0",
    "ghost_1_256": "models-3.0.0",
    "crossface_ghost": "models-3.4.0",
}

MODEL_SHA256 = {
    "yoloface_8n": "821cdbb1e65fbbabdde7dd0933f754797a343e56fd962729c61ffcefcd135929",
    "fan_68_5": "ea05f7d9d014ae1d9cdc6e9c72643d8f00d198b5db3fdd384148ddff499a613e",
    "arcface_w600k_r50": "f1f79dc3b0b79a69f94799af1fffebff09fbd78fd96a275fd8f0cbbea23270d1",
    "ghost_1_256": "4eac28021ae80128e9262080cf0da850e3190e41862650c9e805f9a516b3e924",
    "crossface_ghost": "9ec5862d9ff1f723a7380ea89baf87cddec9c56670d4db766702657939284957",
}

_identity = "|".join(
    [FACEFUSION_REVISION] + [f"{name}:{MODEL_SHA256[name]}" for name in MODEL_RELEASES]
)
BACKEND_REVISION = f"ghost-{hashlib.sha256(_identity.encode()).hexdigest()[:16]}"
