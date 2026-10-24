"""Real CUDA MatMul check without face models, using ORT profiling to reject CPU fallback."""

import ctypes
import ctypes.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def check_runtime():
    # Loader must see the approved image's CUDA/cuDNN libraries; no driver changes.
    cuda = ctypes.CDLL(ctypes.util.find_library("cudart") or "libcudart.so.12")
    version = ctypes.c_int()
    if cuda.cudaRuntimeGetVersion(ctypes.byref(version)) != 0:
        raise RuntimeError("Cannot read actual CUDA runtime version")
    if not 12000 <= version.value < 13000:
        raise RuntimeError("This installer targets CUDA 12, not CUDA 11 or 13")
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"], text=True
    ).strip()
    for line in gpu.splitlines():
        if float(line.rsplit(",", 1)[1]) >= 10 and version.value < 12080:
            raise RuntimeError("Blackwell requires CUDA 12.8 or newer binaries")
    cudnn = ctypes.CDLL(ctypes.util.find_library("cudnn") or "libcudnn.so.9")
    cudnn.cudnnGetVersion.restype = ctypes.c_size_t
    cudnn_version = cudnn.cudnnGetVersion()
    if not 90000 <= cudnn_version < 100000:
        raise RuntimeError("cuDNN 9 required")
    return gpu, version.value, cudnn_version


def main():
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper

    if ort.__version__ != "1.24.4":
        raise RuntimeError("Expected pinned CUDA 12 ONNX Runtime 1.24.4")
    gpu, runtime_version, cudnn_version = check_runtime()
    with tempfile.TemporaryDirectory(prefix="face-replace-kernel-") as tmp:

        def tensor(n):
            return helper.make_tensor_value_info(n, TensorProto.FLOAT, [32, 32])

        graph = helper.make_graph(
            [helper.make_node("MatMul", ["a", "b"], ["c"])],
            "cuda-check",
            [tensor("a"), tensor("b")],
            [tensor("c")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 9
        onnx.checker.check_model(model)
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(Path(tmp) / "kernel")
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        session = ort.InferenceSession(
            model.SerializeToString(), sess_options=options, providers=["CUDAExecutionProvider"]
        )
        session.disable_fallback()
        a = np.arange(1024, dtype=np.float32).reshape(32, 32) / 1024
        b = np.eye(32, dtype=np.float32)
        np.testing.assert_allclose(session.run(None, {"a": a, "b": b})[0], a @ b)
        events = json.loads(Path(session.end_profiling()).read_text())
        if not any(
            e.get("args", {}).get("provider") == "CUDAExecutionProvider"
            and e.get("args", {}).get("op_name") == "MatMul"
            for e in events
        ):
            raise RuntimeError("No actual CUDA MatMul evidence; provider listing is insufficient")
    print(
        json.dumps(
            {
                "kernel": "CUDA MatMul PASS",
                "gpu": gpu,
                "cuda_runtime": runtime_version,
                "cudnn": cudnn_version,
                "face_inference": "NOT_VERIFIED",
            }
        )
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--preflight"]:
        print(check_runtime())
    else:
        main()
