"""本地模型推理设备解析与可审计运行时快照。"""

from __future__ import annotations

from typing import Any


def resolve_torch_device(requested: str = "auto") -> str:
    """将 ``auto/cuda/cpu`` 解析为实际 torch device。"""
    normalized = str(requested or "auto").strip().lower()
    if normalized not in {"auto", "cuda", "cpu"}:
        raise ValueError("runtime.device must be one of: auto, cuda, cpu")

    try:
        import torch
    except ImportError:
        if normalized == "cuda":
            raise RuntimeError("runtime.device=cuda requires PyTorch with CUDA support")
        return "cpu"

    cuda_available = bool(torch.cuda.is_available())
    if normalized == "cuda" and not cuda_available:
        raise RuntimeError("runtime.device=cuda was requested, but torch.cuda.is_available() is false")
    if normalized == "auto":
        return "cuda" if cuda_available else "cpu"
    return normalized


def runtime_snapshot(requested: str, actual: str, batch_size: int) -> dict[str, Any]:
    """返回不含病例数据的设备与 PyTorch provenance。"""
    snapshot: dict[str, Any] = {
        "requested_device": requested,
        "actual_device": actual,
        "batch_size": batch_size,
        "torch_version": "unavailable",
        "torch_cuda_version": None,
        "gpu_name": None,
    }
    try:
        import torch
    except ImportError:
        return snapshot

    snapshot["torch_version"] = str(torch.__version__)
    snapshot["torch_cuda_version"] = torch.version.cuda
    if actual == "cuda" and torch.cuda.is_available():
        snapshot["gpu_name"] = torch.cuda.get_device_name(0)
    return snapshot
