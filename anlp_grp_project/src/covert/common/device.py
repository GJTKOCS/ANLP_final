"""Device, dtype and hardware-profile helpers."""
from __future__ import annotations

import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def torch_dtype(hw: dict, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    want = hw.get("dtype", "bfloat16")
    if want == "bfloat16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def gpu_info() -> dict:
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return {"type": p.name, "mem_gb": round(p.total_memory / 2**30, 1)}
    return {"type": get_device().type, "mem_gb": 0}


def make_scaler(dtype: torch.dtype, device: torch.device):
    """GradScaler only for fp16 on CUDA, else None."""
    if dtype == torch.float16 and device.type == "cuda":
        return torch.amp.GradScaler("cuda")
    return None


def free_memory() -> None:
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
