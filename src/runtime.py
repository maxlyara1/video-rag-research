from __future__ import annotations

import gc
import os
import warnings

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.95")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.90")

try:
    import torch
except ImportError:  # pragma: no cover - allows lightweight file-only workflows
    torch = None


def apply_runtime_config(cfg: dict[str, object] | None) -> None:
    if not cfg:
        return
    if cfg.get("enable_mps_fallback", True):
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    if cfg.get("mps_high_watermark_ratio") is not None:
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(cfg["mps_high_watermark_ratio"])
    if cfg.get("mps_low_watermark_ratio") is not None:
        os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = str(cfg["mps_low_watermark_ratio"])

    # Suppress a known noisy warning on Apple MPS: pin_memory has CUDA semantics and is ignored on MPS.
    # We scope this filter narrowly to avoid hiding other potentially useful UserWarnings.
    if cfg.get("device") in (None, "auto", "mps"):
        warnings.filterwarnings(
            "ignore",
            message=r".*'pin_memory' argument is set as true but not supported on MPS now.*",
            category=UserWarning,
        )


def is_cuda_device(device: str | None) -> bool:
    if not device:
        return False
    normalized = str(device).lower()
    return normalized == "cuda" or normalized.startswith("cuda:")


def is_mps_device(device: str | None) -> bool:
    if not device:
        return False
    normalized = str(device).lower()
    return normalized == "mps" or normalized.startswith("mps:")


def detect_torch_device(device: str | None = None) -> str:
    """Resolve the runtime device for local inference."""
    if device and device != "auto":
        return device

    if torch is None:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_torch_dtype(dtype: str | None, device: str) -> torch.dtype:
    """Map config-friendly dtype names to torch dtypes."""
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Install requirements.txt to run model inference.")
    if dtype and dtype.lower() != "auto":
        normalized = dtype.lower()
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported torch dtype: {dtype}")
        return mapping[normalized]

    if is_cuda_device(device):
        if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    if is_mps_device(device):
        return torch.float16
    return torch.float32


def resolve_cuda_devices(devices: str | None = None) -> list[str]:
    """Resolve the CUDA devices to use for multi-GPU inference."""
    if torch is None:
        return []
    if not torch.cuda.is_available():
        return []

    if not devices or devices == "auto":
        return [f"cuda:{index}" for index in range(torch.cuda.device_count())]

    resolved: list[str] = []
    for raw_device in devices.split(","):
        normalized = raw_device.strip().lower()
        if not normalized:
            continue
        if normalized.isdigit():
            resolved.append(f"cuda:{normalized}")
        elif normalized == "cuda":
            resolved.append("cuda:0")
        else:
            resolved.append(normalized)
    return resolved


def cleanup_torch_memory(device: str | None = None) -> None:
    """Release cached tensors after large inference steps."""
    if torch is None:
        gc.collect()
        return
    resolved_device = device or detect_torch_device()

    if is_cuda_device(resolved_device) and torch.cuda.is_available():
        # On CUDA inference servers we keep allocator caches warm.
        # PyTorch documents that empty_cache() does not increase memory
        # available to PyTorch itself and is mainly useful for fragmentation.
        return

    gc.collect()
    if is_mps_device(resolved_device) and torch.backends.mps.is_available():
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
