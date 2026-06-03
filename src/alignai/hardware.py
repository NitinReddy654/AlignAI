"""Hardware detection, dtype selection, and resource cleanup."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Optional

from alignai.logging_utils import setup_logging

logger = setup_logging(__name__)


@dataclass
class HardwareInfo:
    """Detected hardware capabilities."""

    device: str
    device_type: str  # cuda | mps | cpu
    dtype: str  # bfloat16 | float16 | float32
    bf16_supported: bool
    cuda_available: bool
    mps_available: bool


def detect_hardware() -> HardwareInfo:
    """Detect available compute device and preferred dtype."""
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        bf16_supported = cuda_available and torch.cuda.is_bf16_supported()

        if cuda_available:
            device = "cuda"
            device_type = "cuda"
            dtype = "bfloat16" if bf16_supported else "float16"
        elif mps_available:
            device = "mps"
            device_type = "mps"
            dtype = "bfloat16"
        else:
            device = "cpu"
            device_type = "cpu"
            dtype = "float32"

        info = HardwareInfo(
            device=device,
            device_type=device_type,
            dtype=dtype,
            bf16_supported=bf16_supported,
            cuda_available=cuda_available,
            mps_available=mps_available,
        )
        logger.info(
            "Hardware detected: device=%s dtype=%s cuda=%s mps=%s",
            info.device,
            info.dtype,
            info.cuda_available,
            info.mps_available,
        )
        return info
    except ImportError:
        logger.warning("PyTorch not available; falling back to CPU/float32.")
        return HardwareInfo(
            device="cpu",
            device_type="cpu",
            dtype="float32",
            bf16_supported=False,
            cuda_available=False,
            mps_available=False,
        )


def get_torch_dtype(dtype_name: str) -> Any:
    """Convert dtype string to torch dtype."""
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_name, torch.float32)


def cleanup_resources(model: Optional[Any] = None) -> None:
    """Release GPU memory and run garbage collection."""
    try:
        import torch

        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Resources cleaned up.")
    except ImportError:
        gc.collect()
