import warnings
import torch
from typing import Any

def normalize_dtype_for_device(dtype: Any, device: Any) -> torch.dtype:
    """
    Ensure dtype+device is a supported combination for distributed collectives and CPU ops.
    If dtype is bfloat16 on CPU, fall back to float32 and warn the user.
    """
    # Accept device as either a torch.device or a string like "cpu"
    if isinstance(device, str):
        device = torch.device(device)

    # Guard: if device is CPU and dtype is bfloat16, fall back to float32
    if getattr(device, "type", None) == "cpu" and dtype == torch.bfloat16:
        warnings.warn(
            "bfloat16 on CPU is not reliably supported for distributed collectives on "
            "some PyTorch builds. Falling back to float32 to avoid RuntimeError: Invalid scalar type.",
            UserWarning,
        )
        return torch.float32
    return dtype