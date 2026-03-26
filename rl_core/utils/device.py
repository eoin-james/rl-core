"""Device selection utilities."""

from __future__ import annotations

import torch


def get_device(prefer: str = "auto") -> torch.device:
    """Return a torch device.

    Args:
        prefer: ``"auto"`` selects in priority order: CUDA > MPS > CPU.
                Pass ``"cuda"``, ``"mps"``, or ``"cpu"`` to force a specific device.
    """
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(prefer)
