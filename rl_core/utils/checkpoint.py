"""Checkpoint save and load utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class Checkpoint:
    """Container for a training checkpoint.

    ``state_dicts`` maps component names (e.g. ``"actor"``, ``"optimizer"``)
    to their ``state_dict()``.  ``metrics`` and ``metadata`` are free-form
    dicts for storing eval results or experiment context.
    """

    step: int
    state_dicts: dict[str, dict[str, Any]]
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def save_checkpoint(checkpoint: Checkpoint, path: Path | str) -> None:
    """Serialise a checkpoint to disk with ``torch.save``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": checkpoint.step,
            "state_dicts": checkpoint.state_dicts,
            "metrics": checkpoint.metrics,
            "metadata": checkpoint.metadata,
        },
        path,
    )


def load_checkpoint(path: Path | str, map_location: str | torch.device = "cpu") -> Checkpoint:
    """Load a checkpoint from disk."""
    data = torch.load(Path(path), map_location=map_location, weights_only=False)
    return Checkpoint(
        step=data["step"],
        state_dicts=data["state_dicts"],
        metrics=data.get("metrics", {}),
        metadata=data.get("metadata", {}),
    )
