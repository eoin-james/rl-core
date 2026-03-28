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

    ``rng_state`` captures numpy + torch RNG states at save time so a resumed
    run is numerically identical to an uninterrupted one.  Populated
    automatically by :class:`~rl_core.experiments.run_manager.ExperimentRun`.
    """

    step: int
    state_dicts: dict[str, dict[str, Any]]
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    rng_state: dict[str, Any] | None = None


def capture_rng_state() -> dict[str, Any]:
    """Snapshot current numpy + torch (CPU + CUDA) RNG states."""
    import numpy as np

    state: dict[str, Any] = {
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore numpy + torch RNG states from a snapshot."""
    import numpy as np

    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


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
            "rng_state": checkpoint.rng_state,
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
        rng_state=data.get("rng_state"),
    )
