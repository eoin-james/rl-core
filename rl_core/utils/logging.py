"""Logging backends for experiment tracking.

All loggers satisfy the ``Logger`` protocol:  ``log(metrics, step)`` and ``close()``.
Compose multiple backends with ``CompositeLogger``.

Typical usage::

    logger = CompositeLogger(
        StdoutLogger(),
        CSVLogger("runs/my_run/metrics.csv"),
        WandbLogger(project="my-project", name="run-1", config=cfg_dict),
    )
    logger.log({"loss": 0.4, "reward": 12.3}, step=100)
    logger.close()
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Logger(Protocol):
    """Minimal protocol all loggers must satisfy."""

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Emit *metrics* at *step*."""
        ...

    def close(self) -> None:
        """Flush and release any resources."""
        ...


class StdoutLogger:
    """Prints metrics to stdout."""

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Print *metrics* to stdout."""
        parts = "  ".join(f"{k}={v:.4g}" for k, v in metrics.items())
        print(f"step={step}  {parts}", file=sys.stdout, flush=True)

    def close(self) -> None:
        """No-op; stdout needs no cleanup."""
        pass


class CSVLogger:
    """Appends metrics to a CSV file. Columns are inferred from the first call.

    Opens in append mode so reruns extend rather than overwrite an existing log.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", newline="")
        self._writer: csv.DictWriter | None = None
        self._needs_header = self._path.stat().st_size == 0

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Append *metrics* as a CSV row."""
        row = {"step": step, **metrics}
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            if self._needs_header:
                self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        """Flush and close the underlying CSV file."""
        self._file.close()


class WandbLogger:
    """Logs metrics to Weights & Biases.

    Requires ``wandb`` to be installed (``pip install wandb``).
    Supports run resumption: if ``run_id`` is provided and a prior run exists,
    wandb will resume it.
    """

    def __init__(
        self,
        project: str,
        name: str | None = None,
        run_id: str | None = None,
        config: dict[str, Any] | None = None,
        **wandb_kwargs: Any,
    ) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("wandb is required for WandbLogger. Install with: pip install wandb") from exc

        self._wandb = wandb
        wandb.init(
            project=project,
            name=name,
            id=run_id,
            resume="allow" if run_id else None,
            config=config,
            **wandb_kwargs,
        )

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Forward *metrics* to the active wandb run."""
        self._wandb.log(metrics, step=step)

    def close(self) -> None:
        """Finish the wandb run."""
        self._wandb.finish()


class CompositeLogger:
    """Fans out ``log`` and ``close`` calls to multiple backends."""

    def __init__(self, *loggers: Logger) -> None:
        self._loggers = list(loggers)

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Forward *metrics* to every backend."""
        for logger in self._loggers:
            logger.log(metrics, step)

    def close(self) -> None:
        """Close every backend."""
        for logger in self._loggers:
            logger.close()


def make_run_id(config: Any, prefix: str = "") -> str:
    """Derive a deterministic run ID from a dataclass config.

    The ID is a short SHA-1 digest of the config's field values, prefixed
    with ``prefix`` if given.  Useful for resumable runs: same config → same ID.
    """
    data = {f.name: getattr(config, f.name) for f in fields(config)}  # type: ignore[arg-type]
    digest = hashlib.sha1(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return f"{prefix}__{digest}" if prefix else digest
