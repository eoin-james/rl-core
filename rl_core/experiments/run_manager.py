"""Experiment lifecycle management: run IDs, directories, status, checkpoints.

:class:`RunManager` is the single entry point.  Pass it any frozen-dataclass
config and it will:

* Derive a **deterministic SHA-1 run ID** so the same config always maps to
  the same directory — run the script twice and the second call is a no-op
  (unless ``force=True``).
* Maintain a ``status.json`` file so you can check run state at a glance
  without opening TensorBoard or reading a CSV.
* Save a frozen ``config.json`` on first entry for human reference.
* Discover and load the **latest checkpoint** on resume.
* Capture and restore **full RNG state** (numpy + torch) so resumed runs are
  numerically identical to uninterrupted ones.

Typical usage::

    from pathlib import Path
    from rl_core.experiments import RunManager, NamespacedLogger
    from rl_core.utils.logging import CompositeLogger, CSVLogger, WandbLogger
    from rl_core.utils.config import config_to_dict

    cfg = MyConfig(...)
    run_id_prefix = f"{cfg.env_id}__seed{cfg.seed}"

    inner = CompositeLogger(
        CSVLogger(f"runs/{run_id_prefix}/metrics.csv"),
        WandbLogger(project="my-project", config=config_to_dict(cfg)),
    )
    logger = NamespacedLogger(
        inner,
        algo_keys={"learner_loss", "q_mean"},
        research_keys={"idn_loss", "effective_beta"},
    )
    manager = RunManager(
        config=cfg,
        results_dir=Path("runs/"),
        logger=logger,
        run_id_prefix=run_id_prefix,
    )

    if manager.is_done():
        return  # idempotent: skip already-completed runs

    with manager.run() as run:
        start_step, ckpt = run.resume()
        if ckpt:
            trainer.load_state_dicts(ckpt.state_dicts)
            # RNG already restored — numerically identical continuation

        for step in range(start_step, cfg.total_steps):
            batch = buffer.sample(cfg.batch_size, device)
            metrics = trainer.train_step(batch)
            run.log(step=step, metrics=metrics)

            if step % cfg.checkpoint_freq == 0:
                run.checkpoint(step=step, state_dicts=trainer.state_dicts())
"""

from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import fields as dataclass_fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rl_core.utils.checkpoint import (
    Checkpoint,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from rl_core.utils.config import config_to_dict
from rl_core.utils.logging import Logger, make_run_id


class ExperimentRun:
    """Active experiment handle, yielded by :meth:`RunManager.run`.

    Do not instantiate directly — use ``RunManager.run()`` as a context
    manager.
    """

    def __init__(self, manager: RunManager) -> None:
        self._manager = manager
        self._last_step: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resume(self) -> tuple[int, Checkpoint | None]:
        """Return ``(start_step, checkpoint)`` for resuming an interrupted run.

        If a checkpoint exists, its RNG state is restored automatically so
        the resumed run is numerically identical to an uninterrupted one.
        Load ``checkpoint.state_dicts`` into your trainer after this call::

            start_step, ckpt = run.resume()
            if ckpt:
                trainer.load_state_dicts(ckpt.state_dicts)

        Returns ``(0, None)`` when starting fresh.
        """
        path = self._manager.latest_checkpoint()
        if path is None:
            return 0, None
        ckpt = load_checkpoint(path)
        if ckpt.rng_state is not None:
            restore_rng_state(ckpt.rng_state)
        self._last_step = ckpt.step
        return ckpt.step, ckpt

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Log a metrics dict at ``step`` via the run's logger.

        Keys are routed to ``algo/`` or ``research/`` namespaces automatically
        if the logger is a :class:`~rl_core.experiments.metrics.NamespacedLogger`.
        """
        self._last_step = step
        self._manager._logger.log(metrics, step)
        self._manager._write_status(step=step)

    def checkpoint(
        self,
        step: int,
        state_dicts: dict[str, Any],
        metrics: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save a checkpoint at ``step``.

        RNG state is captured automatically.  The file is written to
        ``<run_dir>/checkpoints/ckpt_{step:08d}.pt``.
        """
        ckpt = Checkpoint(
            step=step,
            state_dicts=state_dicts,
            metrics=metrics or {},
            metadata=metadata or {},
            rng_state=capture_rng_state(),
        )
        path = self._manager.checkpoints_dir / f"ckpt_{step:08d}.pt"
        save_checkpoint(ckpt, path)

    # ------------------------------------------------------------------
    # Internal status helpers
    # ------------------------------------------------------------------

    def _complete(self) -> None:
        self._manager._write_status(step=self._last_step, status="completed")

    def _interrupt(self) -> None:
        self._manager._write_status(step=self._last_step, status="interrupted")

    def _fail(self) -> None:
        self._manager._write_status(step=self._last_step, status="failed")


class RunManager:
    """Manages the lifecycle of a single experiment run.

    Parameters
    ----------
    config:
        A frozen dataclass.  Used to derive a deterministic SHA-1 run ID so
        identical configs always map to the same run directory.
    results_dir:
        Parent directory; ``<results_dir>/<run_id>/`` will be created.
    logger:
        Any :class:`~rl_core.utils.logging.Logger` (e.g.
        :class:`~rl_core.experiments.metrics.NamespacedLogger` wrapping a
        :class:`~rl_core.utils.logging.CompositeLogger`).  Closed automatically
        when the ``run()`` context exits.
    run_id_prefix:
        Human-readable prefix prepended to the SHA-1 digest.
        Example: ``"CartPole-v1__seed42"`` → ``"CartPole-v1__seed42__59274889"``.
    force:
        If ``True``, allow overwriting a completed run.  Defaults to ``False``
        (raises :class:`RuntimeError` if the run is already done).
    """

    def __init__(
        self,
        config: Any,
        results_dir: Path | str,
        logger: Logger,
        run_id_prefix: str = "",
        force: bool = False,
    ) -> None:
        self._config = config
        self._results_dir = Path(results_dir)
        self._logger = logger
        self._force = force
        self._run_id = make_run_id(config, prefix=run_id_prefix)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        """Deterministic SHA-1 run identifier derived from the config."""
        return self._run_id

    @property
    def run_dir(self) -> Path:
        """Root directory for this run: ``<results_dir>/<run_id>/``."""
        return self._results_dir / self._run_id

    @property
    def checkpoints_dir(self) -> Path:
        """Directory where checkpoint files are written."""
        return self.run_dir / "checkpoints"

    # ------------------------------------------------------------------
    # Idempotency
    # ------------------------------------------------------------------

    def is_done(self) -> bool:
        """Return ``True`` if this run has previously completed successfully."""
        status_path = self.run_dir / "status.json"
        if not status_path.exists():
            return False
        data = json.loads(status_path.read_text())
        return data.get("status") == "completed"

    def latest_checkpoint(self) -> Path | None:
        """Return the path of the highest-step checkpoint, or ``None``.

        Checkpoints are sorted lexicographically; the zero-padded filenames
        (``ckpt_00001000.pt``) guarantee correct ordering.
        """
        if not self.checkpoints_dir.exists():
            return None
        checkpoints = sorted(self.checkpoints_dir.glob("ckpt_*.pt"))
        return checkpoints[-1] if checkpoints else None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextmanager
    def run(self) -> Generator[ExperimentRun, None, None]:
        """Context manager wrapping the training loop.

        On **entry**: creates run directory, writes ``config.json`` (first time
        only), sets ``status.json`` to ``"running"``.

        On **exit**: sets status to ``"completed"``, ``"interrupted"`` (Ctrl-C),
        or ``"failed"`` (unhandled exception), then closes the logger.

        Raises :class:`RuntimeError` if the run already completed and
        ``force=False``.
        """
        if self.is_done() and not self._force:
            raise RuntimeError(f"Run '{self._run_id}' is already completed. Pass force=True to RunManager to re-run.")

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Persist config once (human reference, not used by code).
        config_path = self.run_dir / "config.json"
        if not config_path.exists():
            config_path.write_text(json.dumps(_config_to_json(self._config), indent=2, default=str))

        exp_run = ExperimentRun(self)
        self._write_status(step=0, status="running")

        try:
            yield exp_run
        except KeyboardInterrupt:
            exp_run._interrupt()
            raise
        except Exception:
            exp_run._fail()
            raise
        else:
            exp_run._complete()
        finally:
            self._logger.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_status(self, step: int, status: str = "running") -> None:
        data = {
            "run_id": self._run_id,
            "status": status,
            "step": step,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        (self.run_dir / "status.json").write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config_to_json(config: Any) -> dict[str, Any]:
    """Convert a (possibly frozen) dataclass config to a JSON-serialisable dict."""
    try:
        return config_to_dict(config)
    except Exception:
        return {
            f.name: getattr(config, f.name)
            for f in dataclass_fields(config)  # type: ignore[arg-type]
        }
