"""Experiment lifecycle management for RL research.

Provides :class:`RunManager` (idempotency, status, checkpointing) and
:class:`NamespacedLogger` (automatic ``algo/`` / ``research/`` metric routing).

Quick-start::

    from rl_core.experiments import RunManager, NamespacedLogger
    from rl_core.utils.logging import CompositeLogger, CSVLogger, WandbLogger
    from rl_core.utils.config import config_to_dict

    logger = NamespacedLogger(
        CompositeLogger(
            CSVLogger("runs/metrics.csv"),
            WandbLogger(project="my-project", config=config_to_dict(cfg)),
        ),
        algo_keys={"learner_loss", "q_mean"},
        research_keys={"idn_loss", "effective_beta"},
    )
    manager = RunManager(cfg, results_dir="runs/", logger=logger)

    if not manager.is_done():
        with manager.run() as run:
            start_step, ckpt = run.resume()
            if ckpt:
                trainer.load_state_dicts(ckpt.state_dicts)
            for step in range(start_step, cfg.total_steps):
                metrics = trainer.train_step(batch)
                run.log(step=step, metrics=metrics)
                if step % 10_000 == 0:
                    run.checkpoint(step=step, state_dicts=trainer.state_dicts())
"""

from rl_core.experiments.metrics import NamespacedLogger
from rl_core.experiments.run_manager import ExperimentRun, RunManager

__all__ = ["ExperimentRun", "NamespacedLogger", "RunManager"]
