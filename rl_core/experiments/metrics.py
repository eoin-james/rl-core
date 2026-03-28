"""Namespace-routing logger wrapper.

Wraps any :class:`~rl_core.utils.logging.Logger` and automatically prefixes
metric keys so that downstream training code never has to hardcode namespaces.

Example::

    inner = CompositeLogger(
        CSVLogger("runs/my_run/metrics.csv"),
        WandbLogger(project="my-project", run_id=run_id, config=cfg_dict),
    )
    logger = NamespacedLogger(
        inner,
        algo_keys={"learner_loss", "q_mean", "q_target_mean"},
        research_keys={"idn_loss", "effective_beta", "buffer_diversity"},
    )

    logger.log({"learner_loss": 0.3, "idn_loss": 0.1, "episode": 42}, step=100)
    # inner receives:
    #   {"algo/learner_loss": 0.3, "research/idn_loss": 0.1, "episode": 42}
"""

from __future__ import annotations

from rl_core.utils.logging import Logger


class NamespacedLogger:
    """Routes metric keys to ``algo/`` or ``research/`` namespaces automatically.

    Keys in ``algo_keys`` are prefixed with ``"algo/"``.
    Keys in ``research_keys`` are prefixed with ``"research/"``.
    All other keys are passed through unchanged (bare namespace).

    The two sets must be disjoint.  Unknown keys (not in either set) are
    logged without a prefix — useful for step counters, episode numbers, etc.

    Parameters
    ----------
    logger:
        Underlying logger that receives the re-keyed metrics dict.
    algo_keys:
        Metric keys that represent base-algorithm health (losses, Q-values,
        gradient norms, α).  These answer "is the algorithm working?".
    research_keys:
        Metric keys that represent the research signal (intrinsic reward,
        exploration metrics, task-specific outcomes).  These answer "is the
        research contribution working?".
    """

    def __init__(
        self,
        logger: Logger,
        algo_keys: set[str] | None = None,
        research_keys: set[str] | None = None,
    ) -> None:
        self._logger = logger
        self._algo = set(algo_keys or [])
        self._research = set(research_keys or [])
        overlap = self._algo & self._research
        if overlap:
            raise ValueError(f"Keys appear in both algo_keys and research_keys: {overlap}")

    def log(self, metrics: dict[str, float], step: int) -> None:
        routed: dict[str, float] = {}
        for k, v in metrics.items():
            if k in self._algo:
                routed[f"algo/{k}"] = v
            elif k in self._research:
                routed[f"research/{k}"] = v
            else:
                routed[k] = v
        self._logger.log(routed, step)

    def close(self) -> None:
        self._logger.close()
