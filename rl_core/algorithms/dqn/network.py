"""Q-network for DQN and its ES-based variants."""

from __future__ import annotations

from torch import Tensor

from rl_core.nn.flat_params import FlatParamsMixin
from rl_core.nn.mlp import build_mlp


class QNetwork(FlatParamsMixin):
    """MLP Q-network mapping observations to per-action Q-values.

    Inherits ``FlatParamsMixin`` so the same network class can be used as
    both the DQN learner and as an ES actor (where weights are perturbed
    as flat arrays).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim, action_dim, hidden_dims)

    def forward(self, obs: Tensor) -> Tensor:
        """Return Q-values for all actions.  Shape: ``(batch, action_dim)``."""
        return self.net(obs)
