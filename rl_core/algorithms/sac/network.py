"""SAC actor and critic networks."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from rl_core.nn.mlp import build_mlp


class GaussianPolicy(nn.Module):
    """Diagonal Gaussian policy with tanh squashing for bounded action spaces.

    Architecture: shared trunk ending with an activation, then two separate
    linear heads for mean and log-std.  This keeps the two heads independent
    while sharing the learned representation.

    Implements the reparameterisation trick and includes the tanh Jacobian
    correction in the returned log-probability.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Trunk outputs hidden_dims[-1] features with activation applied.
        self.trunk = build_mlp(
            obs_dim,
            hidden_dims[-1],
            hidden_dims[:-1],
            output_activation=nn.ReLU,
        )
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

    def _dist(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Sample an action via the reparameterisation trick.

        Returns:
            action: Tanh-squashed action in (-1, 1).  Shape ``(batch, action_dim)``.
            log_prob: Log-probability with tanh correction.  Shape ``(batch, 1)``.
        """
        mean, log_std = self._dist(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        log_prob = normal.log_prob(x).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic_action(self, obs: Tensor) -> Tensor:
        """Return the squashed mean action (for evaluation, no sampling)."""
        mean, _ = self._dist(obs)
        return torch.tanh(mean)


class TwinQNetwork(nn.Module):
    """Two independent Q-networks to reduce overestimation bias (SAC, TD3).

    Takes ``(obs, action)`` pairs and returns a pair of scalar Q-values.
    Training minimises both; inference uses the minimum.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.q1 = build_mlp(obs_dim + action_dim, 1, hidden_dims)
        self.q2 = build_mlp(obs_dim + action_dim, 1, hidden_dims)

    def forward(self, obs: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(Q1, Q2)`` — each of shape ``(batch, 1)``."""
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def min_q(self, obs: Tensor, action: Tensor) -> Tensor:
        """Element-wise minimum of Q1 and Q2."""
        q1, q2 = self(obs, action)
        return torch.min(q1, q2)
