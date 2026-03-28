"""DQN training logic with soft target-network updates."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from rl_core.algorithms.dqn.network import QNetwork


@dataclass(frozen=True)
class DQNConfig:
    """Frozen configuration for DQNTrainer."""

    obs_dim: int
    action_dim: int
    hidden_dims: tuple[int, ...] = (256, 256)
    lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005  # soft target-network update coefficient
    batch_size: int = 256
    buffer_capacity: int = 100_000
    gradient_clip: float = 10.0


class DQNTrainer:
    """Trains a Q-network with experience replay and a target network.

    Decoupled from the outer training loop: call :meth:`select_action` for
    ε-greedy exploration and :meth:`train_step` with a sampled batch to
    perform one gradient update.

    Example::

        trainer = DQNTrainer(cfg, device)
        # ... fill buffer ...
        batch = buffer.sample(cfg.batch_size, device)
        metrics = trainer.train_step(batch)
    """

    def __init__(self, config: DQNConfig, device: torch.device) -> None:
        self.config = config
        self.device = device

        self.q_network = QNetwork(config.obs_dim, config.action_dim, config.hidden_dims).to(device)
        self.target_network = copy.deepcopy(self.q_network)
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=config.lr)

    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        """ε-greedy action selection.

        Returns a random action with probability ``epsilon``, otherwise the
        greedy action from the Q-network.
        """
        if np.random.random() < epsilon:
            return int(np.random.randint(self.config.action_dim))
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.q_network(obs_t).argmax(dim=-1).item())

    def train_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        """One gradient update step.

        Args:
            batch: Dict with keys ``obs``, ``action``, ``reward``, ``next_obs``,
                   ``done``.  All tensors must be ``float32`` on ``self.device``.
                   ``action`` will be cast to ``long`` internally.

        Returns:
            Dict of scalar training metrics.
        """
        obs = batch["obs"]
        action = batch["action"].long()
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        with torch.no_grad():
            next_q = self.target_network(next_obs).max(dim=-1, keepdim=True).values
            target_q = reward + (1.0 - done) * self.config.gamma * next_q

        current_q = self.q_network(obs).gather(1, action)
        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        self._soft_update_target()

        return {
            "loss/q": loss.item(),
            "q/mean": current_q.mean().item(),
            "q/target_mean": target_q.mean().item(),
        }

    def _soft_update_target(self) -> None:
        tau = self.config.tau
        q_params = zip(self.q_network.parameters(), self.target_network.parameters(), strict=True)
        for p, tp in q_params:
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

    def state_dicts(self) -> dict[str, dict]:
        """Return state dicts for the Q-network and target network."""
        return {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dicts(self, state_dicts: dict[str, dict]) -> None:
        """Restore Q-network and target network from *state_dicts*."""
        self.q_network.load_state_dict(state_dicts["q_network"])
        self.target_network.load_state_dict(state_dicts["target_network"])
        self.optimizer.load_state_dict(state_dicts["optimizer"])
