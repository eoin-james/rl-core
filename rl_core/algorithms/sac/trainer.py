"""SAC training logic with automatic entropy tuning.

Reference: Haarnoja et al., "Soft Actor-Critic Algorithms and Applications" (2018).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from rl_core.algorithms.sac.network import GaussianPolicy, TwinQNetwork


@dataclass(frozen=True)
class SACConfig:
    obs_dim: int
    action_dim: int
    hidden_dims: tuple[int, ...] = (256, 256)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005  # soft target-network update coefficient
    batch_size: int = 256
    buffer_capacity: int = 1_000_000
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    target_entropy: float | None = None  # defaults to -action_dim at construction


class SACTrainer:
    """Trains a Soft Actor-Critic agent with automatic entropy tuning.

    Assumes continuous, bounded action spaces (tanh-squashed outputs).
    Decoupled from the outer training loop: call :meth:`select_action` for
    environment interaction and :meth:`train_step` with a sampled batch for
    one full actor + critic + alpha update.

    Example::

        trainer = SACTrainer(cfg, device)
        # ... warm-up buffer ...
        batch = buffer.sample(cfg.batch_size, device)
        metrics = trainer.train_step(batch)
    """

    def __init__(self, config: SACConfig, device: torch.device) -> None:
        self.config = config
        self.device = device

        self.actor = GaussianPolicy(
            config.obs_dim,
            config.action_dim,
            config.hidden_dims,
            config.log_std_min,
            config.log_std_max,
        ).to(device)

        self.critic = TwinQNetwork(config.obs_dim, config.action_dim, config.hidden_dims).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.eval()

        self.target_entropy: float = (
            config.target_entropy
            if config.target_entropy is not None
            else -float(config.action_dim)
        )
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Sample (or deterministically select) an action from the current policy."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic_action(obs_t)
            else:
                action, _ = self.actor.sample(obs_t)
        return action.squeeze(0).cpu().numpy()

    def train_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        """One joint update for critic, actor, and temperature α.

        Args:
            batch: Dict with keys ``obs``, ``action``, ``reward``, ``next_obs``,
                   ``done``.  All tensors must be ``float32`` on ``self.device``.

        Returns:
            Dict of scalar training metrics.
        """
        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        # ── Critic ─────────────────────────────────────────────────────────────
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            next_q = self.critic_target.min_q(next_obs, next_action)
            target_q = reward + (1.0 - done) * self.config.gamma * (
                next_q - self.alpha.detach() * next_log_prob
            )

        q1, q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ── Actor ──────────────────────────────────────────────────────────────
        new_action, log_prob = self.actor.sample(obs)
        q_val = self.critic.min_q(obs, new_action)
        actor_loss = (self.alpha.detach() * log_prob - q_val).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ── Temperature α ──────────────────────────────────────────────────────
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._soft_update_target()

        return {
            "loss/critic": critic_loss.item(),
            "loss/actor": actor_loss.item(),
            "loss/alpha": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "entropy": -log_prob.mean().item(),
        }

    def _soft_update_target(self) -> None:
        tau = self.config.tau
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters(), strict=True):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

    def state_dicts(self) -> dict[str, dict]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }

    def load_state_dicts(self, state_dicts: dict[str, dict]) -> None:
        self.actor.load_state_dict(state_dicts["actor"])
        self.critic.load_state_dict(state_dicts["critic"])
        self.critic_target.load_state_dict(state_dicts["critic_target"])
        self.actor_optimizer.load_state_dict(state_dicts["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dicts["critic_optimizer"])
        self.log_alpha.data.copy_(state_dicts["log_alpha"].to(self.device))
