"""Generic circular experience replay buffer."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


class ReplayBuffer:
    """Capacity-bounded circular buffer for arbitrary named transition fields.

    Fields are declared at construction time as a mapping from name to shape.
    A scalar field uses shape ``()``; a vector field uses ``(n,)`` etc.

    Example::

        buffer = ReplayBuffer(
            capacity=100_000,
            fields={
                "obs": (4,),
                "action": (1,),
                "reward": (),
                "next_obs": (4,),
                "done": (),
            },
        )
        buffer.push(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
        batch = buffer.sample(256, device=device)
        # batch["obs"] is a float32 Tensor of shape (256, 4) on device
    """

    def __init__(self, capacity: int, fields: dict[str, tuple[int, ...]]) -> None:
        self._capacity = capacity
        self._ptr = 0
        self._size = 0
        self._data: dict[str, np.ndarray] = {
            name: np.zeros((capacity, *shape), dtype=np.float32) for name, shape in fields.items()
        }

    def push(self, **kwargs: np.ndarray | float | int | bool) -> None:
        """Store a single transition. Keyword argument names must match declared fields."""
        for name, value in kwargs.items():
            self._data[name][self._ptr] = value
        self._ptr = (self._ptr + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, Tensor]:
        """Sample a random batch. Returns ``float32`` tensors on ``device``."""
        if batch_size > self._size:
            raise ValueError(
                f"Requested batch of {batch_size} but buffer only contains {self._size} transitions."  # noqa: E501
            )
        indices = np.random.randint(0, self._size, size=batch_size)
        return {
            name: torch.as_tensor(arr[indices], dtype=torch.float32, device=device)
            for name, arr in self._data.items()
        }

    def ready(self, min_size: int) -> bool:
        """Return ``True`` once the buffer holds at least ``min_size`` transitions."""
        return self._size >= min_size

    def __len__(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity
