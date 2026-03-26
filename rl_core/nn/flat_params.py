"""Mixin providing flat parameter access for evolution-strategy compatible networks."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from numpy import ndarray


class FlatParamsMixin(nn.Module):
    """Mixin that adds flat (1D numpy array) parameter get/set to any ``nn.Module``.

    Used by ES-based methods that need to treat network weights as a flat
    parameter vector for perturbation or weight-sharing.
    """

    def get_flat_params(self) -> ndarray:
        """Return all parameters concatenated into a single 1-D numpy array."""
        return np.concatenate([p.data.cpu().numpy().ravel() for p in self.parameters()])

    def set_flat_params(self, params: ndarray) -> None:
        """Load all parameters from a 1-D numpy array in declaration order."""
        offset = 0
        for p in self.parameters():
            size = p.numel()
            p.data.copy_(
                torch.as_tensor(
                    params[offset : offset + size].reshape(p.shape),
                    dtype=p.dtype,
                    device=p.device,
                )
            )
            offset += size

    @property
    def num_params(self) -> int:
        """Total number of parameters (trainable + frozen)."""
        return sum(p.numel() for p in self.parameters())
