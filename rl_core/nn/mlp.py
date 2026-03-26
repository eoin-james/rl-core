"""MLP building block shared across algorithm implementations."""

from __future__ import annotations

from collections.abc import Sequence

import torch.nn as nn


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int] = (256, 256),
    activation: type[nn.Module] = nn.ReLU,
    output_activation: type[nn.Module] | None = None,
    layer_norm: bool = False,
) -> nn.Sequential:
    """Construct a multi-layer perceptron as an ``nn.Sequential``.

    Args:
        input_dim: Input feature dimension.
        output_dim: Output feature dimension.
        hidden_dims: Width of each hidden layer, in order.
        activation: Activation applied after each hidden layer.
        output_activation: Optional activation applied after the final linear layer.
        layer_norm: If ``True``, insert ``LayerNorm`` after each hidden activation.
    """
    layers: list[nn.Module] = []
    in_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(activation())
        in_dim = h
    layers.append(nn.Linear(in_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)
