"""Neural network building blocks: MLP factory and flat-parameter mixin."""

from rl_core.nn.flat_params import FlatParamsMixin
from rl_core.nn.mlp import build_mlp

__all__ = ["FlatParamsMixin", "build_mlp"]
