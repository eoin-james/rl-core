"""Configuration utilities: load YAML files into typed frozen dataclasses.

Example::

    @dataclass(frozen=True)
    class MyConfig:
        lr: float = 3e-4
        gamma: float = 0.99
        hidden_dims: tuple[int, ...] = (256, 256)

    cfg = load_config("config.yaml", MyConfig)
    d = config_to_dict(cfg)  # plain dict for logging
"""

from __future__ import annotations

import dataclasses
import typing
from pathlib import Path
from typing import Any

import yaml


def _coerce(value: Any, hint: Any) -> Any:
    """Best-effort type coercion for primitive types and nested dataclasses."""
    # Unwrap Optional[X] / X | None → X
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)

    if origin is typing.Union or str(origin) == "typing.Union":
        # Take the first non-None arg
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            hint = non_none[0]
            origin = typing.get_origin(hint)
            args = typing.get_args(hint)

    if dataclasses.is_dataclass(hint) and isinstance(value, dict):
        return _from_dict(hint, value)

    # YAML lists → tuples (frozen dataclasses prefer tuples for sequences)
    if origin is tuple and isinstance(value, list):
        return tuple(value)

    return value


def _from_dict[T](cls: type[T], data: dict[str, Any]) -> T:
    """Recursively populate a dataclass from a plain dict."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")

    try:
        hints = typing.get_type_hints(cls)
    except Exception:
        hints = {f.name: f.type for f in dataclasses.fields(cls)}  # type: ignore[arg-type]

    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):  # type: ignore[arg-type]
        if f.name not in data:
            continue  # use field default
        kwargs[f.name] = _coerce(data[f.name], hints.get(f.name, f.type))

    return cls(**kwargs)  # type: ignore[call-arg]


def load_config[T](path: Path | str, cls: type[T]) -> T:
    """Load a YAML file and parse it into a dataclass of type ``cls``."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping at {path}, got {type(raw)}")
    return _from_dict(cls, raw)


def config_to_dict(cfg: Any) -> dict[str, Any]:
    """Convert a dataclass config to a plain dict (for logging or serialisation)."""
    return dataclasses.asdict(cfg)
