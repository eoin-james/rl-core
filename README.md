# rl-core

Shared infrastructure for reinforcement learning research. This library provides reusable components across RL projects, reducing duplication and standardising experiment tooling.

## Overview

`rl-core` is a supporting library intended to be consumed by other RL repos (e.g. `rl-evo-lab`, `lang-conditioned-control`). Rather than reimplementing training loops, logging, and environment utilities in each project, this library provides a single, well-tested home for that shared logic.

## Structure

```
rl_core/
├── envs/        # Environment wrappers and utilities
├── training/    # Training loop scaffolding
├── buffers/     # Replay buffers and data structures
├── logging/     # Metrics, logging, and experiment tracking
└── utils/       # General-purpose helpers
```

> Structure evolves as components are extracted from downstream projects.

## Installation

This package is managed with [Poetry](https://python-poetry.org/).

```bash
# From the repo root
poetry install

# To use in another project (once published or via path dependency)
pip install rl-core
# or in pyproject.toml:
# rl-core = { path = "../rl-core", develop = true }
```

## Tooling

| Tool | Purpose |
|------|---------|
| [ruff](https://docs.astral.sh/ruff/) | Linting and formatting |
| [ty](https://github.com/astral-sh/ty) | Static type checking |

```bash
# Lint and format
poetry run ruff check .
poetry run ruff format .

# Type check
poetry run ty check
```

## License

MIT
