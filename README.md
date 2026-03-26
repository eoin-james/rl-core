# rl-core

Shared infrastructure for reinforcement learning research. Provides reusable components — replay buffers, logging, config, checkpointing, and algorithm implementations — consumed by sibling projects to eliminate duplication and enforce consistent tooling.

## Modules

| Module | Contents |
|--------|----------|
| `rl_core.buffers` | `ReplayBuffer` — generic field-spec circular buffer |
| `rl_core.utils` | `seed_everything`, `get_device`, `load_config`, `Logger` backends, `save/load_checkpoint` |
| `rl_core.nn` | `build_mlp`, `FlatParamsMixin` |
| `rl_core.algorithms.dqn` | `QNetwork`, `DQNConfig`, `DQNTrainer` |
| `rl_core.algorithms.sac` | `GaussianPolicy`, `TwinQNetwork`, `SACConfig`, `SACTrainer` |

## Installation

### In a sibling repo (local development)

Both `pyproject.toml` formats are supported.

**PEP 621 (`[project]` table):**
```toml
[project]
dependencies = [
    "rl-core @ file:../rl-core",
]
```

**Poetry group syntax:**
```toml
[tool.poetry.dependencies]
rl-core = { path = "../rl-core", develop = true }
```

Then:
```bash
poetry install  # or: pip install -e ../rl-core
```

### From GitHub (CI / machines without a local clone)

```toml
[project]
dependencies = [
    "rl-core @ git+https://github.com/eoin-james/rl-core.git",
]
```

Pin to a tag for reproducible installs:
```toml
"rl-core @ git+https://github.com/eoin-james/rl-core.git@v0.1.0"
```

## Usage

### Replay buffer

```python
from rl_core.buffers import ReplayBuffer
from rl_core.utils import get_device

device = get_device()  # auto: cuda > mps > cpu

buffer = ReplayBuffer(
    capacity=100_000,
    fields={"obs": (4,), "action": (1,), "reward": (), "next_obs": (4,), "done": ()},
)
buffer.push(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)

if buffer.ready(min_size=1000):
    batch = buffer.sample(256, device=device)
    # batch["obs"] → float32 Tensor, shape (256, 4)
```

### Logging

```python
from rl_core.utils.logging import CompositeLogger, CSVLogger, StdoutLogger, WandbLogger

logger = CompositeLogger(
    StdoutLogger(),
    CSVLogger("runs/my_run/metrics.csv"),
    WandbLogger(project="my-project", name="run-1", config=cfg_dict),
)
logger.log({"loss/q": 0.04, "reward": 120.3}, step=1000)
logger.close()
```

### Config

```python
from dataclasses import dataclass
from rl_core.utils import load_config

@dataclass(frozen=True)
class TrainConfig:
    env_id: str = "CartPole-v1"
    lr: float = 3e-4
    hidden_dims: tuple[int, ...] = (256, 256)

cfg = load_config("config.yaml", TrainConfig)
```

### DQN

```python
from rl_core.algorithms.dqn import DQNConfig, DQNTrainer

cfg = DQNConfig(obs_dim=4, action_dim=2)
trainer = DQNTrainer(cfg, device=device)

action = trainer.select_action(obs, epsilon=0.1)
metrics = trainer.train_step(batch)   # returns {"loss/q": ..., "q/mean": ...}
```

### SAC

```python
from rl_core.algorithms.sac import SACConfig, SACTrainer

cfg = SACConfig(obs_dim=8, action_dim=2)
trainer = SACTrainer(cfg, device=device)

action = trainer.select_action(obs)                    # stochastic
action = trainer.select_action(obs, deterministic=True)  # eval
metrics = trainer.train_step(batch)  # returns loss/critic, loss/actor, alpha, entropy
```

### Checkpointing

```python
from rl_core.utils import save_checkpoint, load_checkpoint, Checkpoint

save_checkpoint(
    Checkpoint(step=1000, state_dicts=trainer.state_dicts(), metrics={"reward": 200.0}),
    path="checkpoints/step_1000.pt",
)

ckpt = load_checkpoint("checkpoints/step_1000.pt", map_location=device)
trainer.load_state_dicts(ckpt.state_dicts)
```

## Tooling

```bash
poetry run ruff check .       # lint
poetry run ruff format .      # format
poetry run ty check           # type check
poetry run pytest tests/ -v   # tests
```

## License

MIT
