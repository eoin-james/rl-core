# rl-core

Shared infrastructure for reinforcement learning research. Provides reusable components — replay buffers, logging, config, checkpointing, algorithm implementations, and experiment lifecycle management — consumed by sibling projects to eliminate duplication and enforce consistent tooling.

**Consuming repos:** `rl-evo-lab`, `lang-conditioned-control`

---

## Modules

| Module | Contents |
|--------|----------|
| `rl_core.buffers` | `ReplayBuffer` — generic field-spec circular buffer |
| `rl_core.utils` | `seed_everything`, `get_device`, `load_config`, `Logger` backends, `save/load_checkpoint`, `capture/restore_rng_state` |
| `rl_core.nn` | `build_mlp`, `FlatParamsMixin` |
| `rl_core.algorithms.dqn` | `QNetwork`, `DQNConfig`, `DQNTrainer` |
| `rl_core.algorithms.sac` | `GaussianPolicy`, `TwinQNetwork`, `SACConfig`, `SACTrainer` |
| `rl_core.experiments` | `RunManager`, `ExperimentRun`, `NamespacedLogger` |

---

## Installation

### Recommended: GitHub dependency (pinned to a release tag)

```toml
# pyproject.toml in consuming repo
[project]
dependencies = [
    "rl-core @ git+https://github.com/graylayer-labs/rl-core.git@v1.0.0",
]
```

Pin to a specific tag so your repo is insulated from breaking changes on `main`. Upgrade deliberately when ready.

```bash
poetry install
```

### Local development (side-by-side checkout)

If you have both repos cloned under the same parent directory:

```toml
[project]
dependencies = [
    "rl-core @ file:../rl-core",
]
```

```bash
poetry install
```

---

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
metrics = trainer.train_step(batch)  # {"loss/q": ..., "q/mean": ...}
```

### SAC

```python
from rl_core.algorithms.sac import SACConfig, SACTrainer

cfg = SACConfig(obs_dim=8, action_dim=2)
trainer = SACTrainer(cfg, device=device)

action = trainer.select_action(obs)                      # stochastic
action = trainer.select_action(obs, deterministic=True)  # eval
metrics = trainer.train_step(batch)  # {"loss/critic": ..., "loss/actor": ..., "alpha": ..., "entropy": ...}
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

### Experiment management

`RunManager` handles the boilerplate that every training script needs: deterministic run IDs, status tracking, idempotency, and checkpoint/resume with full RNG state.

`NamespacedLogger` separates base-algorithm health metrics (`algo/`) from research-specific signal (`research/`), so you can always tell whether the underlying algorithm is stable independently of whether your research contribution is working.

```python
from pathlib import Path
from rl_core.experiments import RunManager, NamespacedLogger
from rl_core.utils.logging import CompositeLogger, CSVLogger, WandbLogger
from rl_core.utils.config import config_to_dict
from rl_core.utils import seed_everything

cfg = MyConfig(env_id="CartPole-v1", seed=42, total_steps=500_000)
seed_everything(cfg.seed)

logger = NamespacedLogger(
    CompositeLogger(
        CSVLogger(f"runs/{cfg.env_id}/metrics.csv"),
        WandbLogger(project="my-project", config=config_to_dict(cfg)),
    ),
    algo_keys={"loss/q", "q/mean"},           # logged as algo/loss/q, algo/q/mean
    research_keys={"idn_loss", "eff_beta"},   # logged as research/idn_loss, ...
)

manager = RunManager(
    config=cfg,
    results_dir=Path("runs/"),
    logger=logger,
    run_id_prefix=f"{cfg.env_id}__seed{cfg.seed}",
)

if manager.is_done():
    print(f"Run {manager.run_id} already complete, skipping.")
    return

with manager.run() as run:
    start_step, ckpt = run.resume()   # finds latest checkpoint, restores RNG
    if ckpt:
        trainer.load_state_dicts(ckpt.state_dicts)

    for step in range(start_step, cfg.total_steps):
        batch = buffer.sample(cfg.batch_size, device)
        metrics = trainer.train_step(batch)
        run.log(step=step, metrics=metrics)

        if step % 50_000 == 0:
            run.checkpoint(step=step, state_dicts=trainer.state_dicts())
```

**What `RunManager` does automatically:**
- Derives a SHA-1 run ID from the config — same config always maps to the same directory
- Refuses to re-run a completed experiment (`is_done()`) unless `force=True`
- Writes `status.json` (`running` → `completed` / `interrupted` / `failed`) so you can check state without opening TensorBoard
- Saves a frozen `config.json` on first entry for human reference
- Captures numpy + torch RNG state in every checkpoint so resumed runs are numerically identical

**Run directory layout:**
```
runs/
  CartPole-v1__seed42__59274889/
    config.json          ← frozen config at run start
    status.json          ← current state: running | completed | interrupted | failed
    checkpoints/
      ckpt_00000000.pt
      ckpt_00050000.pt
      ckpt_00100000.pt
```

### Checkpointing (standalone)

For use without `RunManager`:

```python
from rl_core.utils import save_checkpoint, load_checkpoint, Checkpoint, capture_rng_state

save_checkpoint(
    Checkpoint(
        step=1000,
        state_dicts=trainer.state_dicts(),
        metrics={"reward": 200.0},
        rng_state=capture_rng_state(),
    ),
    path="checkpoints/step_1000.pt",
)

ckpt = load_checkpoint("checkpoints/step_1000.pt", map_location=device)
trainer.load_state_dicts(ckpt.state_dicts)
```

---

## Versioning

This repo uses [semantic versioning](https://semver.org) with git tags.

| Bump | When |
|------|------|
| `patch` (0.1.**x**) | Bug fix, no API change |
| `minor` (0.**x**.0) | New feature, backwards compatible |
| `major` (**x**.0.0) | Breaking change — renamed, removed, or changed signature |

**Consuming repos should always pin to a tag.** Breaking changes will never land silently — they require a major version bump and are documented in [CHANGELOG.md](CHANGELOG.md).

---

## For consuming repos

### How to request a change

1. Open a [GitHub Issue](https://github.com/graylayer-labs/rl-core/issues) using one of the templates:
   - **Change request** — you need something new, or existing behaviour is blocking you
   - **Bug report** — something is behaving incorrectly

2. The maintainer will triage and reply with one of:
   - **Will do** — implementation is planned; issue stays open until the release is tagged
   - **Won't do** — not appropriate for a shared library; issue closed with a reason
   - **Needs discussion** — scope or API design needs agreement before work starts

3. Once a release is tagged, the issue is closed and the version appears in [CHANGELOG.md](CHANGELOG.md).

### How to upgrade your pin

When a new release is tagged:

```toml
# pyproject.toml — bump the tag
"rl-core @ git+https://github.com/graylayer-labs/rl-core.git@v0.3.0"
```

```bash
poetry update rl-core
```

Check [CHANGELOG.md](CHANGELOG.md) for what changed. If the release is a **major version bump**, read the migration notes before upgrading — something you import will have changed.

### What to expect from breaking changes

Breaking changes will never land silently. A major version bump means:
- The CHANGELOG has a migration note describing exactly what changed and how to update
- The PR was labelled `breaking` and lists affected repos
- You have time to upgrade on your own schedule — your pinned version keeps working until you move it

---

## Tooling

```bash
poetry run ruff check .       # lint
poetry run ruff format .      # format
poetry run ty check           # type check
poetry run pytest tests/ -v   # tests
```

---

## License

MIT
