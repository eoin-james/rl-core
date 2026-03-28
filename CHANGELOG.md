# Changelog

All notable changes to rl-core are documented here.

Format: `## [vX.Y.Z] - YYYY-MM-DD` followed by `### Added / Changed / Fixed / Removed`.
A `breaking` label on the GitHub PR signals a major version bump.

---

## [v1.0.0] - 2026-03-28

First stable release. Public API is now considered settled.

### Added
- Branch protection on `main`, PR-required workflow, auto-merge on CI pass
- PR template and GitHub issue templates (change request, bug report)
- `CONTRIBUTING.md` — release process, breaking change protocol, new algorithm checklist
- `CLAUDE.md` — versioning rules, consumer repo context, release process, what not to do
- `.claude/commands/release.md` and `check.md` slash commands
- `ruff.toml` and `ty.toml` — expanded ruleset (D, ANN, PTH, C4, PERF, RUF), Google docstring convention

### Fixed
- CI now installs dev dependencies (`--with dev`) so ruff, ty, pytest are available
- `test` job depends on `quality` — fails fast on lint errors

---

## [v0.2.0] - 2026-03-28

### Added
- `rl_core.experiments` module: `RunManager`, `ExperimentRun`, `NamespacedLogger`
  - `RunManager`: deterministic SHA-1 run IDs from config, `status.json` lifecycle, idempotency (`is_done()` / `force`), frozen `config.json` on first entry
  - `ExperimentRun`: `resume()` (finds latest checkpoint + restores RNG), `log()`, `checkpoint()`
  - `NamespacedLogger`: wraps any `Logger`, routes declared keys to `algo/` or `research/` prefixes automatically
- `capture_rng_state()` and `restore_rng_state()` in `rl_core.utils.checkpoint`
- `rng_state` field on `Checkpoint` dataclass (optional, defaults to `None` — fully backwards compatible)
- `pytest` added to dev dependencies; `tests/test_experiments.py` with 23 tests

### Changed
- `save_checkpoint` / `load_checkpoint` updated to persist and restore `rng_state`

---

## [v0.1.0] - 2026-03-27

Initial release.

### Added
- `rl_core.buffers.ReplayBuffer` — generic field-spec circular buffer
- `rl_core.utils`: `seed_everything`, `get_device`, `load_config`, `config_to_dict`, `save_checkpoint`, `load_checkpoint`, `Checkpoint`
- `rl_core.utils.logging`: `Logger` protocol, `StdoutLogger`, `CSVLogger`, `WandbLogger`, `CompositeLogger`, `make_run_id`
- `rl_core.nn`: `build_mlp`, `FlatParamsMixin`
- `rl_core.algorithms.dqn`: `QNetwork`, `DQNConfig`, `DQNTrainer`
- `rl_core.algorithms.sac`: `GaussianPolicy`, `TwinQNetwork`, `SACConfig`, `SACTrainer`
