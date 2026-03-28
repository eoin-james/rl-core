# Changelog

All notable changes to rl-core are documented here.

Format: `## [vX.Y.Z] - YYYY-MM-DD` followed by `### Added / Changed / Fixed / Removed`.
A `breaking` label on the GitHub PR signals a major version bump.

---

## [v1.0.0] - 2026-03-28

First stable release. Public API is now considered settled.

### Added
- Branch protection on `main` (PRs required, CI must pass)
- `auto-merge.yml` workflow — PRs squash-merge automatically on CI pass
- PR template with breaking change checklist
- `CONTRIBUTING.md` — release process, PR labels, breaking change protocol, new algorithm checklist
- GitHub issue templates — change request, bug report; blank issues disabled
- `CLAUDE.md` — project-level guidance for Claude Code with versioning rules, consumer repo context, release process
- `.claude/commands/release.md` and `check.md` slash commands
- `RL/CLAUDE.md` — ecosystem-level hive mind loaded across all sibling repos

### Fixed
- Ruff lint errors in `run_manager.py` and `tests/test_experiments.py` (UP035, UP017, I001, SIM117, E501)
- CI now installs dev dependencies (`--with dev`) so ruff, ty, and pytest are available
- `test` job depends on `quality` job — fails fast rather than running in parallel

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
