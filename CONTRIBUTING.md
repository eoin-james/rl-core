# Maintaining rl-core

This document is for the maintainer of rl-core. It covers day-to-day development, the release process, and how to handle breaking changes safely across consuming repos.

---

## Setup

```bash
git clone https://github.com/eoin-james/rl-core.git
cd rl-core
poetry install --with dev  # installs main + dev deps (ruff, ty, pytest)
```

---

## Branch protection (one-time GitHub setup)

This must be configured once in GitHub settings. Go to **Settings → Branches → Add rule** for `main`:

- [x] Require a pull request before merging
- [x] Require status checks to pass before merging
  - Add required checks: `Lint & Type Check`, `Tests`
- [x] Require branches to be up to date before merging
- [x] Do not allow bypassing the above settings

This enforces the feature-branch workflow: nothing lands on `main` without CI passing. The `auto-merge.yml` workflow then merges PRs automatically once checks pass — no manual merge step needed for routine work.

**Workflow for all changes:**

```bash
git checkout -b feat/my-feature   # never work directly on main
# ... make changes, run /check to verify locally ...
git push origin feat/my-feature
gh pr create --fill                # open PR; CI triggers automatically
# CI passes → auto-merge fires → branch deleted → done
```

---

## Making changes

### Branching

Work on a feature branch, not directly on `main`:

```bash
git checkout -b feat/my-feature   # new feature
git checkout -b fix/bug-name      # bug fix
git checkout -b break/rename-foo  # breaking change
```

### PR labels

Every PR needs exactly one label before merging:

| Label | When to use |
|-------|-------------|
| `feat` | New public API, new module, new algorithm |
| `fix` | Bug fix, no API change |
| `refactor` | Internal restructure, no API change |
| `breaking` | Renames, removes, or changes the signature of anything public |
| `docs` | README, CHANGELOG, CLAUDE.md only |

### Before merging

```bash
poetry run ruff check .       # must pass
poetry run ruff format .      # must pass
poetry run ty check           # must pass (warnings OK, errors not)
poetry run pytest tests/ -v   # all tests must pass
```

---

## Release process

Every merge to `main` that is user-visible gets a release tag. Do this immediately after merging, not days later.

### Step 1 — determine the version bump

| What changed | Bump |
|---|---|
| Bug fix only | `patch` → 0.1.**x** |
| New feature, backwards compatible | `minor` → 0.**x**.0 |
| Any public API renamed / removed / signature changed | `major` → **x**.0.0 |

Current version is in `pyproject.toml` under `[project] version`.

### Step 2 — update CHANGELOG.md

Add a new entry at the top (above the previous release, below the header):

```markdown
## [vX.Y.Z] - YYYY-MM-DD

### Added
- ...

### Changed
- ...

### Fixed
- ...

### Removed
- ...
```

Only include sections that have entries. Keep each bullet to one line.

### Step 3 — bump the version in pyproject.toml

```toml
[project]
version = "X.Y.Z"
```

### Step 4 — commit, tag, push

```bash
git add CHANGELOG.md pyproject.toml
git commit -m "Release vX.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

### Step 5 — notify consuming repos

If the release includes anything labelled `breaking`, open a PR or leave a note in the consuming repo pointing to the CHANGELOG entry and listing what needs updating.

---

## Breaking change protocol

A change is **breaking** if it does any of the following to anything in a public module's `__all__` or directly importable namespace:

- Renames a class, function, or dataclass field
- Removes a class or function
- Changes a function's signature (parameter names, types, defaults)
- Changes the return type or structure of a return value

### Checklist before merging a breaking PR

1. Label the PR `breaking`
2. In the PR description, list every consuming repo and exactly what they'll need to change
3. Add a `### Removed` or `### Changed` entry to CHANGELOG with a migration note:
   ```markdown
   ### Changed
   - `save_checkpoint(ckpt, path)` — parameter order reversed to `(path, ckpt)`.
     **Migration:** swap argument order in all call sites.
   ```
4. Bump major version
5. After merging, update consuming repos' pins or open issues in them

---

## Adding a new algorithm

New algorithms live under `rl_core/algorithms/<name>/` with this structure:

```
algorithms/myalgo/
├── __init__.py      # exports: MyAlgoConfig, MyAlgoTrainer, networks
├── network.py       # network definitions
└── trainer.py       # MyAlgoConfig (frozen dataclass) + MyAlgoTrainer
```

`MyAlgoTrainer` must expose:
- `select_action(obs, **kwargs)` — returns action
- `train_step(batch)` — returns `dict[str, float]` of metrics
- `state_dicts()` — returns `dict[str, Any]`
- `load_state_dicts(state_dicts)` — loads from above

Metric keys returned by `train_step` should follow the `loss/`, `q/`, `entropy/` prefix conventions so they route cleanly through `NamespacedLogger`.

Add tests under `tests/algorithms/test_myalgo.py` before opening the PR.

---

## Adding to rl_core.experiments

The experiments module is the most likely place consuming repos will need extensions. Common requests:

- New status fields in `status.json` — add to `_write_status`, backwards compatible
- New checkpoint metadata — add to `ExperimentRun.checkpoint()`, backwards compatible
- New logger routing tiers (e.g. `eval/`) — add to `NamespacedLogger`, backwards compatible

None of the above are breaking unless an existing parameter name or return value changes.
