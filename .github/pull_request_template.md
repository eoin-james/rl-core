## What does this PR do?
<!-- One paragraph. Link to the issue it closes if applicable: "Closes #123" -->

## Is this a breaking change?

- [ ] **Yes — I have labelled this PR `breaking`**
  - Consuming repos affected: <!-- list them -->
  - What they need to update: <!-- be specific: "rename X to Y at all call sites" -->
- [ ] No — fully backwards compatible

## Checklist

- [ ] `poetry run ruff check .` passes
- [ ] `poetry run ruff format --check .` passes
- [ ] `poetry run pytest tests/ -v` passes
- [ ] CHANGELOG.md updated
- [ ] Version bumped in `pyproject.toml` if this is a release PR
