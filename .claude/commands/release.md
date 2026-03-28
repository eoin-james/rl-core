Cut a new rl-core release. The target version is: $ARGUMENTS

Work through these steps in order. Stop and ask if anything is unclear before proceeding.

1. **Run the full check suite** — all three must pass before proceeding:
   ```
   poetry run ruff check .
   poetry run ruff format --check .
   poetry run pytest tests/ -v
   ```
   If any fail, fix them first and ask the user to confirm before continuing.

2. **Determine the version bump** based on commits since the last tag:
   - Run `git log $(git describe --tags --abbrev=0)..HEAD --oneline` to see what changed
   - `patch` if only bug fixes (no new public API, no removals)
   - `minor` if new features added, all backwards compatible
   - `major` if anything public was renamed, removed, or had its signature changed
   - If a target version was provided in $ARGUMENTS, use that — but flag if it doesn't match the expected bump

3. **Update CHANGELOG.md** — add a new entry at the top, above the previous release:
   ```
   ## [vX.Y.Z] - YYYY-MM-DD

   ### Added / Changed / Fixed / Removed
   - ...
   ```
   Derive entries from the git log. Only include sections with actual content.

4. **Bump the version in pyproject.toml** under `[project] version = "X.Y.Z"`.

5. **Commit and tag**:
   ```
   git add CHANGELOG.md pyproject.toml
   git commit -m "Release vX.Y.Z"
   git tag vX.Y.Z
   ```

6. **Push**:
   ```
   git push origin main --tags
   ```

7. **If this is a breaking (major) release**, list the consuming repos and what they need to update:
   - `../rl-evo-lab` — check their pyproject.toml for what they import from rl-core
   - `../lang-conditioned-control` — same

Show the user the final CHANGELOG entry and tag before pushing. Ask for confirmation.
