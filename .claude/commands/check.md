Run the full rl-core quality check suite and report results.

Execute these four commands in sequence:

1. `poetry run ruff check .` — linting
2. `poetry run ruff format --check .` — formatting (check only, don't reformat)
3. `poetry run ty check` — type checking
4. `poetry run pytest tests/ -v` — tests

For each step, report pass or fail. If any step fails, show the relevant output and suggest a fix. Do not proceed to the next step after a failure — fix it first (or ask the user if the fix is non-trivial).

At the end, give a one-line summary: all pass, or list what failed.
