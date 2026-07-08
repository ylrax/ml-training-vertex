# template project with python

My personal template for python projects. Use this as a starting point for new
python projects: clone/copy it, rename the project in [pyproject.toml](pyproject.toml),
and start building.

## What's included

- [pyproject.toml](pyproject.toml) — project metadata and dependencies, managed with `uv`
- [main.py](main.py) — entry point
- [test/](test/) — test directory
- [general_testing.ipynb](general_testing.ipynb) — scratch notebook for quick experiments
- `.python-version` — pins the python version (currently 3.14) for `uv`

## Setup with uv

[uv](https://docs.astral.sh/uv/) is used for python version and dependency management.

1. Sync dependencies (creates `.venv` and installs everything from `uv.lock`):

   ```powershell
   uv sync
   ```

2. Run/check the project:

   ```powershell
   uv run main.py
   ```

3. Add a new dependency:

   ```powershell
   uv add <package-name>
   ```

4. Run tests:

   ```powershell
   uv run pytest
   ```
