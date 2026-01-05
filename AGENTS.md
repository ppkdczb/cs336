# Repository Guidelines

## Project Structure & Module Organization
- `cs336_basics/` contains the assignment code (tokenization, model components, utilities).
- `tests/` holds pytest suites, fixtures, and snapshot data in `tests/_snapshots/`.
- `data/` is for downloaded training/validation corpora (kept out of the package code).
- `make_submission.sh` is the helper script for packaging submission artifacts.
- `cs336_spring2025_assignment1_basics.pdf` is the assignment handout and reference spec.

## Build, Test, and Development Commands
- `uv run pytest`: run the full test suite (default entry point).
- `uv run cs336_basics/pretokenization_example.py`: run the bundled example script.
- `uv run <path/to/script.py>`: run any repo script with the managed environment.
- Data download (optional): follow the `wget` commands in `README.md` to populate `data/`.

## Coding Style & Naming Conventions
- Python, 4-space indentation, standard PEP 8 naming (`snake_case` for functions/vars, `PascalCase` for classes).
- Line length is 120 per `pyproject.toml` (ruff config).
- Keep changes localized to `cs336_basics/` and update adapter hooks in `tests/adapters.py` when required by tests.

## Testing Guidelines
- Framework: `pytest` (configured in `pyproject.toml`).
- Tests live in `tests/` and are named `test_*.py` with `test_*` functions.
- Snapshot files live in `tests/_snapshots/` (`.npz`, `.pkl`) and must stay in sync with test outputs.
- Run targeted tests with `uv run pytest tests/test_tokenizer.py` during development.

## Commit & Pull Request Guidelines
- Commit messages in history are short and imperative (e.g., “Fix typing…”, “Update lockfile…”).
- Version bumps use a prefix like `Version x.y.z:` or `x.y.z, …`; follow this pattern if releasing.
- PRs should include: a brief summary, tests run (command + result), and any data or snapshot changes.
- If your change affects the assignment spec or fixtures, mention the file path explicitly.

## Configuration & Data Notes
- Environment is managed via `uv` with dependencies in `pyproject.toml` and `uv.lock`.
- Large datasets belong under `data/` and should not be committed unless explicitly required.
