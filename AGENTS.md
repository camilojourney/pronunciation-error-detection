# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Default contributor install is `uv sync` (`README.md`). Base dependencies in `pyproject.toml` must include every import checked by `test_setup.py`; `tests/test_setup_contract.py` enforces that. Costly ML stays in the `ml` extra; notebook/Quarto tooling stays in `experiments`.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
