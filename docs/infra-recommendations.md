# Infrastructure recommendations for Pronunciation Error Detection

This document outlines repository architecture, environments, data management, CI/CD, and future app integration. It's optimized for fast research now and smooth evolution into a product later.

## TL;DR
- Start with a single-repo multi-package (monorepo-lite) using Python as the core. Keep web/app in separate directories but same repo.
- Use `uv` or `poetry` for Python env + locking; add `ruff` + `mypy` for quality.
- Use `just` for task automation (modern alternative to Make).
- Use simple local data management with scripts (no DVC for now). Keep large files in `.gitignore`, track metadata in git.
- Pre-commit hooks automate code quality checks before commits.
- Dependency groups separate ML, dev, and experiment dependencies for faster installs.
- Minimal CI: lint + tests on PR; optional workflow to run small pipeline on sample.
- Prepare seams for a future Whisper-powered app (CLI and service boundary).

## Repo architecture options

- Monorepo (recommended)
  - Pros: single source of truth, easy refactors across packages (core, tooling, app), simpler onboarding.
  - Cons: needs simple tooling to avoid dependency hell.

- Polyrepo (not recommended now)
  - Pros: isolation per component, independent releases.
  - Cons: upfront overhead for research; cross-repo changes are slow.

### Recommended layout

```
pronunciation-error-detection/
  docs/
  ped/                  # core python package (processing + pipeline)
  scripts/              # light CLIs to run pipelines
  notebooks/            # experiments
  data/                 # local-only; ignored except tiny samples
  tests/
  apps/
    whisper-assistant/  # future app (CLI/API/desktop)
```

Keep a single `pyproject.toml` at repo root for now. If/when apps diversify (web, mobile), introduce per-app manifests without splitting repos.

## Environments & dependencies
- **Python 3.11+** (for perf + typing). Use `uv` or `poetry`; fallback `pip-tools`.
- **Quality tools**: `ruff` (lint/format), `mypy` (types), `pytest` (tests), `pre-commit` (git hooks).
- **Dependency groups** (see `pyproject.toml`):
  - Core: `python-Levenshtein`, `jiwer` (minimal runtime)
  - ML: `faster-whisper`, `torch`, `spacy`, `phonemizer`, `g2p-en`
  - Dev: `pytest`, `ruff`, `mypy`, `pre-commit`
  - Experiments: `jupyter`, `matplotlib`, `pandas`, `seaborn`
- **macOS system deps**: `ffmpeg` (for Whisper), optional `espeak`/`espeak-ng` for phonemizer.

**Installation:**
```bash
# Full dev setup (recommended)
just setup-dev

# Or manually
pip install -e ".[ml,dev,experiments]"

# Setup git hooks (one-time)
just setup-hooks

# Or do everything in one command
just setup
```

## Data management
- **Simple local approach** (no DVC for now): Keep large files local and in `.gitignore`.
- Track dataset metadata in git-versioned files (e.g., `data/manifests/datasets.yaml`).
- Directory convention:
  - `data/raw/` original downloads (local, .gitignored)
  - `data/interim/` cleaned/intermediate (local, .gitignored)
  - `data/processed/` final aligned outputs (local, .gitignored)
  - `data/sample/` tiny samples for testing (tracked in git)
  - `artifacts/` models, metrics, plots (local, .gitignored)

**When to migrate to DVC/cloud:**
- Multiple collaborators need same datasets
- Need reproducible pipeline versioning
- Moving to production deployment

**Current workflow:**
1. Download datasets manually to `data/raw/`
2. Document sources in `data/README.md` or manifest files
3. Run pipelines locally, outputs go to `data/processed/`
4. Keep everything in `.gitignore` except small samples

## CI/CD
- **Minimal GitHub Actions** (optional for now, can add later):
  - `lint` job: ruff + mypy
  - `test` job: pytest (unit tests only, no ML dependencies)
  - `smoke` job: run tiny pipeline on sample audio (optional, nightly)
- **Pre-commit hooks** handle quality locally, reducing CI failures
- Install only `[dev]` dependencies in CI for speed (skip heavy ML libs)

## Observability & experiment tracking
- **Local-first approach**: Keep logs and metrics under `artifacts/` (JSON, CSV, plots)
- **Optional later**: `mlflow` (local-first, no signup) or `wandb` (better for collaboration)
- Start simple with structured logging to files

## Security & secrets
- Use `.env` for local; GitHub Encrypted Secrets for CI. Never commit creds.

## Future app integration
- Define a thin boundary: a CLI/service that accepts audio file(s) and returns JSON insights.
- Later, wrap as an HTTP API (FastAPI) and plug into web/mobile.
- Keep core logic in `ped/`, never inside app folders.

## Risks & mitigations
- **Whisper dependency weight**: Use `faster-whisper` for inference speed (4x faster, lower memory).
- **Dataset licensing/size**: Keep large files local, use tiny samples for testing/CI.
- **Phoneme alignment complexity**: Start with proxy metrics (WER + grapheme diffs), iterate to phoneme-level with `phonemizer`.
- **Tool complexity**: Start simple (local files, basic tools), migrate to DVC/cloud only when collaboration requires it.

## Task automation with `just`

We use [`just`](https://github.com/casey/just) instead of Make for better ergonomics and cross-platform support.

**Install just:**
```bash
# macOS
brew install just

# Or via cargo
cargo install just
```

**Common commands:**
```bash
just --list              # Show all available commands
just setup               # First-time setup (install deps + hooks)
just format              # Auto-format code
just lint                # Run linters
just test                # Run tests
just check               # Run format + lint + test
just demo                # Run demo with defaults
just demo "hello" "helo" # Run demo with custom text
just clean               # Clean Python cache files
```

## Quick start for new contributors
```bash
# 1. Install just
brew install just  # or: cargo install just

# 2. Clone repo
git clone <repo-url>
cd pronunciation-error-detection

# 3. One-command setup
just setup

# 4. Verify setup
just check

# 5. Run demo
just demo
```