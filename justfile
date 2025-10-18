# Just command runner for pronunciation-error-detection
# Install: https://github.com/casey/just
# Usage: just <command>
# List all commands: just --list

# Format code with ruff
format:
    ruff check --fix .
    ruff format .

# Run linters (ruff + mypy)
lint:
    ruff check .
    mypy ped

# Run tests
test:
    pytest -q

# Run demo with default text
demo ref="this is a test" hyp="this test":
    python3 scripts/run_text_alignment.py --ref "{{ref}}" --hyp "{{hyp}}"

# Install core dependencies only
install:
    pip install -e .

# Install all dependencies for local development (ml + dev + experiments)
setup-dev:
    pip install -e ".[ml,dev,experiments]"

# Setup pre-commit hooks (run once after cloning)
setup-hooks:
    pip install pre-commit
    pre-commit install
    @echo "✓ Pre-commit hooks installed! They will run automatically on git commit."

# Full first-time setup (install everything + hooks)
setup: setup-dev setup-hooks
    @echo "✓ Development environment ready!"

# Run pre-commit hooks manually on all files
pre-commit:
    pre-commit run --all-files

# Clean Python cache files
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Show Python and package versions
versions:
    @echo "Python version:"
    @python3 --version
    @echo "\nInstalled packages:"
    @pip list | grep -E "(ruff|mypy|pytest|whisper|torch|spacy)" || echo "No packages installed yet"

# Run full quality check (format + lint + test)
check: format lint test
    @echo "✓ All checks passed!"
