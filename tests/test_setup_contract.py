"""Guard the documented setup contract: `uv sync` must install test_setup imports.

Does not download the L2-ARCTIC corpus, call APIs, or train models.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _spec_name(spec: str) -> str:
    return spec.split(">")[0].split("=")[0].split("<")[0].split("[")[0].strip().lower()


def _setup_required_packages() -> list[str]:
    tree = ast.parse((ROOT / "test_setup.py").read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict) or not node.keys:
            continue
        keys: list[str] = []
        for key in node.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.append(key.value)
            else:
                keys = []
                break
        if {"nltk", "textgrid", "pandas", "matplotlib", "seaborn"}.issubset(keys):
            return keys
    raise AssertionError("could not find setup-required package dict in test_setup.py")


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def _lock_ped() -> dict:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    for package in lock["package"]:
        if package["name"] == "ped":
            return package
    raise AssertionError("ped package missing from uv.lock")


def test_readme_documents_uv_sync():
    readme = (ROOT / "README.md").read_text()
    assert "uv sync" in readme
    assert "uv run python -c \"import nltk" in readme


def test_base_dependencies_cover_setup_imports():
    required = _setup_required_packages()
    base = {_spec_name(dep) for dep in _pyproject()["project"]["dependencies"]}
    missing = [pkg for pkg in required if pkg not in base]
    assert missing == [], f"pyproject.toml base dependencies omit setup imports: {missing}"


def test_setup_imports_are_not_extra_only_in_lockfile():
    required = set(_setup_required_packages())
    ped = _lock_ped()
    lock_base = {dep["name"] for dep in ped.get("dependencies", [])}
    missing = sorted(required - lock_base)
    assert missing == [], f"uv.lock records setup imports outside default ped deps: {missing}"


def test_costly_ml_stays_optional():
    extras = _pyproject()["project"]["optional-dependencies"]
    ml = {_spec_name(dep) for dep in extras["ml"]}
    base = {_spec_name(dep) for dep in _pyproject()["project"]["dependencies"]}
    for pkg in ("torch", "torchaudio", "faster-whisper", "spacy"):
        assert pkg in ml
        assert pkg not in base
