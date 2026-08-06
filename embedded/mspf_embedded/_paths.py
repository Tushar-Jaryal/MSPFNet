"""Ensure repo ``src/`` is on sys.path when running embedded scripts."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"


def ensure_repo_src() -> Path:
    src = str(_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    return _REPO_ROOT


def repo_root() -> Path:
    return _REPO_ROOT


def embedded_root() -> Path:
    return _REPO_ROOT / "embedded"
