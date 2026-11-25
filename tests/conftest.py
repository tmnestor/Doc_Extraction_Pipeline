"""Fixtures shared by the whole test suite."""

import sys
from pathlib import Path
from typing import Any

import pytest

# The stand-in for the document processor that tests of whole stages share (tests/stages/bench.py).
sys.path.insert(0, str(Path(__file__).resolve().parent / "stages"))


@pytest.fixture
def bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A folder of blank pages, the settings to read them, and a stand-in for the document processor."""
    from bench import Bench

    return Bench(tmp_path, monkeypatch)
