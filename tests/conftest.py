"""Fixtures shared by the whole test suite."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def run_config_path() -> Path:
    """The run config shipped with the repository."""
    return Path(__file__).resolve().parents[1] / "config" / "run_config.yml"
