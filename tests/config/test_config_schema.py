"""A setting the pipeline does not know stops the run, and every setting it knows is in the shipped file.

Before this check, a key that no code read was silently ignored. A settings file could say one thing while
the run did another, and a file left over from an older version of the pipeline ran as if nothing in it
had been removed.
"""

import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from common.config_schema import KNOWN_KEYS, check_known_keys, leaf_paths
from common.settings import read_settings_file

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_CONFIG = REPO_ROOT / "config" / "run_config.yml"
PASSED_TO_THE_ENGINE = ("mm_processor_kwargs", "hf_overrides")


def shipped() -> dict[str, Any]:
    return yaml.safe_load(RUN_CONFIG.read_text())


def config_with(tmp_path: Path, change) -> Path:
    raw = shipped()
    change(raw)
    path = tmp_path / "run_config.yml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def python_sources() -> dict[str, str]:
    listed = subprocess.run(
        ["git", "ls-files", "*.py", "*.sh"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    skipped = ("tests/", "common/config_schema.py")
    return {name: (REPO_ROOT / name).read_text() for name in listed if not name.startswith(skipped)}


def test_the_shipped_settings_hold_no_unknown_key() -> None:
    check_known_keys(shipped(), config_path=RUN_CONFIG)


def test_every_known_key_is_in_the_shipped_settings() -> None:
    present = leaf_paths(shipped())

    unused = sorted(known for known in KNOWN_KEYS if not any(_covers(known, path) for path in present))

    assert unused == []


def _covers(known: str, path: str) -> bool:
    """Whether a declared key accounts for a leaf of the shipped file."""
    declared, parts = known.split("."), path.split(".")
    if declared[-1] == "**":
        declared = declared[:-1]
        parts = parts[: len(declared)]
    return len(parts) == len(declared) and all(d in ("*", p) for d, p in zip(declared, parts, strict=True))


def test_an_unknown_key_stops_the_run_with_a_full_diagnostic(tmp_path: Path) -> None:
    path = config_with(tmp_path, lambda raw: raw.update(pipeline={"trust": {"enabled": False}}))

    with pytest.raises(ValueError) as raised:
        read_settings_file(path)

    message = str(raised.value)
    assert "pipeline.trust.enabled" in message and str(path.resolve()) in message
    assert all(part in message for part in ("What:", "Where:", "Expected:", "How to fix:"))
    assert "removed from the pipeline" in message


@pytest.mark.parametrize(
    "removed",
    [
        ("bootstrap", "gpus", "engine_split", "tensor"),
        ("pipeline", "processing", "bank_strategy", "two_turn"),
        ("bootstrap", "model", "max_tiles", 18),
        ("pipeline", "processing", "bank_v2", True),
    ],
)
def test_a_setting_from_before_the_refactor_is_refused(tmp_path: Path, removed: tuple) -> None:
    *section, key, value = removed

    def put_back(raw: dict[str, Any]) -> None:
        block = raw
        for name in section:
            block = block.setdefault(name, {})
        block[key] = value

    with pytest.raises(ValueError) as raised:
        read_settings_file(config_with(tmp_path, put_back))

    assert ".".join((*section, key)) in str(raised.value)


@pytest.mark.parametrize(
    "old_key",
    [
        ("inference", "generation", "defaults", "temperature"),
        ("inference", "generation", "models", "gemma4-31b", "max_new_tokens_base"),
        ("pipeline", "token_budgets", "fallback_base"),
        ("bootstrap", "model", "device_map"),
        ("pipeline", "processing", "debug"),
    ],
)
def test_a_setting_that_nothing_read_is_now_refused(tmp_path: Path, old_key: tuple[str, ...]) -> None:
    """Removed 24 November 2026: each sat in the file looking like an instruction and changed nothing."""

    def put_back(raw: dict) -> None:
        block = raw
        for part in old_key[:-1]:
            block = block.setdefault(part, {})
        block[old_key[-1]] = 1

    with pytest.raises(ValueError) as raised:
        read_settings_file(config_with(tmp_path, put_back))

    assert ".".join(old_key) in str(raised.value)


def test_a_misspelt_key_is_named_with_the_known_keys_beside_it(tmp_path: Path) -> None:
    path = config_with(tmp_path, lambda raw: raw["bootstrap"]["model"].update(typ="a-model"))

    with pytest.raises(ValueError) as raised:
        read_settings_file(path)

    message = str(raised.value)
    assert "bootstrap.model.typ" in message
    assert "bootstrap.model.type" in message


@pytest.mark.parametrize("key", sorted(KNOWN_KEYS))
def test_every_known_key_is_read_by_the_code(key: str) -> None:
    """A key whose name appears nowhere in the code cannot be read by it."""
    names = [part for part in key.split(".") if part not in ("*", "**")]
    name = names[-1]
    if name in PASSED_TO_THE_ENGINE or key.endswith(".*"):
        section = names[-1]
        assert any(section in text for text in python_sources().values()), section
        return

    assert any(name in text for text in python_sources().values()), f"nothing mentions {name!r}"
