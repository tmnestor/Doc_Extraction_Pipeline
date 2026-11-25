"""Every error about a setting is one kind of error, so a run it stops ends with the reason.

Until 24 November 2026 there were seven kinds, and the task runner knew one. A photo split setting out of
range, a batch size for no document type, an unknown model or a schema file with a field missing each
stopped the run with a traceback, the message somewhere inside it.
"""

import importlib
from pathlib import Path

import pytest

from common.config_schema import ConfigError
from stages import __main__ as starter

FAMILY = [
    ("common.extraction_batching", "BatchingConfigError"),
    ("common.vllm_stage_overrides", "StageOverrideConfigError"),
    ("common.photo_split", "PhotoSplitConfigError"),
    ("common.sheet_finder_config", "FinderConfigError"),
    ("common.field_schema", "FieldSchemaError"),
]


@pytest.mark.parametrize(("module", "name"), FAMILY)
def test_each_kind_of_settings_error_is_the_one_kind(module: str, name: str) -> None:
    kind = getattr(importlib.import_module(module), name)

    assert issubclass(kind, ConfigError)
    # Code that catches a wrong value by its older, wider name still does.
    assert issubclass(kind, ValueError)


def test_no_module_declares_a_settings_error_outside_the_family() -> None:
    """A new reader of settings that brings its own kind of error would be the eighth."""
    repo = Path(__file__).resolve().parents[2]
    outside = []
    for source in [
        *repo.glob("common/**/*.py"),
        *repo.glob("models/**/*.py"),
        *repo.glob("stages/**/*.py"),
    ]:
        for line in source.read_text().splitlines():
            if line.startswith("class ") and "ConfigError(" in line and "(ConfigError)" not in line:
                if line.startswith("class ConfigError("):
                    continue
                outside.append(f"{source.relative_to(repo)}: {line}")

    assert outside == []


def test_a_model_that_is_not_registered_is_a_settings_error() -> None:
    from models.registry import get_model

    with pytest.raises(ConfigError) as refused:
        get_model("a-model-nobody-registered")

    assert "a-model-nobody-registered" in str(refused.value)


def test_a_prompt_file_error_is_a_settings_error(tmp_path: Path) -> None:
    from common.prompt_catalog import PromptCatalog

    (tmp_path / "document_type_detection.yaml").write_text("classification:\n  opening: Look.\n")

    with pytest.raises(ConfigError) as refused:
        PromptCatalog(tmp_path).get_classification_questions()

    assert all(part in str(refused.value) for part in ("What:", "Where:", "Expected:", "How to fix:"))


def test_a_run_stopped_by_any_of_them_ends_with_the_reason_and_not_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from common.photo_split import PhotoSplitConfigError

    def refuses(*args: object, **kwargs: object) -> None:
        raise PhotoSplitConfigError(
            "What: margin_frac is 7.\nWhere: here\nExpected: a share\nHow to fix: so"
        )

    monkeypatch.setattr("stages.split.run", refuses)
    monkeypatch.setattr("stages.split.check", lambda task: None)
    (tmp_path / "pages").mkdir()
    for name, value in {
        "KFP_TASK": "split",
        "model": "gemma4-31b-w4a16-vllm",
        "image_dir": str(tmp_path / "pages"),
        "output": str(tmp_path / "out"),
        "DOC_EXTRACT_LOG_DIR": str(tmp_path / "logs"),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("CLEAR_PREV_OUTPUT", raising=False)

    assert starter.main() == 1

    printed = capsys.readouterr().out
    assert printed.startswith("FATAL: the task cannot be started.")
    assert "margin_frac is 7" in printed
