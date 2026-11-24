"""The run's settings do not work out the folders a task reads and writes: the task's own settings do, once.

Until 25 November 2026 ``Settings.load`` worked the folder of pages and the output folder out a second
time, after ``common.stage_settings.resolve`` had, with its own copy of the message for a folder that was
not set. It also refused to load unless the folder of pages existed. The SROIE task, which reads the
folder of its own block, therefore could not start unless the pipeline's folder of pages was there.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference"))

from model_inputs import test_config  # noqa: E402

from common.config_schema import ConfigError  # noqa: E402
from common.prompt_trace import effective_trace_path  # noqa: E402
from common.settings import Settings  # noqa: E402

test_config.__test__ = False  # type: ignore[attr-defined]

MODEL = "gemma4-31b-w4a16-vllm"


def test_the_settings_load_with_no_folder_of_pages_anywhere(tmp_path: Path) -> None:
    """As the SROIE task loads them: the model, and nothing else."""
    config = test_config(tmp_path)
    assert not (tmp_path / "pages").exists()

    settings = Settings.load({"model_type": MODEL}, config_path=config)

    assert settings.model_type == MODEL
    assert settings.output_dir is None
    assert not hasattr(settings, "data_dir")


def test_a_stage_gives_the_folder_it_writes_to(tmp_path: Path) -> None:
    settings = Settings.load(
        {"model_type": MODEL, "output_dir": str(tmp_path / "out")}, config_path=test_config(tmp_path)
    )

    assert settings.output_dir == tmp_path / "out"


def test_a_folder_of_pages_handed_to_the_settings_is_refused_and_not_passed_over(tmp_path: Path) -> None:
    """A caller that still hands one over is told, where it would otherwise seem to have been heard."""
    with pytest.raises(TypeError, match="data_dir"):
        Settings.load({"model_type": MODEL, "data_dir": str(tmp_path)}, config_path=test_config(tmp_path))


def test_what_is_sent_to_the_model_is_kept_under_the_output_folder_when_no_place_is_named(
    tmp_path: Path,
) -> None:
    settings = Settings.load(
        {"model_type": MODEL, "output_dir": str(tmp_path / "out")}, config_path=test_config(tmp_path)
    )
    settings.trace_raw_prompts = True

    assert effective_trace_path(settings) == str(tmp_path / "out" / "raw_prompt_trace.jsonl")


def test_keeping_what_is_sent_with_no_place_for_it_is_refused_with_how_to_mend_it(tmp_path: Path) -> None:
    """A task with no output folder, the SROIE task, cannot have the record put under one."""
    settings = Settings.load({"model_type": MODEL}, config_path=test_config(tmp_path))
    settings.trace_raw_prompts = True

    with pytest.raises(ConfigError) as refused:
        effective_trace_path(settings)

    message = str(refused.value)
    assert all(part in message for part in ("What:", "Where:", "Expected:", "How to fix:"))
    assert "inference.tracing.path" in message and str(settings.config_path) in message


def test_nothing_is_asked_when_what_is_sent_is_not_kept(tmp_path: Path) -> None:
    settings = Settings.load({"model_type": MODEL}, config_path=test_config(tmp_path))

    assert settings.trace_raw_prompts is False and effective_trace_path(settings) is None
