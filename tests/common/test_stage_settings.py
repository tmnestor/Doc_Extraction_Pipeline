"""A task's settings come from the environment first and the settings file second, and are worked out once.

Until 21 November 2026 the entry script worked these out in shell: it read six values from the settings file
with a helper, laid the environment over them, and built each stage's flags by hand. Now
`common.stage_settings.resolve` does it, and `python -m stages` runs the task with the result.
"""

import logging
from pathlib import Path
from typing import Any

import pytest
import yaml

from common.stage_settings import TASKS, ConfigError, StageSettings, log_dir, resolve
from stages import __main__ as starter

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_CONFIG = REPO_ROOT / "config" / "run_config.yml"
OLD_PREFIX = "L" + "MM_"

# What a sandbox command file sets for a stage, and the paths the entry script gave for them before.
D = "/home/jovyan/nfs_share/tod_2026/evaluation_data"
USUAL = {
    "model": "gemma4-31b-w4a16-vllm",
    "image_dir": f"{D}/snapshot_20261021/plain_20261021",
    "output": f"{D}/output_run",
    "DOC_EXTRACT_LOG_DIR": f"{D}/output_run/logs",
}


def config_with(tmp_path: Path, change) -> Path:
    raw = yaml.safe_load(RUN_CONFIG.read_text())
    change(raw)
    path = tmp_path / "run_config.yml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def shipped(key: str) -> Any:
    value: Any = yaml.safe_load(RUN_CONFIG.read_text())
    for part in key.split("."):
        value = value[part]
    return value


def assert_diagnostic(message: str, *names: str) -> None:
    assert all(part in message for part in ("What:", "Where:", "Expected:", "How to fix:")), message
    assert all(name in message for name in names), message


def test_the_six_tasks() -> None:
    assert TASKS == ("split", "classify", "extract", "clean", "evaluate", "sroie")


@pytest.mark.parametrize("task", ["split", "classify", "extract", "clean"])
def test_the_usual_variables_give_the_paths_the_entry_script_gave(task: str) -> None:
    settings = resolve(task, {"KFP_TASK": task, **USUAL}, RUN_CONFIG)

    assert settings.model == "gemma4-31b-w4a16-vllm"
    assert settings.image_dir == Path(f"{D}/snapshot_20261021/plain_20261021")
    assert settings.output == Path(f"{D}/output_run")
    assert settings.log_dir == Path(f"{D}/output_run/logs")
    assert settings.split_dir == Path(f"{D}/output_run/split")
    assert settings.classifications == Path(f"{D}/output_run/classifications.jsonl")
    assert settings.raw_extractions == Path(f"{D}/output_run/raw_extractions.jsonl")
    assert settings.cleaned_extractions == Path(f"{D}/output_run/cleaned_extractions.jsonl")
    assert settings.evaluation_dir == Path(f"{D}/output_run/evaluation")
    assert settings.clear_previous_output is False and settings.sroie_max_images is None


def test_the_environment_wins_over_the_settings_file() -> None:
    settings = resolve("classify", USUAL, RUN_CONFIG)

    assert str(settings.image_dir) != shipped("pipeline.information_extraction.input.dir")
    assert str(settings.output) != shipped("pipeline.information_extraction.output.dir")


@pytest.mark.parametrize("unset", [None, "", "None"])
def test_the_settings_file_is_used_when_a_variable_is_unset_empty_or_none(unset: str | None) -> None:
    environ = {name: unset for name in USUAL if unset is not None}

    settings = resolve("evaluate", environ, RUN_CONFIG)

    assert settings.model == shipped("bootstrap.model.type")
    assert str(settings.image_dir) == shipped("pipeline.information_extraction.input.dir")
    assert str(settings.output) == shipped("pipeline.information_extraction.output.dir")
    assert str(settings.ground_truth) == shipped("pipeline.information_extraction.input.ground_truth")
    assert str(settings.log_dir) == shipped("bootstrap.logging.log_dir")


@pytest.mark.parametrize("task", ["", "run_info_extract", "trust"])
def test_a_task_that_is_not_one_of_the_six_is_refused(task: str) -> None:
    with pytest.raises(ConfigError) as raised:
        resolve(task, USUAL, RUN_CONFIG)

    assert_diagnostic(str(raised.value), "KFP_TASK", *TASKS)


def test_a_missing_image_folder_is_refused_naming_the_variable_and_the_key(tmp_path: Path) -> None:
    path = config_with(
        tmp_path, lambda raw: raw["pipeline"]["information_extraction"]["input"].update(dir=None)
    )
    environ = {k: v for k, v in USUAL.items() if k != "image_dir"}

    with pytest.raises(ConfigError) as raised:
        resolve("classify", environ, path)

    assert_diagnostic(
        str(raised.value), "image_dir", "pipeline.information_extraction.input.dir", str(path.resolve())
    )


@pytest.mark.parametrize(
    "key", ["/data/pages/ground_truth.csv", "/data/pages/ground_truth", "answers.json"]
)
def test_an_answer_key_that_is_not_json_lines_stops_evaluate_before_any_work(key: str) -> None:
    with pytest.raises(ConfigError) as raised:
        resolve("evaluate", {**USUAL, "ground_truth": key}, RUN_CONFIG)

    assert_diagnostic(
        str(raised.value),
        key,
        "ground_truth",
        "pipeline.information_extraction.input.ground_truth",
        ".jsonl",
    )


def test_the_format_of_the_answer_key_is_checked_for_evaluate_alone() -> None:
    usual = {**USUAL, "ground_truth": "/data/pages/ground_truth.csv"}

    assert resolve("classify", usual, RUN_CONFIG).ground_truth == Path("/data/pages/ground_truth.csv")
    key = "/data/pages/ground_truth.jsonl"
    assert resolve("evaluate", {**USUAL, "ground_truth": key}, RUN_CONFIG).ground_truth == Path(key)


def test_the_answer_key_is_needed_to_evaluate_and_not_to_classify(tmp_path: Path) -> None:
    path = config_with(
        tmp_path, lambda raw: raw["pipeline"]["information_extraction"]["input"].update(ground_truth=None)
    )

    assert resolve("classify", USUAL, path).ground_truth is None
    with pytest.raises(ConfigError) as raised:
        resolve("evaluate", USUAL, path)
    assert_diagnostic(
        str(raised.value), "ground_truth", "pipeline.information_extraction.input.ground_truth"
    )


def test_the_images_are_not_needed_to_clean_or_to_run_the_benchmark(tmp_path: Path) -> None:
    path = config_with(
        tmp_path, lambda raw: raw["pipeline"]["information_extraction"]["input"].update(dir=None)
    )
    environ = {k: v for k, v in USUAL.items() if k != "image_dir"}

    assert resolve("clean", environ, path).image_dir is None
    assert resolve("sroie", environ, path).image_dir is None


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (None, False),
        ("", False),
        ("None", False),
        ("false", False),
        ("False", False),
        ("true", True),
        ("TRUE", True),
    ],
)
def test_starting_afresh_is_asked_for_by_name(given: str | None, expected: bool) -> None:
    environ = dict(USUAL) if given is None else {**USUAL, "CLEAR_PREV_OUTPUT": given}

    assert resolve("classify", environ, RUN_CONFIG).clear_previous_output is expected


def test_any_other_answer_to_starting_afresh_is_refused() -> None:
    with pytest.raises(ConfigError) as raised:
        resolve("classify", {**USUAL, "CLEAR_PREV_OUTPUT": "yes"}, RUN_CONFIG)

    assert_diagnostic(str(raised.value), "CLEAR_PREV_OUTPUT", "true", "false")


def test_the_benchmark_may_be_cut_short_by_a_whole_number() -> None:
    assert resolve("sroie", {**USUAL, "sroie_max_images": "40"}, RUN_CONFIG).sroie_max_images == 40
    with pytest.raises(ConfigError) as raised:
        resolve("sroie", {**USUAL, "sroie_max_images": "forty"}, RUN_CONFIG)
    assert_diagnostic(str(raised.value), "sroie_max_images")


def test_a_variable_under_the_old_prefix_is_refused() -> None:
    with pytest.raises(ConfigError) as raised:
        resolve("classify", {**USUAL, OLD_PREFIX + "LOG_DIR": "/somewhere"}, RUN_CONFIG)

    assert_diagnostic(str(raised.value), OLD_PREFIX + "LOG_DIR", "DOC_EXTRACT_LOG_DIR")


def test_the_log_folder_alone_can_be_asked_for() -> None:
    """The entry script needs it before anything else, to keep the whole run's output."""
    assert log_dir(USUAL, RUN_CONFIG) == Path(f"{D}/output_run/logs")
    assert str(log_dir({}, RUN_CONFIG)) == shipped("bootstrap.logging.log_dir")


# --- running a task with its settings ---------------------------------------------------------------


def settings_for(task: str, folder: Path, **changes: Any) -> StageSettings:
    environ = {
        "model": "gemma4-31b-w4a16-vllm",
        "image_dir": str(folder / "pages"),
        "output": str(folder / "out"),
        "ground_truth": str(folder / "truth.jsonl"),
        "DOC_EXTRACT_LOG_DIR": str(folder / "logs"),
        **changes,
    }
    return resolve(task, environ, RUN_CONFIG)


class Called:
    """Stands in for a stage's ``run``: notes what it was given."""

    def __init__(self) -> None:
        self.args: tuple[Any, ...] = ()
        self.kwargs: dict[str, Any] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.args, self.kwargs = args, kwargs


@pytest.fixture
def stage(monkeypatch: pytest.MonkeyPatch) -> Called:
    called = Called()
    monkeypatch.setattr(starter, "_stage_run", lambda task: called)
    return called


def test_classify_is_given_its_folders_and_says_how_long_it_took(
    tmp_path: Path, stage: Called, caplog
) -> None:
    settings = settings_for("classify", tmp_path)

    with caplog.at_level(logging.INFO):
        starter.run_task(settings)

    assert stage.args == (settings.image_dir, settings.classifications)
    assert stage.kwargs == {"model_type": "gemma4-31b-w4a16-vllm"}
    assert any("Classification complete (" in record.getMessage() for record in caplog.records)
    # The runner keeps no time of its own: a stage notes what it spent (tests/common/test_stage_timing.py).
    assert not (settings.output / ".inference_elapsed").exists()


def test_extract_is_given_its_folders(tmp_path: Path, stage: Called) -> None:
    settings = settings_for("extract", tmp_path)

    starter.run_task(settings)

    assert stage.args == (settings.classifications, settings.image_dir, settings.raw_extractions)
    assert stage.kwargs == {"model_type": "gemma4-31b-w4a16-vllm"}


def test_evaluate_is_given_its_folders_and_no_time(tmp_path: Path, stage: Called) -> None:
    """It reads the time each stage noted from the output folder itself."""
    settings = settings_for("evaluate", tmp_path)

    starter.run_task(settings)

    assert stage.args == (settings.cleaned_extractions, settings.ground_truth, settings.evaluation_dir)
    assert stage.kwargs == {}


def test_the_other_tasks_are_given_their_folders(tmp_path: Path, stage: Called) -> None:
    starter.run_task(settings_for("split", tmp_path))
    assert stage.args[1] == tmp_path / "out" / "split"

    starter.run_task(settings_for("clean", tmp_path))
    assert stage.args == (
        tmp_path / "out" / "raw_extractions.jsonl",
        tmp_path / "out" / "cleaned_extractions.jsonl",
    )

    starter.run_task(settings_for("sroie", tmp_path, sroie_max_images="40"))
    assert stage.kwargs == {"model_type": "gemma4-31b-w4a16-vllm", "max_images": 40}


def test_a_run_resumes_unless_told_to_start_afresh(tmp_path: Path, stage: Called) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "classifications.jsonl").write_text("{}\n")
    (out / "raw_extractions.jsonl").write_text("{}\n")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "kept.log").write_text("kept")

    starter.run_task(settings_for("classify", tmp_path))
    assert (out / "classifications.jsonl").exists()

    starter.run_task(settings_for("classify", tmp_path, CLEAR_PREV_OUTPUT="true"))
    assert not (out / "classifications.jsonl").exists()
    # Only the stage's own output goes: not another stage's, and never a log.
    assert (out / "raw_extractions.jsonl").exists() and (tmp_path / "logs" / "kept.log").exists()


def test_settings_that_cannot_be_worked_out_stop_the_run_with_the_message(monkeypatch, capsys) -> None:
    monkeypatch.setenv("KFP_TASK", "trust")

    assert starter.main() == 1
    assert_diagnostic(capsys.readouterr().out, "KFP_TASK", *TASKS)
