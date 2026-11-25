"""A task's settings come from the environment first and the settings file second, and are worked out once.

Until 21 November 2026 the entry script worked these out in shell: it read six values from the settings file
with a helper, laid the environment over them, and built each stage's flags by hand. Now
`common.stage_settings.resolve` does it, and `python -m stages` runs the task with the result.
"""

import importlib
import inspect
import logging
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from common.stage_settings import (
    TASKS,
    ConfigError,
    StageSettings,
    check_folder_of_pages,
    log_dir,
    resolve,
)
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


def test_the_five_tasks() -> None:
    assert TASKS == ("split", "classify", "extract", "clean", "evaluate")


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
    assert settings.clear_previous_output is False


IN_THE_FILE = {
    "pages": "/data/trial/pages",
    "key": "/data/trial/pages/ground_truth.jsonl",
    "output": "/data/trial/output",
    "logs": "/data/trial/output/logs",
}


def with_folders(raw: dict[str, Any]) -> None:
    """A settings file that names its own folders, as one kept for trials on one machine may."""
    raw["pipeline"]["information_extraction"]["input"].update(
        dir=IN_THE_FILE["pages"], ground_truth=IN_THE_FILE["key"]
    )
    raw["pipeline"]["information_extraction"]["output"]["dir"] = IN_THE_FILE["output"]
    raw["bootstrap"]["logging"]["log_dir"] = IN_THE_FILE["logs"]


def test_the_environment_wins_over_the_settings_file(tmp_path: Path) -> None:
    settings = resolve("classify", USUAL, config_with(tmp_path, with_folders))

    assert str(settings.image_dir) == USUAL["image_dir"] != IN_THE_FILE["pages"]
    assert str(settings.output) == USUAL["output"] != IN_THE_FILE["output"]
    assert str(settings.log_dir) == USUAL["DOC_EXTRACT_LOG_DIR"] != IN_THE_FILE["logs"]


@pytest.mark.parametrize("unset", [None, "", "None"])
def test_the_settings_file_is_used_when_a_variable_is_unset_empty_or_none(
    tmp_path: Path, unset: str | None
) -> None:
    environ = {name: unset for name in USUAL if unset is not None}

    settings = resolve("evaluate", environ, config_with(tmp_path, with_folders))

    assert settings.model == shipped("bootstrap.model.type")
    assert str(settings.image_dir) == IN_THE_FILE["pages"]
    assert str(settings.output) == IN_THE_FILE["output"]
    assert str(settings.ground_truth) == IN_THE_FILE["key"]
    assert str(settings.log_dir) == IN_THE_FILE["logs"]


# --- the shipped file names no machine's folders ------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "bootstrap.logging.log_dir",
        "pipeline.information_extraction.input.dir",
        "pipeline.information_extraction.input.ground_truth",
        "pipeline.information_extraction.output.dir",
    ],
)
def test_the_shipped_file_names_no_folder_for_a_run_to_fall_back_on(key: str) -> None:
    """A folder named there would be one machine's, and a run that forgot its own would use it."""
    assert shipped(key) is None


@pytest.mark.parametrize(
    ("task", "left_out", "variable", "key"),
    [
        ("classify", "image_dir", "image_dir", "pipeline.information_extraction.input.dir"),
        ("classify", "output", "output", "pipeline.information_extraction.output.dir"),
        ("clean", "output", "output", "pipeline.information_extraction.output.dir"),
        ("evaluate", None, "ground_truth", "pipeline.information_extraction.input.ground_truth"),
        ("clean", "DOC_EXTRACT_LOG_DIR", "DOC_EXTRACT_LOG_DIR", "bootstrap.logging.log_dir"),
    ],
)
def test_a_job_that_does_not_name_a_folder_its_task_needs_is_stopped_and_told_what_to_set(
    task: str, left_out: str | None, variable: str, key: str
) -> None:
    environ = {name: value for name, value in USUAL.items() if name != left_out}

    with pytest.raises(ConfigError) as raised:
        resolve(task, environ, RUN_CONFIG)

    assert_diagnostic(str(raised.value), variable, key, str(RUN_CONFIG.resolve()))


@pytest.mark.parametrize("task", ["split", "classify", "extract"])
def test_a_task_that_reads_pages_is_not_started_without_its_folder_of_pages(
    tmp_path: Path, task: str
) -> None:
    environ = {**USUAL, "image_dir": str(tmp_path / "not_there")}
    settings = resolve(task, environ, RUN_CONFIG)

    with pytest.raises(ConfigError) as raised:
        check_folder_of_pages(settings, RUN_CONFIG)

    assert_diagnostic(
        str(raised.value),
        "does not exist",
        str(tmp_path / "not_there"),
        "image_dir",
        "pipeline.information_extraction.input.dir",
        str(RUN_CONFIG.resolve()),
    )


@pytest.mark.parametrize("task", ["clean", "evaluate"])
def test_a_task_that_reads_no_pages_is_not_asked_whether_a_folder_of_pages_is_there(task: str) -> None:
    environ = {**USUAL, "image_dir": "/nowhere/at/all", "ground_truth": "/data/key.jsonl"}

    check_folder_of_pages(resolve(task, environ, RUN_CONFIG), RUN_CONFIG)


def test_a_folder_of_pages_that_is_there_passes(tmp_path: Path) -> None:
    check_folder_of_pages(
        resolve("classify", {**USUAL, "image_dir": str(tmp_path)}, RUN_CONFIG), RUN_CONFIG
    )


@pytest.mark.parametrize("task", ["", "run_info_extract", "trust", "sroie"])
def test_a_task_that_is_not_one_of_the_five_is_refused(task: str) -> None:
    """The benchmark that had a task of its own until 25 November 2026 is read as a dataset now."""
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


def test_the_images_are_not_needed_to_clean(tmp_path: Path) -> None:
    path = config_with(
        tmp_path, lambda raw: raw["pipeline"]["information_extraction"]["input"].update(dir=None)
    )
    environ = {k: v for k, v in USUAL.items() if k != "image_dir"}

    assert resolve("clean", environ, path).image_dir is None


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


def test_the_variable_that_cut_the_benchmark_short_is_no_longer_read() -> None:
    """How many pages are read is pipeline.information_extraction.input.max_images, for every run."""
    settings = resolve("classify", {**USUAL, "sroie_max_images": "forty"}, RUN_CONFIG)

    assert not hasattr(settings, "sroie_max_images")


def test_a_settings_file_that_still_has_the_old_tasks_block_is_refused(tmp_path: Path) -> None:
    path = config_with(
        tmp_path, lambda raw: raw["pipeline"].update(sroie={"data_dir": "/data/sroie", "batch_size": 8})
    )

    with pytest.raises(ConfigError) as raised:
        resolve("classify", USUAL, path)

    assert_diagnostic(str(raised.value), "pipeline.sroie.data_dir", "pipeline.sroie.batch_size")


def test_a_variable_under_the_old_prefix_is_refused() -> None:
    with pytest.raises(ConfigError) as raised:
        resolve("classify", {**USUAL, OLD_PREFIX + "LOG_DIR": "/somewhere"}, RUN_CONFIG)

    assert_diagnostic(str(raised.value), OLD_PREFIX + "LOG_DIR", "DOC_EXTRACT_LOG_DIR")


def test_the_log_folder_alone_can_be_asked_for() -> None:
    """The entry script needs it before anything else, to keep the whole run's output."""
    assert log_dir(USUAL, RUN_CONFIG) == Path(f"{D}/output_run/logs")


def test_the_log_folder_comes_from_the_settings_file_when_the_job_names_none(tmp_path: Path) -> None:
    assert str(log_dir({}, config_with(tmp_path, with_folders))) == IN_THE_FILE["logs"]


def test_a_job_with_no_log_folder_anywhere_is_stopped_before_anything_runs() -> None:
    with pytest.raises(ConfigError) as raised:
        log_dir({}, RUN_CONFIG)

    assert_diagnostic(str(raised.value), "DOC_EXTRACT_LOG_DIR", "bootstrap.logging.log_dir")


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


CHECKED = object()


class Called:
    """Stands in for a stage's ``run``: notes what it was given."""

    def __init__(self) -> None:
        self.args: tuple[Any, ...] = ()
        self.kwargs: dict[str, Any] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.args, self.kwargs = args, kwargs

    def check(self, task: StageSettings) -> Any:
        """Stands in for a stage's ``check``: notes the task it was given, and gives a mark that a
        card stage's ``run`` must then be handed."""
        self.checked = task
        return CHECKED


@pytest.fixture
def stage(monkeypatch: pytest.MonkeyPatch) -> Called:
    """Every stage's own ``run`` replaced, so that each stage's ``start`` is what is being tried."""
    called = Called()
    for task in TASKS:
        monkeypatch.setattr(importlib.import_module(f"stages.{task}"), "run", called)
        monkeypatch.setattr(importlib.import_module(f"stages.{task}"), "check", called.check)
    return called


def test_classify_is_given_its_folders_and_says_how_long_it_took(
    tmp_path: Path, stage: Called, caplog
) -> None:
    settings = settings_for("classify", tmp_path)

    with caplog.at_level(logging.INFO):
        starter.run_task(settings)

    # What the stage's check gave is what its run is handed: nothing is worked out a second time.
    assert stage.args == (settings.image_dir, settings.classifications, CHECKED)
    # No dataset is named, so no type is given: the model is asked.
    assert stage.kwargs == {"given_type": None}
    assert any("Classification complete (" in record.getMessage() for record in caplog.records)
    # The runner keeps no time of its own: a stage notes what it spent (tests/common/test_stage_timing.py).
    assert not (settings.output / ".inference_elapsed").exists()


def test_extract_is_given_its_folders(tmp_path: Path, stage: Called) -> None:
    settings = settings_for("extract", tmp_path)

    starter.run_task(settings)

    assert stage.args == (settings.raw_extractions, CHECKED)
    assert stage.kwargs == {}


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


def test_the_settings_file_a_task_was_worked_out_from_is_the_one_its_stage_reads(
    tmp_path: Path, stage: Called
) -> None:
    """Until 25 November 2026 a stage read the shipped file, whatever file the task was worked out from."""
    another = config_with(tmp_path, with_folders)
    environ = {
        "model": "gemma4-31b-w4a16-vllm",
        "image_dir": str(tmp_path),
        "output": str(tmp_path / "out"),
    }
    settings = resolve("classify", environ, another)

    starter.run_task(settings)

    assert settings.config_path == another != RUN_CONFIG
    # The stage's check is given the task, which holds that file and what it held when it was read.
    assert stage.checked.config_path == another
    assert stage.checked.raw == yaml.safe_load(another.read_text())


@pytest.mark.parametrize("task", TASKS)
def test_every_stage_has_a_banner_and_is_started_the_one_way(task: str) -> None:
    """The runner knows no stage by name: it asks each for these three, and nothing else."""
    module = importlib.import_module(f"stages.{task}")
    parameters = list(inspect.signature(module.start).parameters)

    assert isinstance(module.BANNER, str) and task in module.BANNER.lower()
    assert parameters == ["task", "checked"]
    assert list(inspect.signature(module.check).parameters) == ["task"]
    assert inspect.signature(module.start).return_annotation is str


def test_the_runner_names_no_stage_but_where_a_stage_differs_in_what_it_carries_on_from() -> None:
    source = (REPO_ROOT / "stages" / "__main__.py").read_text()
    code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
    code = code.split("def _stage(", 1)[1]

    named = sorted({task for task in TASKS if re.search(rf"[\"\x27]{task}[\"\x27]", code)})
    # The split stage always checks what made its folder, and extract warns if classify was made
    # another way: both are about the note a stage leaves, not about how a stage is run.
    assert named == ["classify", "extract", "split"]
    assert "elif" not in code


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


# --- a dataset whose pages are all of one type ------------------------------------------------------------


def with_datasets(tmp_path: Path, datasets: Any) -> Path:
    def change(raw: dict[str, Any]) -> None:
        if datasets is None:
            del raw["pipeline"]["datasets"]
        else:
            raw["pipeline"]["datasets"] = datasets

    return config_with(tmp_path, change)


def test_the_shipped_file_has_the_sroie_receipts_as_a_dataset() -> None:
    assert shipped("pipeline.datasets") == {
        "sroie": {
            "document_type": "SROIE_RECEIPT",
            "image_dir": "/home/jovyan/nfs_share/tod_2026/data/sroie/test/img",
        }
    }


def test_a_job_that_names_a_dataset_is_given_its_type_and_its_folder_of_pages() -> None:
    environ = {k: v for k, v in USUAL.items() if k != "image_dir"} | {"dataset": "sroie"}

    settings = resolve("classify", environ, RUN_CONFIG)

    assert settings.dataset == "sroie" and settings.given_type == "SROIE_RECEIPT"
    assert settings.image_dir == Path("/home/jovyan/nfs_share/tod_2026/data/sroie/test/img")


def test_the_job_s_own_folder_of_pages_stands_in_place_of_the_dataset_s() -> None:
    settings = resolve("classify", USUAL | {"dataset": "sroie"}, RUN_CONFIG)

    assert settings.given_type == "SROIE_RECEIPT" and settings.image_dir == Path(USUAL["image_dir"])


def test_a_dataset_with_no_folder_of_its_own_leaves_the_job_to_name_one(tmp_path: Path) -> None:
    config = with_datasets(tmp_path, {"kept": {"document_type": "receipt", "image_dir": None}})

    assert resolve("classify", USUAL | {"dataset": "kept"}, config).given_type == "receipt"
    with pytest.raises(ConfigError) as refused:
        resolve(
            "classify", {k: v for k, v in USUAL.items() if k != "image_dir"} | {"dataset": "kept"}, config
        )
    assert_diagnostic(str(refused.value), "image_dir")


@pytest.mark.parametrize("unset", [None, "", "None"])
def test_a_job_that_names_no_dataset_is_as_it_always_was(unset: str | None) -> None:
    environ = USUAL if unset is None else USUAL | {"dataset": unset}

    settings = resolve("classify", environ, RUN_CONFIG)

    assert settings.dataset is None and settings.given_type is None
    assert settings.image_dir == Path(USUAL["image_dir"])


def test_a_dataset_that_is_not_in_the_file_is_refused_with_those_that_are() -> None:
    with pytest.raises(ConfigError) as refused:
        resolve("classify", USUAL | {"dataset": "wildreceipt"}, RUN_CONFIG)

    assert_diagnostic(str(refused.value), "wildreceipt", "sroie", "pipeline.datasets", str(RUN_CONFIG))


def test_a_dataset_of_a_type_the_schema_does_not_have_is_refused_with_the_types_there_are(
    tmp_path: Path,
) -> None:
    config = with_datasets(tmp_path, {"odd": {"document_type": "PERMIT", "image_dir": None}})

    with pytest.raises(ConfigError) as refused:
        resolve("clean", USUAL, config)

    assert_diagnostic(str(refused.value), "PERMIT", "pipeline.datasets.odd.document_type", "sroie_receipt")


@pytest.mark.parametrize(
    ("block", "said"),
    [
        ({"document_type": "receipt"}, "image_dir"),
        ({"image_dir": None}, "document_type"),
        ({"document_type": "receipt", "image_dir": 7}, "image_dir"),
        ("receipt", "pipeline.datasets.odd"),
    ],
)
def test_a_dataset_that_is_not_its_two_keys_is_refused(tmp_path: Path, block: Any, said: str) -> None:
    """Checked for every dataset whatever the job names, so that a fault is found before it is needed."""
    config = with_datasets(tmp_path, {"odd": block})

    with pytest.raises(ConfigError) as refused:
        resolve("clean", USUAL, config)

    assert_diagnostic(str(refused.value), said, "pipeline.datasets.odd")


def test_a_dataset_with_a_key_the_pipeline_does_not_know_is_refused(tmp_path: Path) -> None:
    config = with_datasets(
        tmp_path, {"odd": {"document_type": "receipt", "image_dir": None, "scoring": "strict"}}
    )

    with pytest.raises(ConfigError) as refused:
        resolve("clean", USUAL, config)

    assert_diagnostic(str(refused.value), "pipeline.datasets.odd.scoring")


def test_a_settings_file_with_no_datasets_block_is_refused_and_one_with_none_in_it_is_not(
    tmp_path: Path,
) -> None:
    with pytest.raises(ConfigError) as refused:
        resolve("clean", USUAL, with_datasets(tmp_path, None))
    assert_diagnostic(str(refused.value), "pipeline.datasets", "datasets: {}")

    assert resolve("clean", USUAL, with_datasets(tmp_path, {})).given_type is None


def test_classify_is_handed_the_type_its_dataset_gives(tmp_path: Path, stage: Called) -> None:
    settings = settings_for("classify", tmp_path, dataset="sroie")

    starter.run_task(settings)

    assert stage.kwargs["given_type"] == "SROIE_RECEIPT"
