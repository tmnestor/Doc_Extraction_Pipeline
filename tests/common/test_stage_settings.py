"""A task's settings are the job's variables, read once into one record.

``entrypoint.sh`` checks them, in bash: that the task has those it needs and that each value can be used
(``tests/scripts/test_the_launch_is_decided_in_the_entry_script.py``). ``common.stage_settings.resolve``
reads them as the script left them, and ``python -m stages`` runs the task with the result. Until
26 November 2026 each was looked for in the environment and then in a settings file, and checked here.
"""

import importlib
import inspect
import logging
import re
from pathlib import Path
from typing import Any

import pytest

from common.stage_settings import TASKS, StageSettings, resolve, run_settings
from stages import __main__ as starter

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_CONFIG = REPO_ROOT / "config" / "run_config.yml"

# What a sandbox command file sets for a stage.
D = "/home/jovyan/nfs_share/tod_2026/evaluation_data"
USUAL = {
    "model": "gemma4-31b-w4a16-vllm",
    "image_dir": f"{D}/snapshot_20261021/plain_20261021",
    "output": f"{D}/output_run",
    "DOC_EXTRACT_LOG_DIR": f"{D}/output_run/logs",
    # The two the entry script always writes out.
    "photo_split": "false",
    "CLEAR_PREV_OUTPUT": "false",
}


def test_the_five_tasks() -> None:
    assert TASKS == ("split", "classify", "extract", "clean", "evaluate")


@pytest.mark.parametrize("task", ["split", "classify", "extract", "clean"])
def test_the_usual_variables_give_the_task_its_folders(task: str) -> None:
    settings = resolve(task, USUAL, RUN_CONFIG)

    assert settings.task == task and settings.model == "gemma4-31b-w4a16-vllm"
    assert settings.image_dir == Path(f"{D}/snapshot_20261021/plain_20261021")
    assert settings.split_dir == Path(f"{D}/output_run/split")
    assert settings.classifications == Path(f"{D}/output_run/classifications.jsonl")
    assert settings.raw_extractions == Path(f"{D}/output_run/raw_extractions.jsonl")
    assert settings.cleaned_extractions == Path(f"{D}/output_run/cleaned_extractions.jsonl")
    assert settings.evaluation_dir == Path(f"{D}/output_run/evaluation")


def test_a_job_that_makes_no_choice_reads_every_page_and_asks_the_model() -> None:
    settings = resolve("classify", USUAL, RUN_CONFIG)

    assert settings.document_types is None and settings.max_images is None
    assert settings.photo_split is False and settings.clear_previous_output is False
    assert settings.given_type is None and settings.trace_file is None
    assert settings.gpu_memory_utilization is None and settings.ground_truth is None


def test_each_choice_of_a_run_is_read_from_its_variable() -> None:
    job = {
        **USUAL,
        "document_types": "receipt",
        "max_images": "12",
        "photo_split": "true",
        "CLEAR_PREV_OUTPUT": "true",
        "document_type": "SROIE_RECEIPT",
        "DOC_EXTRACT_TRACE_FILE": "/data/run/sent.jsonl",
        "DOC_EXTRACT_GPU_MEMORY_UTILIZATION": "0.85",
        "ground_truth": "/data/pages/ground_truth.jsonl",
    }

    settings = resolve("split", job, RUN_CONFIG)

    assert settings.document_types == ("receipt",) and settings.max_images == 12
    assert settings.photo_split is True and settings.clear_previous_output is True
    assert settings.given_type == "SROIE_RECEIPT"
    assert settings.trace_file == Path("/data/run/sent.jsonl")
    assert settings.gpu_memory_utilization == 0.85
    assert settings.ground_truth == Path("/data/pages/ground_truth.jsonl")


@pytest.mark.parametrize(
    ("written", "selects"),
    [
        ("RECEIPT, invoice,", ("RECEIPT", "invoice")),
        ("receipt", ("receipt",)),
        (" bank_statement ,receipt", ("bank_statement", "receipt")),
    ],
)
def test_document_types_written_loosely_select_the_types_named(
    written: str, selects: tuple[str, ...]
) -> None:
    assert resolve("classify", {**USUAL, "document_types": written}, RUN_CONFIG).document_types == selects


def test_clean_and_evaluate_are_given_no_model_and_no_pages() -> None:
    settings = resolve("clean", {"output": "/data/run"}, RUN_CONFIG)

    assert settings.model is None and settings.image_dir is None


def test_the_choices_reach_the_settings_the_engine_is_started_with(tmp_path: Path) -> None:
    """The pages selected, and the file that what is sent to the model is kept in."""
    import yaml

    raw = yaml.safe_load(RUN_CONFIG.read_text())
    (tmp_path / "config.json").write_text("{}")
    raw["bootstrap"]["model"]["default_paths"] = dict.fromkeys(
        raw["bootstrap"]["model"]["default_paths"], str(tmp_path)
    )
    config = tmp_path / "run_config.yml"
    config.write_text(yaml.safe_dump(raw))
    job = {
        **USUAL,
        "document_types": "receipt,invoice",
        "max_images": "3",
        "DOC_EXTRACT_TRACE_FILE": str(tmp_path / "sent.jsonl"),
        "DOC_EXTRACT_GPU_MEMORY_UTILIZATION": "0.85",
    }

    settings = run_settings(resolve("classify", job, config))

    assert settings.document_types == ("receipt", "invoice") and settings.max_images == 3
    assert settings.trace_file == tmp_path / "sent.jsonl" and settings.gpu_memory_utilization == 0.85


def test_the_variable_that_cut_the_benchmark_short_is_no_longer_read() -> None:
    """How many pages are read is max_images, for every run."""
    settings = resolve("classify", {**USUAL, "sroie_max_images": "forty"}, RUN_CONFIG)

    assert settings.max_images is None and not hasattr(settings, "sroie_max_images")


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
    assert stage.args == (settings.classifications, CHECKED)
    # The job gives the pages no type, so the model is asked.
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

    assert stage.args == (CHECKED, settings.evaluation_dir)
    assert stage.kwargs == {}


def test_the_other_tasks_are_given_their_folders(tmp_path: Path, stage: Called) -> None:
    starter.run_task(settings_for("split", tmp_path))
    assert stage.args == (tmp_path / "out" / "split", CHECKED)

    starter.run_task(settings_for("clean", tmp_path))
    assert stage.args == (CHECKED, tmp_path / "out" / "cleaned_extractions.jsonl")


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


def test_classify_is_handed_the_type_the_job_gives_every_page(tmp_path: Path, stage: Called) -> None:
    settings = settings_for("classify", tmp_path, document_type="SROIE_RECEIPT")

    starter.run_task(settings)

    assert stage.kwargs["given_type"] == "SROIE_RECEIPT"
