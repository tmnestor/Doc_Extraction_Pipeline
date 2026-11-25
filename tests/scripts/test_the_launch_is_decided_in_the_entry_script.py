"""What is decided at launch is decided in ``entrypoint.sh``, in bash, and is tried here by running it.

Every job starts with that script, and it is where a reader looks first. So what is settled before a
task is worked out is settled there, in a few lines each:

- a job's variables: that the task has those it needs, and that each value can be used. A variable that
  is unset, empty or the word ``None`` is not set: a pipeline manifest passes a blank parameter as the
  text "None";
- that a variable the pipeline no longer reads is refused by name, and not dropped in silence;
- that nothing is written beside the code: a folder the job names for its output, its log or its
  record of what was sent to the model is refused if it is inside the code's folder;
- which commit of the code this is, read from the checkout's own files, since git is not run once a
  job has started.

Until 26 November 2026 a run's choices were keys of a settings file, each read and checked in Python.

A start that is refused stops before anything is written. A start that gets past the checks writes its
log and then stops for want of the conda environment, unless the test names one: those tests run the
script to its end with the ``clean`` task, which needs no model, and are skipped where the environment
they name is not installed.

Every value written into a file here is invented.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FOUR_PARTS = ("What:", "Where:", "Expected:", "Fix:")
A_CONDA_ENV = Path("/opt/homebrew/Caskroom/miniforge/base/envs/du")
JOB_VARIABLES = (
    "KFP_TASK",
    "model",
    "image_dir",
    "output",
    "ground_truth",
    "document_types",
    "max_images",
    "photo_split",
    "document_type",
    "dataset",
    "run_config",
    "CLEAR_PREV_OUTPUT",
)


def started(
    variables: dict[str, str], conda_env: Path = Path("/nowhere/at/all")
) -> subprocess.CompletedProcess[str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in JOB_VARIABLES and not key.startswith("DOC_EXTRACT_")
    }
    return subprocess.run(
        ["bash", "entrypoint.sh"],
        cwd=REPO,
        env={**environment, "DOC_EXTRACT_CONDA_ENV": str(conda_env), **variables},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def a_clean_job(folder: Path, **more: str) -> dict[str, str]:
    """The variables of a job that cleans: an output folder and a log folder, where a job may write."""
    return {
        "KFP_TASK": "clean",
        "output": str(folder / "out"),
        "DOC_EXTRACT_LOG_DIR": str(folder / "logs"),
        **more,
    }


def refused(result: subprocess.CompletedProcess[str], *said: str) -> None:
    assert result.returncode == 1, result.stdout[-1500:]
    assert "FATAL:" in result.stdout and all(part in result.stdout for part in FOUR_PARTS)
    assert all(word in result.stdout for word in said), result.stdout[-1500:]


def a_reply_to_clean(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "image_name": "a.png",
        "image_path": "/pages/a.png",
        "document_type": "RECEIPT",
        "raw_response": "TOTAL_AMOUNT: $31.07",
        "processing_time": 1.0,
        "prompt_used": "receipt",
        "error": None,
    }
    (output / "raw_extractions.jsonl").write_text(json.dumps(record) + "\n")


needs_the_environment = pytest.mark.skipif(
    not (A_CONDA_ENV / "bin" / "python").exists() or shutil.which("conda") is None,
    reason="runs the entry script to its end, which needs a conda environment installed here",
)


# --- the task, and what it needs ------------------------------------------------------------------------


@pytest.mark.parametrize("task", ["", "None", "run_info_extract", "sroie"])
def test_a_task_that_is_not_one_of_the_five_is_refused(tmp_path: Path, task: str) -> None:
    result = started(a_clean_job(tmp_path, KFP_TASK=task))

    refused(result, "KFP_TASK", "split, classify, extract, clean or evaluate")
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize(
    ("task", "left_out"),
    [
        ("split", "model"),
        ("classify", "image_dir"),
        ("extract", "output"),
        ("clean", "output"),
        ("evaluate", "ground_truth"),
        ("clean", "DOC_EXTRACT_LOG_DIR"),
    ],
)
@pytest.mark.parametrize("blank", [None, "", "None"])
def test_a_task_without_a_variable_it_needs_is_refused_and_told_which(
    tmp_path: Path, task: str, left_out: str, blank: str | None
) -> None:
    (tmp_path / "pages").mkdir()
    job = {
        "KFP_TASK": task,
        "model": "gemma4-31b-w4a16-vllm",
        "image_dir": str(tmp_path / "pages"),
        "output": str(tmp_path / "out"),
        "ground_truth": str(tmp_path / "ground_truth.jsonl"),
        "DOC_EXTRACT_LOG_DIR": str(tmp_path / "logs"),
    }
    if blank is None:
        del job[left_out]
    else:
        job[left_out] = blank

    result = started(job)

    refused(result, f"the task {task} needs {left_out}")
    assert not (tmp_path / "logs").exists()


def test_clean_needs_no_model_no_pages_and_no_answer_key(tmp_path: Path) -> None:
    result = started(a_clean_job(tmp_path))

    assert "FATAL: the task" not in result.stdout
    assert (tmp_path / "logs").is_dir()


def test_a_folder_of_pages_that_is_not_there_is_refused(tmp_path: Path) -> None:
    job = a_clean_job(
        tmp_path, KFP_TASK="classify", model="gemma4-31b-w4a16-vllm", image_dir=str(tmp_path / "mislaid")
    )

    refused(started(job), "the folder of pages does not exist", "mislaid", "image_dir")


@pytest.mark.parametrize("key", ["ground_truth.csv", "ground_truth.json", "ground_truth"])
def test_an_answer_key_that_is_not_json_lines_is_refused_for_evaluate(tmp_path: Path, key: str) -> None:
    job = a_clean_job(tmp_path, KFP_TASK="evaluate", ground_truth=str(tmp_path / key))

    refused(started(job), "is not a JSON lines file", key, "ground_truth.jsonl")


def test_the_format_of_the_answer_key_is_checked_for_evaluate_alone(tmp_path: Path) -> None:
    result = started(a_clean_job(tmp_path, ground_truth=str(tmp_path / "ground_truth.csv")))

    assert "JSON lines" not in result.stdout and (tmp_path / "logs").is_dir()


# --- a run's choices ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("max_images", "0"),
        ("max_images", "-3"),
        ("max_images", "twelve"),
        ("max_images", "2.5"),
        ("document_types", ","),
        ("document_types", " , ,"),
        ("photo_split", "yes"),
        ("photo_split", "1"),
        ("CLEAR_PREV_OUTPUT", "yes"),
        ("DOC_EXTRACT_GPU_MEMORY_UTILIZATION", "lots"),
        ("DOC_EXTRACT_GPU_MEMORY_UTILIZATION", "0"),
        ("DOC_EXTRACT_GPU_MEMORY_UTILIZATION", "1.5"),
        ("DOC_EXTRACT_GPU_MEMORY_UTILIZATION", "-0.2"),
    ],
)
def test_a_value_that_cannot_be_used_is_refused_before_anything_is_written(
    tmp_path: Path, variable: str, value: str
) -> None:
    result = started(a_clean_job(tmp_path, **{variable: value}))

    refused(result, f"FATAL: {variable} ", value)
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("max_images", "12"),
        ("document_types", "RECEIPT, invoice,"),
        ("photo_split", "true"),
        ("photo_split", "False"),
        ("CLEAR_PREV_OUTPUT", "True"),
        ("document_type", "RECEIPT"),
        ("DOC_EXTRACT_GPU_MEMORY_UTILIZATION", "0.85"),
        ("DOC_EXTRACT_GPU_MEMORY_UTILIZATION", "1"),
    ],
)
def test_a_value_that_can_be_used_is_logged_as_the_job_set_it(
    tmp_path: Path, variable: str, value: str
) -> None:
    result = started(a_clean_job(tmp_path, **{variable: value}))

    assert f"FATAL: {variable} " not in result.stdout
    logged = [line for line in result.stdout.splitlines() if f" {variable} " in f"{line} "]
    assert any(line.rstrip().lower().endswith(value.lower()) for line in logged), result.stdout[-1500:]


@pytest.mark.parametrize(
    "variable",
    [
        "max_images",
        "document_types",
        "photo_split",
        "document_type",
        "ground_truth",
        "CLEAR_PREV_OUTPUT",
        "DOC_EXTRACT_TRACE_FILE",
        "DOC_EXTRACT_GPU_MEMORY_UTILIZATION",
    ],
)
@pytest.mark.parametrize("blank", ["", "None"])
def test_a_variable_that_is_empty_or_the_word_none_is_not_set(
    tmp_path: Path, variable: str, blank: str
) -> None:
    result = started(a_clean_job(tmp_path, **{variable: blank}))

    assert f"FATAL: {variable} " not in result.stdout and "would write beside the code" not in result.stdout
    assert (tmp_path / "logs").is_dir()


def test_every_job_variable_is_logged_and_the_two_with_a_default_say_what_it_is(tmp_path: Path) -> None:
    result = started(a_clean_job(tmp_path))

    lines = result.stdout.splitlines()

    def logged(name: str) -> str:
        (line,) = [line for line in lines if f" {name} " in f"{line} " and "FATAL" not in line]
        return line.split(name, 1)[1].strip()

    assert logged("KFP_TASK") == "clean" and logged("output") == str(tmp_path / "out")
    assert logged("photo_split") == "false" and logged("CLEAR_PREV_OUTPUT") == "false"
    assert logged("document_types") == logged("max_images") == logged("model") == "<not set>"


@pytest.mark.parametrize(
    ("gone", "value", "use_instead"),
    [
        ("dataset", "sroie", "document_type"),
        ("run_config", "/data/run_20261126/run_config.yml", "the variables listed at the top"),
    ],
)
def test_a_variable_the_pipeline_no_longer_reads_is_refused_by_name(
    tmp_path: Path, gone: str, value: str, use_instead: str
) -> None:
    result = started(a_clean_job(tmp_path, **{gone: value}))

    refused(result, f"FATAL: {gone} ", "no longer read", use_instead)
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize("gone", ["dataset", "run_config"])
def test_one_of_those_left_blank_by_a_manifest_is_not_refused(tmp_path: Path, gone: str) -> None:
    result = started(a_clean_job(tmp_path, **{gone: "None"}))

    assert "no longer read" not in result.stdout and (tmp_path / "logs").is_dir()


# --- nothing beside the code, and no git ----------------------------------------------------------------


@pytest.mark.parametrize("which", ["output", "DOC_EXTRACT_LOG_DIR", "DOC_EXTRACT_TRACE_FILE"])
def test_a_place_named_beside_the_code_stops_the_job_before_it_is_made(tmp_path: Path, which: str) -> None:
    beside = REPO / "written_by_a_job"

    result = started(a_clean_job(tmp_path, **{which: str(beside)}))

    refused(result, "FATAL: the job would write beside the code", str(beside))
    assert not beside.exists() and not (tmp_path / "logs").exists()


def test_the_script_runs_git_and_python_nowhere_before_the_task() -> None:
    """git is not run once a job has started, and no Python decides anything before the task's own."""
    code = [
        line
        for line in (REPO / "entrypoint.sh").read_text().splitlines()
        if not line.lstrip().startswith("#")
    ]

    assert not any(" git " in f" {line} " for line in code)
    assert [line for line in code if "python" in line and "log " not in line] == [
        'python3 -m stages "$CONFIG_FILE"'
    ]


# --- the script run to its end, with the clean task -----------------------------------------------------


@needs_the_environment
def test_a_job_is_run_from_its_variables_alone_and_the_code_is_not_touched(tmp_path: Path) -> None:
    a_reply_to_clean(tmp_path / "out")
    before = subprocess.run(
        ["git", "status", "--short"], cwd=REPO, capture_output=True, text=True, check=True
    )

    result = started(a_clean_job(tmp_path, max_images="None", document_type=""), A_CONDA_ENV)

    assert result.returncode == 0, result.stdout[-2000:]
    (log,) = (tmp_path / "logs").glob("entrypoint_*.log")
    assert "Exited with code 0" in log.read_text()
    assert len((tmp_path / "out" / "cleaned_extractions.jsonl").read_text().splitlines()) == 1
    # The commit is the checkout's, read without git: the test asks git, to check the reading.
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout[:7]
    assert f"Commit:   {commit}" in result.stdout
    after = subprocess.run(
        ["git", "status", "--short"], cwd=REPO, capture_output=True, text=True, check=True
    )
    assert after.stdout == before.stdout
