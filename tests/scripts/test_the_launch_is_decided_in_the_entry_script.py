"""What is decided at launch is decided in ``entrypoint.sh``, in bash, and is tried here by running it.

Every job starts with that script, and it is where a reader looks first. So the three things that are
settled before a task is worked out are settled there, in a few lines each:

- which settings file is read: the shipped one, or the job's own, named in ``run_config``;
- that nothing is written beside the code: Python is told to write no compiled copies there, and a
  folder the job names for its output or its log is refused if it is inside the code's folder;
- which commit of the code this is, read from the checkout's own files, since git is not run once a
  job has started.

The first two tests stop before the conda environment is looked for, and run anywhere. The rest run the
script to its end with the ``clean`` task, which needs no model, and are skipped where the conda
environment they name is not installed.

Every value written into a file here is invented.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SHIPPED = REPO / "config" / "run_config.yml"
FOUR_PARTS = ("What:", "Where:", "Expected:", "Fix:")
A_CONDA_ENV = Path("/opt/homebrew/Caskroom/miniforge/base/envs/du")
JOB_VARIABLES = ("model", "image_dir", "output", "ground_truth", "dataset", "run_config", "KFP_TASK")


def started(
    variables: dict[str, str], conda_env: Path = Path("/nowhere/at/all")
) -> subprocess.CompletedProcess[str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in JOB_VARIABLES and not key.startswith(("DOC_EXTRACT_", "PYTHONDONTWRITE"))
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


def a_file_of_its_own(folder: Path) -> Path:
    """A copy of the shipped settings file where a job can write, with its own log folder named."""
    raw = yaml.safe_load(SHIPPED.read_text())
    raw["bootstrap"]["logging"]["log_dir"] = str(folder / "logs_named_in_the_file")
    path = folder / "run_config.yml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


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


# --- before the conda environment is looked for ---------------------------------------------------------


def test_a_settings_file_that_is_not_there_stops_the_job(tmp_path: Path) -> None:
    result = started({"KFP_TASK": "clean", "run_config": str(tmp_path / "mislaid.yml")})

    assert result.returncode == 1
    assert "FATAL: the settings file does not exist" in result.stdout and "mislaid.yml" in result.stdout
    assert all(part in result.stdout for part in FOUR_PARTS) and "run_config" in result.stdout


@pytest.mark.parametrize("share", ["lots", "0", "1.5", "-0.2"])
def test_a_share_of_memory_that_is_no_share_stops_the_job(share: str) -> None:
    result = started({"KFP_TASK": "clean", "DOC_EXTRACT_GPU_MEMORY_UTILIZATION": share})

    assert result.returncode == 1
    assert (
        "FATAL: DOC_EXTRACT_GPU_MEMORY_UTILIZATION is not a share" in result.stdout
        and share in result.stdout
    )
    assert all(part in result.stdout for part in FOUR_PARTS)


@pytest.mark.parametrize("unset", ["", "None", "0.85", "1"])
def test_a_share_that_is_one_or_is_not_set_gets_past_that_check(unset: str) -> None:
    """It then stops for want of the conda environment, which is named nowhere here."""
    result = started(
        {"KFP_TASK": "clean", "DOC_EXTRACT_GPU_MEMORY_UTILIZATION": unset, "run_config": "None"}
    )

    assert "is not a share" not in result.stdout and "settings file does not exist" not in result.stdout
    assert "bootstrap interpreter not found" in result.stdout


def test_python_is_told_to_write_no_compiled_code_before_the_first_python_starts() -> None:
    code = [
        line
        for line in (REPO / "entrypoint.sh").read_text().splitlines()
        if not line.lstrip().startswith("#")
    ]
    told = next(i for i, line in enumerate(code) if line.strip() == "export PYTHONDONTWRITEBYTECODE=1")

    assert all("python" not in line.lower() for line in code[:told])


def test_the_script_runs_git_nowhere() -> None:
    code = [
        line
        for line in (REPO / "entrypoint.sh").read_text().splitlines()
        if not line.lstrip().startswith("#")
    ]

    assert not any(" git " in f" {line} " for line in code)


# --- the script run to its end, with the clean task ----------------------------------------------------


@needs_the_environment
def test_a_job_that_names_its_own_settings_file_is_run_with_it_and_the_code_is_not_touched(
    tmp_path: Path,
) -> None:
    named = a_file_of_its_own(tmp_path)
    a_reply_to_clean(tmp_path / "out")
    before = subprocess.run(
        ["git", "status", "--short"], cwd=REPO, capture_output=True, text=True, check=True
    )

    result = started(
        {"KFP_TASK": "clean", "run_config": str(named), "output": str(tmp_path / "out")}, A_CONDA_ENV
    )

    assert result.returncode == 0, result.stdout[-2000:]
    # The log folder is the one the named file gives, and the task was worked out from that file.
    logs = tmp_path / "logs_named_in_the_file"
    assert f"Settings: {named}" in result.stdout and f"settings file:         {named}" in result.stdout
    (log,) = logs.glob("entrypoint_*.log")
    # The settings the start was run with are kept beside its log.
    assert (logs / f"{log.stem}_run_config.yml").read_text() == named.read_text()
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
    assert not list(REPO.glob("common/__pycache__/stage_settings.*.pyc.*"))


@needs_the_environment
@pytest.mark.parametrize("which", ["output", "DOC_EXTRACT_LOG_DIR"])
def test_a_folder_named_beside_the_code_stops_the_job_before_it_is_made(tmp_path: Path, which: str) -> None:
    named = a_file_of_its_own(tmp_path)
    beside = REPO / "written_by_a_job"
    job = {
        "KFP_TASK": "clean",
        "run_config": str(named),
        "output": str(tmp_path / "out"),
        which: str(beside),
    }

    result = started(job, A_CONDA_ENV)

    assert result.returncode == 1
    assert "FATAL: the job would write beside the code" in result.stdout and str(beside) in result.stdout
    assert all(part in result.stdout for part in FOUR_PARTS)
    assert not beside.exists() and not (tmp_path / "logs_named_in_the_file").exists()
