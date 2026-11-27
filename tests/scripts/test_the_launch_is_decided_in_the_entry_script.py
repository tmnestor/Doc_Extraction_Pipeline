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
import re
import shutil
import subprocess
import tempfile
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
    "preprocess_photographs",
    "document_type",
    "dataset",
    "run_config",
    "photo_split",
    "CLEAR_PREV_OUTPUT",
)


def a_folder_with_a_python(folder: Path) -> Path:
    """A folder that passes for a conda environment as far as the script looks: it holds bin/python.

    The script gets past its check of the environment, opens its log, and stops when conda will not
    activate the folder.
    """
    (folder / "bin").mkdir(parents=True)
    (folder / "bin" / "python").write_text("#!/bin/sh\nexit 1\n")
    (folder / "bin" / "python").chmod(0o755)
    return folder


NO_REAL_ENVIRONMENT = a_folder_with_a_python(Path(tempfile.mkdtemp(prefix="doc_extract_no_env_")) / "env")


def started(
    variables: dict[str, str], conda_env: Path = NO_REAL_ENVIRONMENT, cwd: Path = REPO
) -> subprocess.CompletedProcess[str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in JOB_VARIABLES and not key.startswith("DOC_EXTRACT_")
    }
    return subprocess.run(
        ["bash", str(REPO / "entrypoint.sh")],
        cwd=cwd,
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

    refused(result, "KFP_TASK", "preprocess, classify, extract, clean or evaluate")
    assert not (tmp_path / "logs").exists()


def test_the_name_the_first_task_had_is_refused_with_the_name_it_has_now(tmp_path: Path) -> None:
    """Until 27 November 2026 the task was split. A job written before then is told what to write now,
    and where the task's images are, and is not told only that its task is unknown."""
    result = started(a_clean_job(tmp_path, KFP_TASK="split"))

    refused(
        result,
        "FATAL: KFP_TASK=split is no longer a task: it is now preprocess",
        "KFP_TASK=preprocess",
        "preprocess_photographs",
        "<output>/preprocess/images",
    )
    assert "is not a task of this pipeline" not in result.stdout
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize(
    ("task", "left_out"),
    [
        ("preprocess", "model"),
        ("classify", "image_dir"),
        ("extract", "output"),
        ("clean", "output"),
        ("evaluate", "ground_truth"),
        ("clean", "DOC_EXTRACT_LOG_DIR"),
        ("preprocess", "DOC_EXTRACT_MODELS_DIR"),
        ("classify", "DOC_EXTRACT_MODELS_DIR"),
        ("extract", "DOC_EXTRACT_MODELS_DIR"),
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
        "DOC_EXTRACT_MODELS_DIR": str(tmp_path / "models"),
        # The switch is on, so that the preprocess task loads a model and needs its weights.
        "preprocess_photographs": "true",
    }
    if blank is None:
        del job[left_out]
    else:
        job[left_out] = blank

    result = started(job)

    # The example in the message is of the variable that is missing, and not of another.
    refused(result, f"the task {task} needs {left_out}", f"e.g. {left_out}=")
    assert not (tmp_path / "logs").exists()


def test_clean_needs_no_model_no_weights_no_pages_and_no_answer_key(tmp_path: Path) -> None:
    result = started(a_clean_job(tmp_path))

    assert "FATAL: the task" not in result.stdout
    assert (tmp_path / "logs").is_dir()


MODEL = "gemma4-31b-w4a16-vllm"


@pytest.mark.parametrize(
    "job",
    [
        {"KFP_TASK": "preprocess"},
        {"KFP_TASK": "preprocess", "preprocess_photographs": "false"},
        {"KFP_TASK": "classify", "document_type": "RECEIPT"},
    ],
    ids=["preprocess passing through", "preprocess told not to ask", "classify given the type"],
)
def test_a_task_that_loads_no_model_needs_no_folder_of_weights(tmp_path: Path, job: dict[str, str]) -> None:
    """Until 26 November 2026 the three tasks that can load a model were asked for it whether or not
    this start would load one."""
    (tmp_path / "pages").mkdir()

    result = started({**a_clean_job(tmp_path), "model": MODEL, "image_dir": str(tmp_path / "pages"), **job})

    assert "FATAL: the task" not in result.stdout and (tmp_path / "logs").is_dir()


@pytest.mark.parametrize(
    "job",
    [
        {"KFP_TASK": "preprocess", "preprocess_photographs": "True"},
        {"KFP_TASK": "classify"},
        {"KFP_TASK": "classify", "document_type": "None"},
        {"KFP_TASK": "extract", "document_type": "RECEIPT"},
    ],
    ids=["preprocess asking the model", "classify", "classify with the type left blank", "extract"],
)
def test_a_task_that_loads_a_model_needs_the_folder_of_weights(tmp_path: Path, job: dict[str, str]) -> None:
    (tmp_path / "pages").mkdir()

    result = started({**a_clean_job(tmp_path), "model": MODEL, "image_dir": str(tmp_path / "pages"), **job})

    refused(result, f"the task {job['KFP_TASK']} needs DOC_EXTRACT_MODELS_DIR")
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize("which", ["output", "DOC_EXTRACT_LOG_DIR"])
@pytest.mark.parametrize("under", ["", "inside"], ids=["a file", "under a file"])
def test_a_folder_the_job_writes_to_that_cannot_be_a_folder_is_refused(
    tmp_path: Path, which: str, under: str
) -> None:
    """Until 26 November 2026 an output that named a file ended the task in a traceback."""
    (tmp_path / "a_file").write_text("not a folder")
    named = tmp_path / "a_file" / under if under else tmp_path / "a_file"

    result = started(a_clean_job(tmp_path, **{which: str(named)}))

    refused(result, f"FATAL: {which} is not a folder that can be written in", "a_file")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["a_file"]


def test_a_folder_of_pages_that_is_not_there_is_refused(tmp_path: Path) -> None:
    job = a_clean_job(
        tmp_path,
        KFP_TASK="classify",
        model="gemma4-31b-w4a16-vllm",
        image_dir=str(tmp_path / "mislaid"),
        DOC_EXTRACT_MODELS_DIR=str(tmp_path / "models"),
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
        ("preprocess_photographs", "yes"),
        ("preprocess_photographs", "1"),
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
        ("preprocess_photographs", "true"),
        ("preprocess_photographs", "False"),
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
        "preprocess_photographs",
        "document_type",
        "ground_truth",
        "CLEAR_PREV_OUTPUT",
        "DOC_EXTRACT_TRACE_FILE",
        "DOC_EXTRACT_GPU_MEMORY_UTILIZATION",
        "DOC_EXTRACT_MODELS_DIR",
    ],
)
@pytest.mark.parametrize("blank", ["", "None"])
def test_a_variable_that_is_empty_or_the_word_none_is_not_set(
    tmp_path: Path, variable: str, blank: str
) -> None:
    result = started(a_clean_job(tmp_path, **{variable: blank}))

    assert "FATAL" not in result.stdout.split("What this job set:")[0]
    # The log says what the task will be given: nothing, or for the two with a default, false.
    expected = "false" if variable in ("preprocess_photographs", "CLEAR_PREV_OUTPUT") else "<not set>"
    (line,) = [line for line in result.stdout.splitlines() if f" {variable} " in f"{line} "]
    assert line.rstrip().endswith(expected), line


def test_every_job_variable_is_logged_and_the_two_with_a_default_say_what_it_is(tmp_path: Path) -> None:
    result = started(a_clean_job(tmp_path))

    lines = result.stdout.splitlines()

    def logged(name: str) -> str:
        (line,) = [line for line in lines if f" {name} " in f"{line} " and "FATAL" not in line]
        return line.split(name, 1)[1].strip()

    assert logged("KFP_TASK") == "clean" and logged("output") == str(tmp_path / "out")
    assert logged("preprocess_photographs") == "false" and logged("CLEAR_PREV_OUTPUT") == "false"
    assert logged("document_types") == logged("max_images") == logged("model") == "<not set>"
    assert logged("DOC_EXTRACT_MODELS_DIR") == "<not set>"
    assert logged("DOC_EXTRACT_CONDA_ENV") == str(NO_REAL_ENVIRONMENT)


@pytest.mark.parametrize(
    ("gone", "value", "use_instead"),
    [
        ("dataset", "sroie", "document_type"),
        ("run_config", "/data/run_20261126/run_config.yml", "the variables listed at the top"),
        # The switch of the first task, under the name it had until 27 November 2026.
        ("photo_split", "true", "preprocess_photographs=true"),
    ],
)
def test_a_variable_the_pipeline_no_longer_reads_is_refused_by_name(
    tmp_path: Path, gone: str, value: str, use_instead: str
) -> None:
    result = started(a_clean_job(tmp_path, **{gone: value}))

    refused(result, f"FATAL: {gone} ", "no longer read", use_instead)
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize("gone", ["dataset", "run_config", "photo_split"])
def test_one_of_those_left_blank_by_a_manifest_is_not_refused(tmp_path: Path, gone: str) -> None:
    result = started(a_clean_job(tmp_path, **{gone: "None"}))

    assert "no longer read" not in result.stdout and (tmp_path / "logs").is_dir()


# --- the folder the script is started in, the environment, and the record of what is sent ---------------


def test_a_job_started_from_another_folder_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    """The script takes the folder it is started in for the code's: the commit is read there, the task is
    started there, and the places a job writes to are refused if they are inside it. Until 26 November
    2026 a start from another folder stopped for want of the settings file, which is gone."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    result = started(a_clean_job(tmp_path), cwd=elsewhere)

    refused(result, "FATAL: entrypoint.sh was not started from its own folder", str(REPO), str(elsewhere))
    assert not (tmp_path / "logs").exists()


def test_a_conda_environment_that_is_not_there_is_refused_before_anything_is_written(
    tmp_path: Path,
) -> None:
    result = started(a_clean_job(tmp_path), conda_env=tmp_path / "no_such_env")

    refused(result, "FATAL: the conda environment is not there", "no_such_env", "DOC_EXTRACT_CONDA_ENV")
    assert not (tmp_path / "logs").exists()


def test_a_folder_that_holds_no_python_is_not_taken_for_a_conda_environment(tmp_path: Path) -> None:
    (tmp_path / "a_folder").mkdir()

    result = started(a_clean_job(tmp_path), conda_env=tmp_path / "a_folder")

    refused(result, "FATAL: the conda environment is not there", "a_folder", "bin/python")
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize("blank", ["", "None"])
def test_a_conda_environment_left_blank_is_the_one_written_in_the_script(
    tmp_path: Path, blank: str
) -> None:
    """It is refused on this machine, which has no such folder: by the folder's own name, and not as 'None'."""
    result = started({**a_clean_job(tmp_path), "DOC_EXTRACT_CONDA_ENV": blank})

    assert "not there: None" not in result.stdout and "not there: \n" not in result.stdout
    assert "/home/jovyan/.conda/envs/vllm_env4" in result.stdout


def test_a_record_of_what_is_sent_that_names_a_folder_is_refused(tmp_path: Path) -> None:
    """Until 26 November 2026 it got through, each reply then failed to be kept, and the stage stopped
    saying that the engine had stopped answering."""
    (tmp_path / "out").mkdir()

    result = started(a_clean_job(tmp_path, DOC_EXTRACT_TRACE_FILE=str(tmp_path / "out")))

    refused(
        result, "FATAL: DOC_EXTRACT_TRACE_FILE is not a file that can be written", str(tmp_path / "out")
    )
    assert not (tmp_path / "logs").exists()


def test_a_record_of_what_is_sent_under_something_that_is_no_folder_is_refused(tmp_path: Path) -> None:
    (tmp_path / "a_file").write_text("not a folder")

    result = started(a_clean_job(tmp_path, DOC_EXTRACT_TRACE_FILE=str(tmp_path / "a_file" / "sent.jsonl")))

    refused(result, "FATAL: DOC_EXTRACT_TRACE_FILE is not a file that can be written", "a_file")
    assert not (tmp_path / "logs").exists()


def test_a_record_of_what_is_sent_in_a_folder_yet_to_be_made_is_accepted(tmp_path: Path) -> None:
    result = started(
        a_clean_job(tmp_path, DOC_EXTRACT_TRACE_FILE=str(tmp_path / "new" / "deep" / "sent.jsonl"))
    )

    assert "FATAL: DOC_EXTRACT_TRACE_FILE" not in result.stdout and (tmp_path / "logs").is_dir()


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
    # A line that RUNS Python: the word begins a command. The check that the environment holds a python
    # names one and runs none, and the two lines that log its version come after the environment is on.
    runs_python = [line for line in code if re.search(r"(^|[;&|(]\s*)python[0-9.]*\s", line.strip())]
    assert [line for line in runs_python if "log " not in line] == ["python3 -m stages"]
    assert not any("run_config.yml" in line or "CONFIG_FILE" in line for line in code)


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


# --- the look at the cards ------------------------------------------------------------------------------


def a_machine_with_cards(folder: Path, second_cards_state: str) -> str:
    """A PATH on which `nvidia-smi` is a stand-in that answers the script's four questions for two cards.

    The values are invented. They are written as nvidia-smi writes them: a comma and a space between.
    """
    folder.mkdir()
    stand_in = folder / "nvidia-smi"
    stand_in.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        "  --query-gpu=count) printf '2\\n2\\n' ;;\n"
        "  --query-gpu=index,name,*) printf '0, NVIDIA L4, 23034 MiB, 22594 MiB, 41, 0\\n"
        "1, NVIDIA L4, 23034 MiB, 20110 MiB, 43, 0\\n' ;;\n"
        f"  --query-gpu=index,pstate) printf '0, P8\\n1, {second_cards_state}\\n' ;;\n"
        f"  --query-gpu=index,pstate,*) printf 'index, pstate, ecc\\n0, P8, 0\\n1, {second_cards_state}, 0\\n' ;;\n"
        "esac\n"
    )
    stand_in.chmod(0o755)
    return f"{folder}{os.pathsep}{os.environ['PATH']}"


@needs_the_environment
def test_each_card_is_logged_on_a_line_of_its_own_with_its_values_evenly_set_out(tmp_path: Path) -> None:
    """The line is set out by awk. Until 27 November 2026 a loop in bash trimmed three of the six values
    and left the card's name and its temperature with the space nvidia-smi puts after each comma."""
    a_reply_to_clean(tmp_path / "out")

    result = started(a_clean_job(tmp_path, PATH=a_machine_with_cards(tmp_path / "bin", "P0")), A_CONDA_ENV)

    assert result.returncode == 0, result.stdout[-2000:]
    stamp = r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]"
    for line in (
        "  GPU 0: NVIDIA L4 | 23034 MiB total | 22594 MiB free | 41C | ECC errors: 0",
        "  GPU 1: NVIDIA L4 | 23034 MiB total | 20110 MiB free | 43C | ECC errors: 0",
    ):
        assert re.search(rf"^{stamp} {re.escape(line)}$", result.stdout, flags=re.MULTILINE), result.stdout
    assert "Detected 2 GPU(s):" in result.stdout
    assert "report error state" not in result.stdout


@needs_the_environment
@pytest.mark.parametrize("state", ["ERR!", "[Unknown Error]"])
def test_a_card_in_an_error_state_is_warned_of_and_the_job_goes_on(tmp_path: Path, state: str) -> None:
    a_reply_to_clean(tmp_path / "out")

    result = started(a_clean_job(tmp_path, PATH=a_machine_with_cards(tmp_path / "bin", state)), A_CONDA_ENV)

    assert result.returncode == 0, result.stdout[-2000:]
    assert "WARNING: One or more GPUs report error state. Run may fail." in result.stdout
    assert f"1, {state}, 0" in result.stdout


def test_awk_is_kept_out_of_the_path_the_log_takes() -> None:
    """Fed from a pipe, some awks hold lines back until their input ends, and a live log would stall.
    Measured on 27 November 2026: mawk passed on two lines sent two seconds apart together, at the end."""
    script = (REPO / "entrypoint.sh").read_text()
    (redirection,) = [line for line in script.splitlines() if line.startswith("exec > ")]

    assert redirection == 'exec > >(tee -a "$LOG_FILE") 2>&1'
    # Only what every awk has: no function that gawk alone gives.
    code = [line for line in script.splitlines() if "awk " in line and not line.lstrip().startswith("#")]
    assert len(code) == 2
    assert not any(word in line for line in code for word in ("strftime", "gensub", "systime", "asort"))
