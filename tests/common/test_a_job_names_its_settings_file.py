"""A job names a settings file of its own, so that no setting is changed by writing in the code's folder.

Until 26 November 2026 the settings file read was always the one beside the code. Under a pipeline that
folder is read-only: a job can write only to storage mounted beside it. So every setting that is not one
of the job's variables could be changed only where a job cannot write, and each run on the sandbox began
by editing the tracked file in place.

The job's variable ``run_config`` names a whole settings file, anywhere the job can read. It stands in
the shipped file's place and is checked exactly as the shipped file is. Nothing is merged between them.

The folders are temporary ones, and every value written into a settings file here is invented.
"""

import hashlib
import importlib
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from common import run_manifest
from common import stage_settings
from common.config_schema import ConfigError
from common.stage_settings import SETTINGS_FILE, resolve, settings_file
from stages import __main__ as starter

REPO = Path(__file__).resolve().parents[2]
SHIPPED = REPO / "config" / "run_config.yml"
PARTS = ("What:", "Where:", "Expected:", "How to fix:")
A_MODEL = "gemma4-31b-w4a16-vllm"


def a_file_of_its_own(folder: Path, change: Any = None, name: str = "run_config.yml") -> Path:
    """A copy of the shipped settings file somewhere a job can write, with one change made to it."""
    raw = yaml.safe_load(SHIPPED.read_text())
    raw["bootstrap"]["logging"]["log_dir"] = str(folder / "logs_named_in_the_file")
    if change is not None:
        change(raw)
    path = folder / name
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def environ(folder: Path, **more: str) -> dict[str, str]:
    (folder / "pages").mkdir(exist_ok=True)
    return {"model": A_MODEL, "image_dir": str(folder / "pages"), "output": str(folder / "out"), **more}


class Called:
    """Stands in for a stage's ``run``: notes what it was given."""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.kwargs = kwargs


@pytest.fixture
def stage(monkeypatch: pytest.MonkeyPatch) -> Called:
    called = Called()
    module = importlib.import_module("stages.classify")
    monkeypatch.setattr(module, "run", called)
    monkeypatch.setattr(module, "check", lambda task: None)
    monkeypatch.setattr(run_manifest, "code_version", lambda repository: ("0a1b2c3", False))
    return called


# --- which file is read ---------------------------------------------------------------------------------


@pytest.mark.parametrize("unset", [{}, {SETTINGS_FILE: ""}, {SETTINGS_FILE: "None"}])
def test_a_job_that_names_no_file_reads_the_shipped_one(unset: dict[str, str]) -> None:
    assert settings_file(unset, SHIPPED) == SHIPPED


def test_a_job_that_names_a_file_reads_that_one(tmp_path: Path) -> None:
    named = a_file_of_its_own(tmp_path)

    assert settings_file({SETTINGS_FILE: str(named)}, SHIPPED) == named


def test_a_file_that_is_not_there_stops_the_run_with_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as refused:
        settings_file({SETTINGS_FILE: str(tmp_path / "no_such.yml")}, SHIPPED)

    message = str(refused.value)
    assert all(part in message for part in PARTS)
    assert SETTINGS_FILE in message and "no_such.yml" in message and str(SHIPPED) in message


def test_a_folder_is_not_a_settings_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=SETTINGS_FILE):
        settings_file({SETTINGS_FILE: str(tmp_path)}, SHIPPED)


# --- the task is worked out from the file named, and its stage reads the same file ----------------------


def started_with(monkeypatch: pytest.MonkeyPatch, variables: dict[str, str]) -> int:
    monkeypatch.setattr(starter.os, "environ", {"KFP_TASK": "classify", **variables})
    return starter.main()


def test_the_task_is_worked_out_from_the_file_the_job_names(
    tmp_path: Path, stage: Called, monkeypatch: pytest.MonkeyPatch
) -> None:
    named = a_file_of_its_own(tmp_path)

    assert started_with(monkeypatch, environ(tmp_path, **{SETTINGS_FILE: str(named)})) == 0

    assert stage.kwargs["config_path"] == named != SHIPPED


def test_a_value_changed_in_the_file_named_is_the_value_the_task_has(tmp_path: Path) -> None:
    def another_model(raw: dict[str, Any]) -> None:
        raw["bootstrap"]["model"]["type"] = "gemma4-12b-unified-w4a16-vllm"

    named = a_file_of_its_own(tmp_path, another_model)
    job = {key: value for key, value in environ(tmp_path).items() if key != "model"}

    task = resolve("clean", job, settings_file({SETTINGS_FILE: str(named)}, SHIPPED))

    assert task.model == "gemma4-12b-unified-w4a16-vllm"
    assert task.log_dir == tmp_path / "logs_named_in_the_file"
    assert yaml.safe_load(SHIPPED.read_text())["bootstrap"]["model"]["type"] != task.model


def test_nothing_is_merged_from_the_shipped_file(
    tmp_path: Path, stage: Called, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A key the file named leaves out is missing, though the shipped file has it."""

    def left_out(raw: dict[str, Any]) -> None:
        del raw["pipeline"]["datasets"]

    named = a_file_of_its_own(tmp_path, left_out)

    assert started_with(monkeypatch, environ(tmp_path, **{SETTINGS_FILE: str(named)})) == 1

    printed = capsys.readouterr().out
    assert "FATAL: the task cannot be started." in printed
    assert "pipeline.datasets" in printed and str(named) in printed
    assert "datasets" in yaml.safe_load(SHIPPED.read_text())["pipeline"]


def test_the_file_named_is_checked_as_the_shipped_file_is(
    tmp_path: Path, stage: Called, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def a_key_nothing_reads(raw: dict[str, Any]) -> None:
        raw["pipeline"]["processing"]["speed"] = "fast"

    named = a_file_of_its_own(tmp_path, a_key_nothing_reads)

    assert started_with(monkeypatch, environ(tmp_path, **{SETTINGS_FILE: str(named)})) == 1

    printed = capsys.readouterr().out
    assert "pipeline.processing.speed" in printed and str(named) in printed
    assert not (tmp_path / "out").exists()


def test_a_job_that_names_a_file_that_is_not_there_ends_with_its_reason(
    tmp_path: Path, stage: Called, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    job = environ(tmp_path, **{SETTINGS_FILE: str(tmp_path / "mislaid.yml")})

    assert started_with(monkeypatch, job) == 1

    printed = capsys.readouterr().out
    assert "FATAL: the task cannot be started." in printed and "mislaid.yml" in printed
    assert all(part in printed for part in PARTS)


# --- the log folder, which the entry script asks for before anything else -------------------------------


def log_folder_printed(variables: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "common.stage_settings", str(SHIPPED)],
        cwd=REPO,
        env={"PATH": "", **variables},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_the_log_folder_is_the_one_of_the_file_the_job_names(tmp_path: Path) -> None:
    named = a_file_of_its_own(tmp_path)

    printed = log_folder_printed({SETTINGS_FILE: str(named)})

    assert printed.returncode == 0
    assert printed.stdout.strip() == str(tmp_path / "logs_named_in_the_file")


def test_the_log_folder_the_job_sets_still_comes_first(tmp_path: Path) -> None:
    named = a_file_of_its_own(tmp_path)

    printed = log_folder_printed({SETTINGS_FILE: str(named), "DOC_EXTRACT_LOG_DIR": str(tmp_path / "mine")})

    assert printed.stdout.strip() == str(tmp_path / "mine")


def test_the_entry_script_is_told_of_a_file_that_is_not_there(tmp_path: Path) -> None:
    printed = log_folder_printed({SETTINGS_FILE: str(tmp_path / "mislaid.yml")})

    assert printed.returncode == 1
    assert "FATAL" in printed.stdout and "mislaid.yml" in printed.stdout


def test_the_entry_script_chooses_no_settings_file_itself() -> None:
    """It hands the shipped file to Python, where the job's own is read in its place: one place decides."""
    script = (REPO / "entrypoint.sh").read_text()
    code = "\n".join(line for line in script.splitlines() if not line.lstrip().startswith("#"))

    assert 'CONFIG_FILE="./config/run_config.yml"' in code
    # The variable is read nowhere in the script: neither as $run_config nor as ${run_config...}.
    assert f"${SETTINGS_FILE}" not in code and f"${{{SETTINGS_FILE}" not in code


# --- the note of what made the folder -------------------------------------------------------------------


def test_a_stage_notes_the_settings_file_it_read_and_what_it_held(
    tmp_path: Path, stage: Called, monkeypatch: pytest.MonkeyPatch
) -> None:
    named = a_file_of_its_own(tmp_path)

    started_with(monkeypatch, environ(tmp_path, **{SETTINGS_FILE: str(named)}))

    note = run_manifest.read(tmp_path / "out")["classify"]
    assert note.settings_file == str(named)
    assert note.settings_sha256 == hashlib.sha256(named.read_bytes()).hexdigest()


def test_a_stage_that_carries_on_with_other_settings_says_so_and_goes_on(
    tmp_path: Path, stage: Called, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    named = a_file_of_its_own(tmp_path)
    job = environ(tmp_path, **{SETTINGS_FILE: str(named)})
    started_with(monkeypatch, job)
    first = run_manifest.read(tmp_path / "out")["classify"]
    a_file_of_its_own(tmp_path, lambda raw: raw["pipeline"]["processing"].update(verbose=True))

    with caplog.at_level(logging.WARNING):
        assert started_with(monkeypatch, job) == 0

    said = "\n".join(record.getMessage() for record in caplog.records)
    assert first.settings_sha256 is not None
    assert "not the ones it began with" in said and first.settings_sha256[:12] in said
    assert run_manifest.read(tmp_path / "out")["classify"].settings_sha256 != first.settings_sha256


def test_a_stage_that_carries_on_with_the_same_settings_says_nothing(
    tmp_path: Path, stage: Called, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    job = environ(tmp_path, **{SETTINGS_FILE: str(a_file_of_its_own(tmp_path))})
    started_with(monkeypatch, job)

    with caplog.at_level(logging.WARNING):
        started_with(monkeypatch, job)

    assert "not the ones it began with" not in "\n".join(r.getMessage() for r in caplog.records)


def test_a_note_written_before_the_settings_file_was_noted_is_still_read(tmp_path: Path) -> None:
    (tmp_path / run_manifest.FILE).write_text(
        '{"stages": {"classify": {"model": "m", "image_dir": "/pages", "commit": "0a1b2c3", '
        '"uncommitted_changes": false, "started": "2026-11-25T10:00:00", "last_run": "2026-11-25T10:00:00"}}}'
    )

    note = run_manifest.read(tmp_path)["classify"]

    assert note.settings_file is None and note.settings_sha256 is None
    assert "settings" not in run_manifest.describe(tmp_path)[0]


def test_the_variable_is_one_the_readme_and_the_entry_script_list() -> None:
    assert stage_settings.SETTINGS_FILE == "run_config"
    assert "#   run_config " in (REPO / "entrypoint.sh").read_text()
    assert "| `run_config` |" in (REPO / "README.md").read_text()
