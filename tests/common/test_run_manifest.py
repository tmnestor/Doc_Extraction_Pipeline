"""An output folder knows what made it, and a stage that carries on in one checks that it is the same run.

Until 24 November 2026 a stage carried on from whatever its output folder held, by the names of the pages
already there. A second run into the same folder with another model, another folder of pages or another
version of the code skipped every page the first had finished and reported a score that was partly the
first run's. The settings file warned of this in five places, and nothing stopped it. On that day a
measurement was also made at the wrong commit, and nothing in what came back said so.

Folder names, model names and commits are invented, except the one registered model the settings need.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from common import run_manifest
from common.config_schema import ConfigError
from common.stage_settings import StageSettings, resolve
from stages import __main__ as starter

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_CONFIG = REPO_ROOT / "config" / "run_config.yml"
A_MODEL = "gemma4-31b-w4a16-vllm"
ANOTHER_MODEL = "gemma4-12b-unified-w4a16-vllm"
FOUR_PARTS = ("What:", "Where:", "Expected:", "How to fix:")


@pytest.fixture
def commit(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stands in for git: the commit and the state of the tree are what the test says."""
    code: dict[str, Any] = {"commit": "0a1b2c3", "changed": False}
    monkeypatch.setattr(run_manifest, "code_version", lambda repository: (code["commit"], code["changed"]))
    return code


def note(output: Path, stage: str = "classify", **given: Any) -> run_manifest.StageNote:
    with_these: dict[str, Any] = {
        "model": A_MODEL,
        "image_dir": "/pages/set_one",
        "repository": REPO_ROOT,
        "starting_afresh": False,
        **given,
    }
    return run_manifest.check_and_note(output, stage, **with_these)


def test_a_stage_notes_what_it_was_run_with(tmp_path: Path, commit: dict[str, Any]) -> None:
    written = note(tmp_path, now=datetime(2031, 2, 9, 10, 15))

    held = json.loads((tmp_path / run_manifest.FILE).read_text())
    assert held == {
        "stages": {
            "classify": {
                "model": A_MODEL,
                "image_dir": "/pages/set_one",
                "commit": "0a1b2c3",
                "uncommitted_changes": False,
                "started": "2031-02-09T10:15:00",
                "last_run": "2031-02-09T10:15:00",
            }
        }
    }
    assert run_manifest.read(tmp_path) == {"classify": written}


def test_the_same_run_carries_on_and_keeps_when_it_started(tmp_path: Path, commit: dict[str, Any]) -> None:
    note(tmp_path, now=datetime(2031, 2, 9, 10, 15))

    again = note(tmp_path, now=datetime(2031, 2, 9, 14, 40))

    assert (again.started, again.last_run) == ("2031-02-09T10:15:00", "2031-02-09T14:40:00")


@pytest.mark.parametrize(
    ("change", "was", "now"),
    [
        ({"model": ANOTHER_MODEL}, A_MODEL, ANOTHER_MODEL),
        ({"image_dir": "/pages/set_two"}, "/pages/set_one", "/pages/set_two"),
        ({"commit": "9f8e7d6"}, "0a1b2c3", "9f8e7d6"),
    ],
)
def test_a_stage_does_not_carry_on_from_another_runs_output(
    tmp_path: Path, commit: dict[str, Any], change: dict[str, str], was: str, now: str
) -> None:
    note(tmp_path)
    commit["commit"] = change.pop("commit", commit["commit"])

    with pytest.raises(ConfigError) as refused:
        note(tmp_path, **change)

    message = str(refused.value)
    assert all(part in message for part in FOUR_PARTS)
    assert f"was {was}, and is now {now}" in message
    assert str(tmp_path / run_manifest.FILE) in message or run_manifest.FILE in message
    assert "CLEAR_PREV_OUTPUT=true" in message and "folder of its own" in message
    # The earlier run's note is left as it was.
    assert run_manifest.read(tmp_path)["classify"].model == A_MODEL
    assert run_manifest.read(tmp_path)["classify"].commit == "0a1b2c3"


def test_every_difference_is_named_at_once(tmp_path: Path, commit: dict[str, Any]) -> None:
    note(tmp_path)
    commit["commit"] = "9f8e7d6"

    with pytest.raises(ConfigError) as refused:
        note(tmp_path, model=ANOTHER_MODEL, image_dir="/pages/set_two")

    message = str(refused.value)
    assert "the model was" in message and "the folder of pages was" in message
    assert "the code's commit was" in message


def test_a_commit_that_cannot_be_told_is_no_evidence_of_another_run(
    tmp_path: Path, commit: dict[str, Any]
) -> None:
    commit["commit"] = run_manifest.UNKNOWN
    note(tmp_path)
    commit["commit"] = "9f8e7d6"
    note(tmp_path)
    commit["commit"] = run_manifest.UNKNOWN

    assert note(tmp_path).commit == run_manifest.UNKNOWN


def test_uncommitted_changes_are_noted_and_not_compared(tmp_path: Path, commit: dict[str, Any]) -> None:
    note(tmp_path)
    commit["changed"] = True

    assert note(tmp_path).uncommitted_changes is True


def test_a_stage_starting_afresh_checks_nothing_and_notes_itself_anew(
    tmp_path: Path, commit: dict[str, Any]
) -> None:
    note(tmp_path, now=datetime(2031, 2, 9, 10, 15))

    anew = note(tmp_path, model=ANOTHER_MODEL, starting_afresh=True, now=datetime(2031, 2, 10, 9, 0))

    assert (anew.model, anew.started) == (ANOTHER_MODEL, "2031-02-10T09:00:00")
    assert run_manifest.read(tmp_path)["classify"] == anew


def test_each_stage_has_its_own_note(tmp_path: Path, commit: dict[str, Any]) -> None:
    """The split stage reads the photographs and the classify stage reads what it wrote: two folders."""
    note(tmp_path, "split", image_dir="/photographs")
    note(tmp_path, "classify", image_dir="/out/split/images")

    assert list(run_manifest.read(tmp_path)) == ["split", "classify"]
    assert note(tmp_path, "split", image_dir="/photographs").image_dir == "/photographs"


def test_a_stage_run_another_way_than_the_one_it_reads_is_said_and_allowed(
    tmp_path: Path, commit: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    note(tmp_path, "classify")
    commit["commit"] = "9f8e7d6"
    note(tmp_path, "extract")

    with caplog.at_level(logging.WARNING):
        run_manifest.warn_if_made_another_way(tmp_path, "extract", after="classify")

    (said,) = [record.getMessage() for record in caplog.records]
    assert "the code's commit was 0a1b2c3 there and is 9f8e7d6 here" in said


def test_stages_run_the_same_way_are_not_remarked_on(
    tmp_path: Path, commit: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    note(tmp_path, "classify")
    note(tmp_path, "extract")

    with caplog.at_level(logging.WARNING):
        run_manifest.warn_if_made_another_way(tmp_path, "extract", after="classify")
        run_manifest.warn_if_made_another_way(tmp_path, "extract", after="split")

    assert caplog.records == []


@pytest.mark.parametrize("held", ["not json", '{"stages": {"classify": {"model": "m"}}}', "[]"])
def test_notes_that_cannot_be_read_stop_the_run(tmp_path: Path, held: str) -> None:
    (tmp_path / run_manifest.FILE).write_text(held)

    with pytest.raises(ConfigError) as refused:
        run_manifest.read(tmp_path)

    assert all(part in str(refused.value) for part in FOUR_PARTS)


def test_a_folder_with_no_notes_has_none(tmp_path: Path) -> None:
    assert run_manifest.read(tmp_path) == {}
    assert run_manifest.describe(tmp_path) == []


def test_the_commit_of_this_repository_is_found_and_of_no_repository_is_unknown(tmp_path: Path) -> None:
    found, changed = run_manifest.code_version(REPO_ROOT)
    assert found != run_manifest.UNKNOWN and 7 <= len(found) <= 12
    assert isinstance(changed, bool)

    assert run_manifest.code_version(tmp_path) == (run_manifest.UNKNOWN, None)


# --- through the task runner ------------------------------------------------------------------------


def settings_for(task: str, folder: Path, **changes: Any) -> StageSettings:
    environ = {
        "model": A_MODEL,
        "image_dir": str(folder / "pages"),
        "output": str(folder / "out"),
        "ground_truth": str(folder / "truth.jsonl"),
        "DOC_EXTRACT_LOG_DIR": str(folder / "logs"),
        **changes,
    }
    return resolve(task, environ, RUN_CONFIG)


class Called:
    """Stands in for a stage's ``run``: counts its calls."""

    def __init__(self) -> None:
        self.times = 0

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.times += 1


@pytest.fixture
def stage(monkeypatch: pytest.MonkeyPatch) -> Called:
    called = Called()
    monkeypatch.setattr(starter, "_stage_run", lambda task: called)
    return called


def test_a_second_model_does_not_carry_on_in_the_first_models_folder(
    tmp_path: Path, stage: Called, commit: dict[str, Any]
) -> None:
    starter.run_task(settings_for("classify", tmp_path))
    (tmp_path / "out" / "classifications.jsonl").write_text('{"image_name": "page_1.png"}\n')

    with pytest.raises(ConfigError):
        starter.run_task(settings_for("classify", tmp_path, model=ANOTHER_MODEL))

    # Stopped before the stage was started, and before anything of the first run was touched.
    assert stage.times == 1
    assert (tmp_path / "out" / "classifications.jsonl").exists()


def test_a_second_model_may_start_the_stage_afresh(
    tmp_path: Path, stage: Called, commit: dict[str, Any]
) -> None:
    starter.run_task(settings_for("classify", tmp_path))

    starter.run_task(settings_for("classify", tmp_path, model=ANOTHER_MODEL, CLEAR_PREV_OUTPUT="true"))

    assert stage.times == 2
    assert run_manifest.read(tmp_path / "out")["classify"].model == ANOTHER_MODEL


def test_the_split_stage_checks_even_when_told_to_start_afresh(
    tmp_path: Path, stage: Called, commit: dict[str, Any]
) -> None:
    """Nothing of the split stage's is deleted for it, so its earlier output is still carried on from."""
    starter.run_task(settings_for("split", tmp_path))

    with pytest.raises(ConfigError) as refused:
        starter.run_task(settings_for("split", tmp_path, model=ANOTHER_MODEL, CLEAR_PREV_OUTPUT="true"))

    assert f"delete {tmp_path / 'out' / 'split'}" in str(refused.value)


def test_every_stage_that_carries_on_leaves_a_note_and_the_others_leave_none(
    tmp_path: Path, stage: Called, commit: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    for task in ("split", "classify", "extract"):
        starter.run_task(settings_for(task, tmp_path))
    with caplog.at_level(logging.INFO):
        for task in ("clean", "evaluate"):
            starter.run_task(settings_for(task, tmp_path))

    assert tuple(run_manifest.read(tmp_path / "out")) == run_manifest.STAGES
    # The stages that score say what made the folder, so that a score carries its model and commit.
    said = "\n".join(record.getMessage() for record in caplog.records)
    assert f"made by extract: {A_MODEL}" in said and "commit 0a1b2c3" in said


def test_a_run_refused_ends_with_the_reason_and_not_a_traceback(
    tmp_path: Path,
    stage: Called,
    commit: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    starter.run_task(settings_for("classify", tmp_path))
    for name, value in {
        "KFP_TASK": "classify",
        "model": ANOTHER_MODEL,
        "image_dir": str(tmp_path / "pages"),
        "output": str(tmp_path / "out"),
        "DOC_EXTRACT_LOG_DIR": str(tmp_path / "logs"),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("CLEAR_PREV_OUTPUT", raising=False)

    assert starter.main() == 1

    printed = capsys.readouterr().out
    assert printed.startswith("FATAL: the task cannot be started.")
    assert all(part in printed for part in FOUR_PARTS)
