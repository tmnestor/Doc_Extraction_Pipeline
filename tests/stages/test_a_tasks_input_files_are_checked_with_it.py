"""A task's input files are part of what is checked before it starts.

Until 26 November 2026 a task met a missing input only when it came to read it, and ended with a
traceback: the reading function's own error for a file that was not there, the JSON parser's for a line
that could not be read, and ``FileNotFoundError`` for a folder with no page in it. The runner says the
reason for two kinds of ending, a setting that cannot be used and a stage that was stopped, and these
were neither.

Each is now the same kind of fault as a setting that cannot be used: the task's check reads its input,
and says what is wrong, where, what is expected and how to mend it. What the check read is what the task
then runs with: no file is read a second time.

The pages are blank images made by the bench, and every reply and every answer is invented.
"""

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from bench import Bench
from common.stage_settings import resolve
from stages import __main__ as starter
from stages.io import read_input

PARTS = ("What:", "Where:", "Expected:", "How to fix:")
# The file each task reads, the task that writes it, and the task that reads it.
INPUTS = (
    ("extract", "classifications.jsonl", "classify"),
    ("clean", "raw_extractions.jsonl", "extract"),
    ("evaluate", "cleaned_extractions.jsonl", "clean"),
)


def environ(bench: Bench, out: Path, **more: str) -> dict[str, str]:
    return {
        "model": "gemma4-31b-w4a16-vllm",
        "image_dir": str(bench.pages),
        "output": str(out),
        "DOC_EXTRACT_MODELS_DIR": str(bench.models),
        "ground_truth": str(out.parent / "ground_truth.jsonl"),
        **more,
    }


def started(bench: Bench, monkeypatch: pytest.MonkeyPatch, name: str, out: Path, **more: str) -> int:
    monkeypatch.setattr(starter.os, "environ", {"KFP_TASK": name, **environ(bench, out, **more)})
    return starter.main()


def run_up_to(bench: Bench, name: str, out: Path) -> None:
    """Run every task before ``name``, so that its input is there."""
    bench.answer_key(out.parent / "ground_truth.jsonl")
    for earlier in ("classify", "extract", "clean"):
        if earlier == name:
            return
        starter.run_task(resolve(earlier, environ(bench, out)))


def everything_under(folder: Path) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in folder.rglob("*") if path.is_file()}


# --- a file an earlier task wrote ----------------------------------------------------------------------


@pytest.mark.parametrize(("name", "file", "written_by"), INPUTS)
def test_a_task_whose_input_is_not_there_ends_with_its_reason_and_no_traceback(
    bench: Bench,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    name: str,
    file: str,
    written_by: str,
) -> None:
    out = tmp_path / "out"
    run_up_to(bench, name, out)
    (out / file).unlink()
    before = everything_under(out)
    capsys.readouterr()

    assert started(bench, monkeypatch, name, out) == 1

    printed = capsys.readouterr().out
    assert "FATAL: the task cannot be started." in printed and "Traceback" not in printed
    assert all(part in printed for part in PARTS)
    assert file in printed and f"the {written_by} task" in printed and str(out) in printed
    assert everything_under(out) == before
    assert bench.engines_started == (1 if name == "extract" else 2)


@pytest.mark.parametrize(("name", "file", "written_by"), INPUTS)
@pytest.mark.parametrize(
    ("spoilt", "said"),
    [
        # Cut short in the middle of a line: the six records are in two halves of the same length.
        (lambda text: text[: len(text) // 2 - 7], "cannot be read at line"),
        (lambda text: "", "holds no record"),
        (lambda text: "[1, 2]\n" + text, "what is not a record"),
    ],
)
def test_a_task_whose_input_cannot_be_read_says_which_line_and_what_to_run(
    bench: Bench,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    name: str,
    file: str,
    written_by: str,
    spoilt: Any,
    said: str,
) -> None:
    out = tmp_path / "out"
    run_up_to(bench, name, out)
    (out / file).write_text(spoilt((out / file).read_text()))
    capsys.readouterr()

    assert started(bench, monkeypatch, name, out) == 1

    printed = capsys.readouterr().out
    assert "FATAL: the task cannot be started." in printed and said in printed
    assert all(part in printed for part in PARTS) and f"the {written_by} task" in printed


def test_a_file_that_can_be_read_is_given_record_by_record(tmp_path: Path) -> None:
    path = tmp_path / "classifications.jsonl"
    path.write_text('{"image_name": "a.png"}\n\n{"image_name": "b.png"}\n')

    assert read_input(path, written_by="classify") == [{"image_name": "a.png"}, {"image_name": "b.png"}]


# --- a folder of pages -----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["split", "classify"])
def test_a_folder_with_no_page_in_it_ends_with_its_reason_and_no_traceback(
    bench: Bench,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    name: str,
) -> None:
    empty = tmp_path / "no_pages"
    empty.mkdir()

    assert started(bench, monkeypatch, name, tmp_path / "out", image_dir=str(empty)) == 1

    printed = capsys.readouterr().out
    assert "FATAL: the task cannot be started." in printed and "Traceback" not in printed
    assert all(part in printed for part in PARTS)
    assert "holds no image" in printed and str(empty) in printed and "image_dir" in printed
    assert not (tmp_path / "out").exists()
    assert bench.engines_started == 0


def test_a_folder_with_no_page_of_the_types_selected_says_which_setting_selects(
    bench: Bench, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert started(bench, monkeypatch, "classify", tmp_path / "out", document_types="payslip") == 1

    printed = capsys.readouterr().out
    assert "payslip" in printed and "the variable document_types" in printed


# --- the answer key ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "said"),
    [
        (None, "is not there"),
        ("", "holds no answer"),
        ("not json\n", "cannot be read as an answer"),
        ('{"TOTAL_AMOUNT": ".07"}\n', "holds no answer"),
    ],
)
def test_an_answer_key_that_cannot_be_used_ends_with_its_reason_and_no_traceback(
    bench: Bench,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    key: str | None,
    said: str,
) -> None:
    out = tmp_path / "out"
    run_up_to(bench, "evaluate", out)
    starter.run_task(resolve("clean", environ(bench, out)))
    answers = tmp_path / "ground_truth.jsonl"
    answers.unlink()
    if key is not None:
        answers.write_text(key)
    capsys.readouterr()

    assert started(bench, monkeypatch, "evaluate", out) == 1

    printed = capsys.readouterr().out
    assert "FATAL: the task cannot be started." in printed and "Traceback" not in printed
    assert all(part in printed for part in PARTS)
    assert said in printed and "ground_truth" in printed
    assert not (out / "evaluation").exists()


# --- what the check read is what the task runs with -----------------------------------------------------


@pytest.mark.parametrize(("name", "file", "written_by"), INPUTS)
def test_a_tasks_input_is_read_once(
    bench: Bench, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, file: str, written_by: str
) -> None:
    out = tmp_path / "out"
    run_up_to(bench, name, out)
    stage = importlib.import_module(f"stages.{name}")
    read: list[str] = []
    real = stage.read_input

    def counted(path: Path, *, written_by: str) -> list[dict[str, Any]]:
        read.append(path.name)
        return real(path, written_by=written_by)

    monkeypatch.setattr(stage, "read_input", counted)

    starter.run_task(resolve(name, environ(bench, out)))

    assert read == [file]


def test_the_tasks_all_run_and_score_after_these_checks(bench: Bench, tmp_path: Path) -> None:
    out = tmp_path / "out"
    run_up_to(bench, "evaluate", out)
    for name in ("clean", "evaluate"):
        starter.run_task(resolve(name, environ(bench, out)))

    results = [
        json.loads(line)
        for line in (out / "evaluation" / "evaluation_results.jsonl").read_text().splitlines()
    ]
    assert len(results) == len(bench.names)
