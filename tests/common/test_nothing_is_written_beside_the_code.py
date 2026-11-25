"""Nothing of the pipeline is written beside the code, and no task runs git.

Under a pipeline the code's folder is in the job's pod. That storage is not kept, and a job that writes
there is stopped part way. git writes that folder once, when the job is launched from its commit, and is
not used after that.

What is decided at launch is decided in ``entrypoint.sh``, in bash: the folders a job names are refused
there if they are beside the code, Python is told to write no compiled copies of the code, and the
commit is read from the checkout's files (``tests/scripts/test_the_launch_is_decided_in_the_entry_script.py``).
Here the Python is held to its part: no module a task runs starts git, and a whole run of the five tasks
is watched, by Python's own audit of every file opened for writing, made, moved or removed, and touches
nothing beside the code.

The pages are blank images made by the bench, and every reply and every answer is invented.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from bench import AN_ENGINE_ON_EACH_CARD, ONE_ENGINE, Bench
from common import run_manifest, stage_settings
from common.stage_settings import resolve
from stages import __main__ as starter

REPO = Path(__file__).resolve().parents[2]
PARTS = ("What:", "Where:", "Expected:", "How to fix:")


def environ(bench: Bench, out: Path, **more: str) -> dict[str, str]:
    return {
        "model": ONE_ENGINE,
        "image_dir": str(bench.pages),
        "output": str(out),
        "DOC_EXTRACT_LOG_DIR": str(out.parent / "logs"),
        "ground_truth": str(out.parent / "ground_truth.jsonl"),
        **more,
    }


# --- what is not the pipeline's own writing ------------------------------------------------------------


def test_python_is_told_to_write_no_compiled_code_before_the_first_python_starts() -> None:
    script = (REPO / "entrypoint.sh").read_text()
    code = [line for line in script.splitlines() if not line.lstrip().startswith("#")]
    told = next(i for i, line in enumerate(code) if line.strip() == "export PYTHONDONTWRITEBYTECODE=1")
    first_python = next(i for i, line in enumerate(code) if "CONDA_PY" in line and "-m " in line)

    assert told < first_python
    assert all("python" not in line for line in code[:told])


# --- git is not used once a task has started -------------------------------------------------------------
# A job under a pipeline is launched from a commit: git writes the code's folder then, and that is the
# last it is used. Until 26 November 2026 every task that uses the cards ran git twice as it started.


def test_no_module_a_task_runs_starts_git() -> None:
    for module in sorted(
        [*(REPO / "common").glob("*.py"), *(REPO / "stages").glob("*.py"), *(REPO / "models").rglob("*.py")]
    ):
        source = module.read_text()
        assert '"git"' not in source and "'git'" not in source, module.name
    assert "subprocess" not in (REPO / "common" / "run_manifest.py").read_text()
    script = [
        line
        for line in (REPO / "entrypoint.sh").read_text().splitlines()
        if not line.lstrip().startswith("#")
    ]
    assert not any("git " in line for line in script)


def test_a_task_that_notes_its_folder_starts_no_process(
    bench: Bench, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refused(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"a process was started: {args[0] if args else kwargs}")

    monkeypatch.setattr(subprocess, "run", refused)
    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setenv("DOC_EXTRACT_COMMIT", "0a1b2c3")

    starter.run_task(resolve("classify", environ(bench, tmp_path / "out"), bench.config))

    note = run_manifest.read(tmp_path / "out")["classify"]
    assert (note.commit, note.uncommitted_changes) == ("0a1b2c3", None)


# --- a whole run, watched --------------------------------------------------------------------------------

_WATCHING: list[list[str]] = []
_NOT_THE_PIPELINES = ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".coverage")
_WRITES = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
_CHANGES = (
    "os.mkdir",
    "os.remove",
    "os.rename",
    "os.rmdir",
    "os.truncate",
    "shutil.copyfile",
    "shutil.move",
)


def _audit(event: str, arguments: tuple[Any, ...]) -> None:
    """Note a file beside the code that is opened for writing, made, moved or removed."""
    if not _WATCHING:
        return
    if event == "open":
        if not isinstance(arguments[2], int) or not arguments[2] & _WRITES:
            return
        touched = [arguments[0]]
    elif event in _CHANGES:
        touched = list(arguments[:2])
    else:
        return
    for path in touched:
        if not isinstance(path, str | bytes | os.PathLike):
            continue
        resolved = Path(os.fsdecode(path)).resolve()
        beside = resolved == REPO or REPO in resolved.parents
        if beside and not any(part.startswith(_NOT_THE_PIPELINES) for part in resolved.parts):
            _WATCHING[-1].append(f"{event}: {resolved.relative_to(REPO)}")


sys.addaudithook(_audit)


def test_the_watch_sees_a_file_written_beside_the_code_and_not_one_written_elsewhere(
    tmp_path: Path,
) -> None:
    """A watch that saw nothing would pass every run: this one is shown a write, and sees it."""
    _WATCHING.append([])
    try:
        (tmp_path / "elsewhere.txt").write_text("x")
        planted = REPO / "written_beside_the_code_by_a_test.txt"
        planted.write_text("x")
        planted.unlink()
    finally:
        seen = _WATCHING.pop()

    assert seen == [
        "open: written_beside_the_code_by_a_test.txt",
        "os.remove: written_beside_the_code_by_a_test.txt",
    ]


@pytest.mark.parametrize("model", [ONE_ENGINE, AN_ENGINE_ON_EACH_CARD])
def test_a_whole_run_of_the_five_tasks_writes_nothing_beside_the_code(
    bench: Bench, tmp_path: Path, model: str
) -> None:
    out = tmp_path / "out"
    bench.answer_key(tmp_path / "ground_truth.jsonl")
    job = environ(bench, out, model=model)

    _WATCHING.append([])
    try:
        for name in stage_settings.TASKS:
            if name == "split":
                continue  # its images would become the pages: it is run on its own, below
            starter.run_task(resolve(name, job, bench.config))
        starter.run_task(
            resolve("split", environ(bench, tmp_path / "photographs", model=model), bench.config)
        )
        starter.run_task(resolve("classify", {**job, "CLEAR_PREV_OUTPUT": "true"}, bench.config))
    finally:
        seen = _WATCHING.pop()

    assert seen == []
    assert (out / "evaluation" / "evaluation_results.jsonl").is_file()
    assert (tmp_path / "photographs" / "split" / "splits.jsonl").is_file()
