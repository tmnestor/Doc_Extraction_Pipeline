"""A dataset whose pages are all of one type is YAML, and nothing in Python names one.

The receipts of the public SROIE test split were read, until 25 November 2026, by a task of their own:
``stages/sroie.py`` and the package ``common/sroie``, 1,445 lines, with a second copy of what the
pipeline has for every other page: a prompt (a Python string), the fields (a Python tuple), a parser, a
scorer, a loader of answers and a report. They are now a document type in the schema file, a prompt of
its name, a size of engine call and an entry under ``pipeline.datasets``, read by the stages there are.

Before the task was deleted its parser and the pipeline's were set to every reply of a run of each
model, 694 in all, and found the same value for every field; and the model was shown to be sent the
same, request for request.

A module that named a dataset would be the start of a second way to read one.
"""

import subprocess
from pathlib import Path

import pytest
import yaml

from common.stage_settings import TASKS

REPO = Path(__file__).resolve().parents[2]


def tracked(*folders: str) -> list[Path]:
    listed = subprocess.run(
        [
            "git",
            "ls-files",
            *[f"{folder}/*.py" for folder in folders],
            *[f"{folder}/**/*.py" for folder in folders],
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [REPO / name for name in sorted(set(listed))]


def datasets() -> list[str]:
    return list(yaml.safe_load((REPO / "config" / "run_config.yml").read_text())["pipeline"]["datasets"])


def test_the_shipped_file_has_a_dataset_to_look_for() -> None:
    assert "sroie" in datasets()


@pytest.mark.parametrize("name", datasets())
def test_nothing_in_python_names_a_dataset(name: str) -> None:
    named = [
        f"{path.relative_to(REPO)}:{number}: {line.strip()[:80]}"
        for path in tracked("stages", "common", "models")
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if name in line.lower()
    ]

    assert named == []


def test_the_task_that_read_the_sroie_receipts_is_gone() -> None:
    assert not (REPO / "stages" / "sroie.py").exists()
    assert not (REPO / "common" / "sroie").exists()
    assert TASKS == ("split", "classify", "extract", "clean", "evaluate")
