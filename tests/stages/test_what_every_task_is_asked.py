"""Two things are asked of every task before it is checked: that its model is one that is registered,
and that a document type the job gives is a type of the schema file.

Both need the pipeline's own files, so they are asked in Python and not by the entry script. Until
26 November 2026 each was asked by some tasks and not by others: a split that passed its photographs
through took any name for a model and wrote it into the run's note, and a ``document_type`` that was no
type was refused by ``classify`` alone.

The pages are blank images made by the bench, and every reply is invented.
"""

from pathlib import Path

import pytest

from bench import ONE_ENGINE, Bench
from common.errors import ConfigError
from common.stage_settings import StageSettings, resolve
from stages import __main__ as starter
from stages.io import read_jsonl

PARTS = ("What:", "Where:", "Expected:", "How to fix:")
TASKS = ("split", "classify", "extract", "clean", "evaluate")


def job(bench: Bench, out: Path, task: str, **more: str) -> StageSettings:
    environ = {
        "model": ONE_ENGINE,
        "image_dir": str(bench.pages),
        "output": str(out),
        "ground_truth": str(out.parent / "ground_truth.jsonl"),
        "DOC_EXTRACT_MODELS_DIR": str(bench.models),
        **more,
    }
    return resolve(task, environ)


@pytest.mark.parametrize("task", TASKS)
def test_a_document_type_that_is_no_type_is_refused_by_every_task(
    bench: Bench, tmp_path: Path, task: str
) -> None:
    with pytest.raises(ConfigError) as refused:
        starter.run_task(job(bench, tmp_path / "out", task, document_type="LEDGER"))

    message = str(refused.value)
    assert all(part in message for part in PARTS)
    assert "LEDGER" in message and "document_type" in message and "sroie_receipt" in message
    assert not (tmp_path / "out").exists() and bench.engines_started == 0


@pytest.mark.parametrize(
    ("task", "more"),
    [
        ("split", {}),
        ("split", {"photo_split": "true"}),
        ("classify", {"document_type": "RECEIPT"}),
        ("classify", {}),
        ("extract", {}),
        ("clean", {}),
    ],
    ids=[
        "split passing through",
        "split asking",
        "classify given the type",
        "classify",
        "extract",
        "clean",
    ],
)
def test_a_model_that_is_not_registered_is_refused_by_every_task_given_one(
    bench: Bench, tmp_path: Path, task: str, more: dict[str, str]
) -> None:
    with pytest.raises(ConfigError) as refused:
        starter.run_task(job(bench, tmp_path / "out", task, model="no-such-model", **more))

    message = str(refused.value)
    assert "no-such-model" in message and ONE_ENGINE in message
    assert not (tmp_path / "out").exists() and bench.engines_started == 0


def test_clean_and_evaluate_are_given_no_model_and_asked_nothing_about_one(
    bench: Bench, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    for task in ("classify", "extract"):
        starter.run_task(job(bench, out, task))

    starter.run_task(resolve("clean", {"output": str(out)}))

    assert len(read_jsonl(out / "cleaned_extractions.jsonl")) == len(bench.names)


def test_classify_given_the_type_of_every_page_needs_no_weights(bench: Bench, tmp_path: Path) -> None:
    """It starts no engine, so it looks for no model's folder: none is named here at all."""
    out = tmp_path / "out"
    environ = {
        "model": ONE_ENGINE,
        "image_dir": str(bench.pages),
        "output": str(out),
        "document_type": "receipt",
    }

    starter.run_task(resolve("classify", environ))

    records = read_jsonl(out / "classifications.jsonl")
    assert [record["document_type"] for record in records] == ["receipt"] * len(bench.names)
    assert bench.engines_started == 0
