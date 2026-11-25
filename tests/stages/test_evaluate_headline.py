"""The evaluate stage says how many pages its scores leave out.

Found 24 November 2026. A page whose extraction failed, and a page with no answer in the key, were left
out of the mean and the median, and nothing in the summary said how many there were. A run that lost
twenty pages reported the mean of the rest as if it were the run's.

All names and amounts are invented.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from stages import evaluate

GOOD = {"SUPPLIER_NAME": "Quillfeather Books", "TOTAL_AMOUNT": "$31.07"}
HALF = {"SUPPLIER_NAME": "Quillfeather Books", "TOTAL_AMOUNT": "$99.99"}


def cleaned(name: str, fields: dict[str, str] | None = None, error: str | None = None) -> dict:
    record = {
        "image_name": name,
        "image_path": f"/pages/{name}",
        "document_type": "INVOICE",
        "extracted_data": fields or {},
        "processing_time": 1.0,
    }
    if error:
        record["error"] = error
    return record


def run_on(tmp_path: Path, records: list[dict], answered: list[str]) -> Path:
    extractions = tmp_path / "cleaned_extractions.jsonl"
    extractions.write_text("".join(json.dumps(record) + "\n" for record in records))
    key = tmp_path / "ground_truth.jsonl"
    key.write_text("".join(json.dumps({"filename": name, **GOOD}) + "\n" for name in answered))
    return evaluate.run(evaluate.checked(extractions, key), tmp_path / "evaluation")


def test_a_run_with_every_page_scored_has_no_row_for_pages_left_out() -> None:
    results = [{"median_f1": 1.0, "overall_accuracy": 1.0}, {"median_f1": 0.5, "overall_accuracy": 0.5}]

    scores = evaluate.headline(results, failed=0, unanswered=0, not_read=0)

    assert (scores.pages, scores.scored, scores.failed, scores.unanswered) == (2, 2, 0, 0)
    assert scores.mean_f1 == scores.mean_f1_counting_failures == 0.75
    assert [metric for metric, _ in evaluate.score_rows(scores)] == [
        "Images Scored",
        "Avg F1 (mean)",
        "Avg F1 (median)",
    ]
    assert evaluate.score_rows(scores)[0] == ("Images Scored", "2 of 2")


def test_failed_pages_are_counted_and_the_mean_is_also_given_with_each_as_nothing() -> None:
    results: list[dict[str, Any]] = [
        {"median_f1": 1.0, "overall_accuracy": 1.0},
        {"median_f1": 0.5, "overall_accuracy": 0.5},
        {"median_f1": 0, "overall_accuracy": 0, "error": "the engine ran out of memory"},
        {"median_f1": 0, "overall_accuracy": 0, "error": "the engine ran out of memory"},
    ]

    scores = evaluate.headline(results, failed=2, unanswered=0, not_read=0)

    assert (scores.pages, scores.scored, scores.failed) == (4, 2, 2)
    # The mean over the scored pages is what it was; the failed pages are beside it, not hidden in it.
    assert scores.mean_f1 == 0.75
    assert scores.mean_f1_counting_failures == 0.375
    rows = dict(evaluate.score_rows(scores))
    assert rows["Images Scored"] == "2 of 4"
    assert rows["Failed Extraction"].startswith("2 ")
    assert rows["Avg F1 (mean), failed as 0"] == "0.375"


def test_a_page_with_no_answer_is_counted_and_is_in_neither_mean() -> None:
    results: list[dict[str, Any]] = [
        {"median_f1": 1.0, "overall_accuracy": 1.0},
        {"error": "No ground truth for page_2.png", "overall_accuracy": 0},
    ]

    scores = evaluate.headline(results, failed=0, unanswered=1, not_read=0)

    assert (scores.pages, scores.scored, scores.unanswered) == (2, 1, 1)
    assert scores.mean_f1 == scores.mean_f1_counting_failures == 1.0
    rows = dict(evaluate.score_rows(scores))
    assert rows["Images Scored"] == "1 of 2"
    assert rows["No Answer In Key"].startswith("1 ")
    assert "Avg F1 (mean), failed as 0" not in rows


def test_a_run_with_nothing_scored_gives_no_means() -> None:
    scores = evaluate.headline(
        [{"error": "x", "median_f1": 0, "overall_accuracy": 0}], failed=1, unanswered=0, not_read=0
    )

    assert scores.scored == 0 and scores.mean_f1_counting_failures == 0.0
    assert "Avg F1 (mean)" not in dict(evaluate.score_rows(scores))


def test_the_stage_names_the_pages_it_could_not_score(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    records = [
        cleaned("page_1.png", GOOD),
        cleaned("page_2.png", HALF),
        cleaned("page_3.png", error="the engine ran out of memory"),
        cleaned("page_4.png", GOOD),
    ]

    with caplog.at_level(logging.INFO):
        written = run_on(tmp_path, records, answered=["page_1.png", "page_2.png", "page_3.png"])

    said = "\n".join(record.getMessage() for record in caplog.records)
    assert "2 of 4 images scored" in said
    assert "1 page(s) failed extraction" in said and "page_3.png" in said
    assert "1 page(s) have no answer" in said and "page_4.png" in said
    # The records are what they were: the stage counts the pages left out, it does not rewrite them.
    lines = [json.loads(line) for line in written.read_text().splitlines()]
    assert [line["image_name"] for line in lines] == [record["image_name"] for record in records]
    assert lines[2]["error"] == "the engine ran out of memory" and lines[2]["median_f1"] == 0
    assert lines[3]["error"] == "No ground truth for page_4.png" and "median_f1" not in lines[3]


def test_many_pages_left_out_are_named_in_part() -> None:
    names = [f"page_{number}.png" for number in range(14)]

    assert evaluate._some(names).endswith("page_9.png and 4 more")
    assert evaluate._some(names[:3]) == "page_0.png, page_1.png, page_2.png"
