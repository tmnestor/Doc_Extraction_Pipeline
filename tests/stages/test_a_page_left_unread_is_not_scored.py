"""A page the extract stage was set not to read is not scored as if it had been read and got nothing.

``pipeline.extraction.skip_labels`` names document types whose pages are not read. The extract stage
writes a record for each, with no reply, so that every page has a record. Until 25 November 2026 the
clean stage parsed that empty reply like any other, every field came out as not found, and the evaluate
stage scored the page against its answer as all wrong: a setting that asked for pages to be left alone
lowered the mean.

Now the page is passed on as not read, and is in no score.

The pages are blank images made by the bench. All names, amounts and replies are invented.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from bench import Bench
from stages import clean, evaluate
from stages.io import read_jsonl

GOOD = {"SUPPLIER_NAME": "Quillfeather Books", "TOTAL_AMOUNT": "$31.07"}


def raw(name: str, reply: str = "", **more: Any) -> dict[str, Any]:
    return {
        "image_name": name,
        "image_path": f"/pages/{name}",
        "document_type": "INVOICE",
        "raw_response": reply,
        "processing_time": 1.5,
        "prompt_used": "invoice",
        "error": None,
        **more,
    }


def not_read(name: str) -> dict[str, Any]:
    """The record the extract stage writes for a page it was set not to read."""
    return {
        **raw(name, processing_time=0.0, prompt_used="skipped"),
        "document_type": "TRAVEL",
        "skipped": True,
        "skip_reason": "TRAVEL",
        "fields": {},
    }


def cleaned_from(tmp_path: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    extractions = tmp_path / "raw_extractions.jsonl"
    extractions.write_text("".join(json.dumps(record) + "\n" for record in records))
    return read_jsonl(clean.run(extractions, tmp_path / "cleaned_extractions.jsonl"))


# --- the clean stage ------------------------------------------------------------------------------------


def test_a_page_left_unread_is_passed_on_as_not_read_and_not_as_fields_not_found(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        read, unread = cleaned_from(
            tmp_path, [raw("page_1.png", "SUPPLIER_NAME: Quillfeather Books"), not_read("page_2.png")]
        )

    assert unread["skipped"] is True and unread["skip_reason"] == "TRAVEL"
    assert unread["extracted_data"] == {} and unread["field_count"] == 0
    assert "error" not in unread
    # The page that was read is cleaned as it always was.
    assert "skipped" not in read and read["extracted_data"]["SUPPLIER_NAME"] == "Quillfeather Books"
    assert "1 page(s) were not read, by pipeline.extraction.skip_labels" in caplog.text


def test_a_run_with_no_page_left_unread_says_nothing_of_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        (read,) = cleaned_from(tmp_path, [raw("page_1.png", "SUPPLIER_NAME: Quillfeather Books")])

    assert "skipped" not in read
    assert "were not read" not in caplog.text


# --- the evaluate stage ---------------------------------------------------------------------------------


def evaluated(tmp_path: Path, records: list[dict[str, Any]], answered: list[str]) -> Path:
    extractions = tmp_path / "cleaned_extractions.jsonl"
    extractions.write_text("".join(json.dumps(record) + "\n" for record in records))
    key = tmp_path / "ground_truth.jsonl"
    key.write_text("".join(json.dumps({"filename": name, **GOOD}) + "\n" for name in answered))
    return evaluate.run(extractions, key, tmp_path / "evaluation")


def cleaned(name: str, fields: dict[str, str]) -> dict[str, Any]:
    return {
        "image_name": name,
        "image_path": f"/pages/{name}",
        "document_type": "INVOICE",
        "extracted_data": fields,
        "processing_time": 1.0,
    }


def unread_cleaned(name: str) -> dict[str, Any]:
    return {**cleaned(name, {}), "document_type": "TRAVEL", "skipped": True, "skip_reason": "TRAVEL"}


def test_a_page_left_unread_is_in_no_score_though_the_key_has_an_answer_for_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    records = [cleaned("page_1.png", GOOD), unread_cleaned("page_2.png"), cleaned("page_3.png", GOOD)]

    with caplog.at_level(logging.INFO):
        written = evaluated(tmp_path, records, answered=["page_1.png", "page_2.png", "page_3.png"])

    said = "\n".join(record.getMessage() for record in caplog.records)
    assert "2 of 3 images scored" in said and "avg F1 (mean)=1.000" in said
    assert "1 page(s) were not read, by pipeline.extraction.skip_labels" in said and "page_2.png" in said
    lines = [json.loads(line) for line in written.read_text().splitlines()]
    assert [line["image_name"] for line in lines] == ["page_1.png", "page_2.png", "page_3.png"]
    assert lines[1] == {
        "image_name": "page_2.png",
        "document_type": "TRAVEL",
        "skipped": True,
        "skip_reason": "TRAVEL",
    }
    assert lines[0]["median_f1"] == 1.0 and "skipped" not in lines[0]


def test_pages_left_unread_have_a_row_of_their_own_in_the_summary() -> None:
    results: list[dict[str, Any]] = [
        {"median_f1": 1.0, "overall_accuracy": 1.0},
        {"skipped": True, "skip_reason": "TRAVEL"},
        {"skipped": True, "skip_reason": "TRAVEL"},
    ]

    scores = evaluate.headline(results, failed=0, unanswered=0, not_read=2)

    assert (scores.pages, scores.scored, scores.not_read) == (3, 1, 2)
    assert scores.mean_f1 == scores.mean_f1_counting_failures == 1.0
    assert dict(evaluate.score_rows(scores))["Not Read (skip_labels)"] == "2 (not scored)"
    assert evaluate.score_rows(scores)[0] == ("Images Scored", "1 of 3")


def test_a_run_with_no_page_left_unread_has_no_such_row() -> None:
    scores = evaluate.headline(
        [{"median_f1": 1.0, "overall_accuracy": 1.0}], failed=0, unanswered=0, not_read=0
    )

    assert "Not Read (skip_labels)" not in dict(evaluate.score_rows(scores))


# --- from the extract stage to the scores ---------------------------------------------------------------


def test_from_extract_to_evaluate_a_page_left_unread_lowers_no_score(bench: Bench, tmp_path: Path) -> None:
    """The three stages together, as a run makes them: extract writes the record the others must read."""
    bench.with_settings(lambda settings: settings["pipeline"]["extraction"].update(skip_labels=["RECEIPT"]))
    out = tmp_path / "out"
    out.mkdir()
    records = []
    for name in bench.names:
        kind = next(k for k in ("bank_statement", "invoice", "receipt") if k in name).upper()
        records.append({"image_name": name, "image_path": str(bench.pages / name), "document_type": kind})
    classifications = out / "classifications.jsonl"
    classifications.write_text("".join(json.dumps(record) + "\n" for record in records))

    def replies(image_paths: list[str], document_type: str) -> list[str]:
        return ["SUPPLIER_NAME: Quillfeather Books\nTOTAL_AMOUNT: $31.07"] * len(image_paths)

    bench.processor.read_pages = replies  # type: ignore[method-assign]

    raw_path = bench.extract(classifications, out / "raw_extractions.jsonl")
    cleaned_path = clean.run(raw_path, out / "cleaned_extractions.jsonl")
    key = out / "ground_truth.jsonl"
    key.write_text("".join(json.dumps({"filename": name, **GOOD}) + "\n" for name in bench.names))
    results = read_jsonl(evaluate.run(cleaned_path, key, out / "evaluation"))

    receipts = [name for name in bench.names if "receipt" in name]
    by_name = {record["image_name"]: record for record in results}
    assert len(results) == len(bench.names)
    assert all(by_name[name]["skipped"] is True for name in receipts)
    assert all("median_f1" not in by_name[name] for name in receipts)
    # The four pages that were read are scored, and the invoices, whose fields the reply holds, in full.
    read = [by_name[name] for name in bench.names if name not in receipts]
    assert len(read) == 4 and all("median_f1" in record and "skipped" not in record for record in read)
    assert all(by_name[name]["overall_accuracy"] == 1.0 for name in bench.names if "invoice" in name)
