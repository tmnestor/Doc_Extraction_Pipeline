"""Each record one stage hands another is defined once, and made only by the functions beside its definition.

Until 25 November 2026 the keys of a record were spelt out wherever one was made or read: by the stage
that wrote it, by the stage that read it, and again for each special case. One stage could write a key
another had never heard of, which is how a page the extract stage was set not to read came to be scored
as all wrong.

The records are plain dictionaries when written, so these tests hold the functions that make them to the
definitions, key for key, and to the order the keys have always had in the files.

All names, amounts and replies are invented.
"""

import re
from pathlib import Path
from typing import Any, get_type_hints

import pytest

from stages import records

REPO = Path(__file__).resolve().parents[2]

PAGE = {"image_name": "page_1.png", "image_path": "/pages/page_1.png", "document_type": "INVOICE"}
ASKED = {"document_type": "INVOICE", "type_from": "rule", "confidence": 1.0, "raw_response": "1. NONE"}


def made() -> dict[str, tuple[Any, dict[str, Any]]]:
    """One of every record a stage makes, with the definition it must keep to."""
    read = records.as_read(
        records.read(PAGE, reply="TOTAL_AMOUNT: $31.07", seconds=1.5, prompt="invoice", error=None)
    )
    could_not = records.as_read(
        records.read(PAGE, reply=None, seconds=0.2, prompt="invoice", error="the engine refused it")
    )
    unread = records.as_read(records.left_unread(PAGE))
    return {
        "classified": (records.Classified, records.classified("/pages/page_1.png", ASKED, 0.4)),
        "classified, not asked": (
            records.Classified,
            records.classified("/pages/page_1.png", {**ASKED, "error": "refused"}, 0.4),
        ),
        "read": (records.Read, dict(read)),
        "read, failed": (records.Read, dict(could_not)),
        "read, left unread": (records.Read, dict(unread)),
        "cleaned": (
            records.Cleaned,
            records.cleaned(read, {"TOTAL_AMOUNT": "$31.07"}, expected=14, found=1),
        ),
        "cleaned, failed": (records.Cleaned, records.cleaned_but_failed(could_not)),
        "cleaned, left unread": (records.Cleaned, records.cleaned_but_unread(unread)),
    }


@pytest.mark.parametrize("which", list(made()))
def test_a_record_made_holds_only_the_keys_of_its_definition_and_every_one_it_must(which: str) -> None:
    definition, record = made()[which]

    assert set(record) <= set(get_type_hints(definition)), "a key the definition does not have"
    assert set(definition.__required_keys__) <= set(record), "a key the definition requires is missing"


def test_the_keys_of_each_record_are_in_the_order_the_files_have_always_had() -> None:
    """A run writes what it wrote before, to the byte: the order of the keys is part of that."""
    order = {which: list(record) for which, (_, record) in made().items()}

    assert order["classified"] == [
        "image_path",
        "image_name",
        "document_type",
        "type_from",
        "confidence",
        "raw_response",
        "prompt_used",
        "processing_time",
    ]
    assert order["classified, not asked"][-3:] == ["prompt_used", "error", "processing_time"]
    assert (
        order["read"]
        == order["read, failed"]
        == [
            "image_name",
            "image_path",
            "document_type",
            "raw_response",
            "processing_time",
            "prompt_used",
            "error",
        ]
    )
    assert order["read, left unread"] == [*order["read"], "skipped", "skip_reason", "fields"]
    assert order["cleaned"] == [
        "image_name",
        "image_path",
        "document_type",
        "extracted_data",
        "field_count",
        "extracted_fields_count",
        "processing_time",
    ]
    assert order["cleaned, failed"] == [*order["cleaned"], "error"]
    assert order["cleaned, left unread"] == [*order["cleaned"], "skipped", "skip_reason"]


def test_a_page_that_could_not_be_read_has_no_reply_and_says_why() -> None:
    record = records.read(PAGE, reply=None, seconds=0.2, prompt="invoice", error="the engine refused it")

    assert (record["raw_response"], record["prompt_used"], record["error"]) == (
        "",
        "error",
        "the engine refused it",
    )
    assert records.failed(record) and not records.not_read(record)


def test_a_page_left_unread_is_marked_from_extract_to_evaluate() -> None:
    """The mark one stage writes is the mark the next two look for: all three ask the one question."""
    unread = records.left_unread(PAGE)
    cleaned = records.cleaned_but_unread(records.as_read(unread))
    scored = records.scored_but_unread(records.as_cleaned(cleaned))

    assert records.not_read(unread) and records.not_read(cleaned) and records.not_read(scored)
    assert not records.failed(unread) and not records.failed(cleaned)
    assert scored == {
        "image_name": "page_1.png",
        "document_type": "INVOICE",
        "skipped": True,
        "skip_reason": "INVOICE",
    }


def test_a_page_that_failed_is_marked_from_extract_to_evaluate() -> None:
    could_not = records.read(PAGE, reply=None, seconds=0.2, prompt="invoice", error="refused")
    cleaned = records.cleaned_but_failed(records.as_read(could_not))
    scored = records.scored_but_failed(records.as_cleaned(cleaned))

    assert records.failed(could_not) and records.failed(cleaned) and records.failed(scored)
    assert cleaned["extracted_data"] == {} and cleaned["error"] == "refused"
    assert scored == {
        "image_name": "page_1.png",
        "document_type": "INVOICE",
        "error": "refused",
        "overall_accuracy": 0,
        "median_f1": 0,
    }


def test_a_scored_page_carries_its_name_its_type_and_its_scores() -> None:
    page = records.as_cleaned(
        records.cleaned(records.as_read(PAGE | {"processing_time": 1.0}), {}, expected=0, found=0)  # type: ignore[arg-type]
    )

    assert records.scored(page, {"median_f1": 1.0, "overall_accuracy": 0.5}) == {
        "image_name": "page_1.png",
        "document_type": "INVOICE",
        "median_f1": 1.0,
        "overall_accuracy": 0.5,
    }


@pytest.mark.parametrize("stage", ["classify", "extract", "clean", "evaluate"])
def test_no_stage_makes_a_record_between_stages_with_a_literal_of_its_own(stage: str) -> None:
    """A record's keys written out in a stage are keys that can drift from the next stage's."""
    source = (REPO / "stages" / f"{stage}.py").read_text()
    code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))

    assert re.findall(r'"(image_name|image_path|raw_response|extracted_data|skip_reason)":', code) == []
    assert 'get("skipped")' not in code and 'get("error")' not in code
