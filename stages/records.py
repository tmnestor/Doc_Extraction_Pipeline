"""The records the stages hand one another, each defined once.

A stage writes one JSON object a line, and the next stage reads them. Until 25 November 2026 the keys of
each record were spelt out wherever a record was made or read: by the stage that wrote it, by the stage
that read it, and again for each special case. One stage could then write a key another had never heard
of. That is how a page the extract stage was set not to read came to be scored as all wrong: extract
marked its record ``skipped``, and neither clean nor evaluate looked for the mark.

Here each record has one definition, the functions that make it, and the questions asked of it
(``failed``, ``not_read``). A stage makes a record with these functions and never with a literal of its
own. A record read back is given its type, so that a key spelt wrongly where it is read is found by the
type checker, and a test holds every function that makes a record to the record's definition.

The records are plain dictionaries when they are written: what goes into a file is what always did, key
for key and in the same order.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast


class Classified(TypedDict):
    """A page and its document type: a line of classifications.jsonl."""

    image_path: str
    image_name: str
    document_type: str
    # How the type was reached: one of ``common.classification.REACHED_BY``.
    type_from: str | None
    confidence: float
    raw_response: str
    prompt_used: str
    # Why the model could not be asked, or its reply not read. Absent when it could.
    error: NotRequired[str]
    processing_time: float
    # The card that read the page, with an engine on each card.
    gpu_id: NotRequired[int]


class Read(TypedDict):
    """A page and the model's reply about its fields: a line of raw_extractions.jsonl."""

    image_name: str
    image_path: str
    document_type: str
    raw_response: str
    processing_time: float
    # The prompt of the page's type; "error" for a page that could not be read, "skipped" for one the
    # stage was set not to read.
    prompt_used: str
    error: str | None
    gpu_id: NotRequired[int]
    # A page of a type named in pipeline.extraction.skip_labels, with why.
    skipped: NotRequired[bool]
    skip_reason: NotRequired[str]
    fields: NotRequired[dict[str, str]]


class Cleaned(TypedDict):
    """A page and its fields, parsed and tidied: a line of cleaned_extractions.jsonl."""

    image_name: str
    image_path: str
    document_type: str
    extracted_data: dict[str, str]
    field_count: int
    extracted_fields_count: int
    processing_time: float
    error: NotRequired[str]
    skipped: NotRequired[bool]
    skip_reason: NotRequired[str]


# --- what is asked of a record --------------------------------------------------------------------------


def failed(record: Mapping[str, Any]) -> bool:
    """Whether a record is of a page the stage could not read."""
    return bool(record.get("error"))


def not_read(record: Mapping[str, Any]) -> bool:
    """Whether a record is of a page the extract stage was set not to read."""
    return bool(record.get("skipped"))


# --- a record read back from a file ---------------------------------------------------------------------


def as_classified(record: dict[str, Any]) -> Classified:
    return cast(Classified, record)


def as_read(record: dict[str, Any]) -> Read:
    return cast(Read, record)


def as_cleaned(record: dict[str, Any]) -> Cleaned:
    return cast(Cleaned, record)


# --- classify -------------------------------------------------------------------------------------------


def classified(image_path: str, result: Mapping[str, Any], seconds: float) -> dict[str, Any]:
    """A page's classification record.

    Args:
        image_path: The page.
        result: What the document processor made of the model's reply: ``document_type``,
            ``type_from``, ``raw_response``, and an ``error`` if the model could not be asked.
        seconds: The page's share of the time its engine call took.
    """
    record: dict[str, Any] = {
        "image_path": image_path,
        "image_name": Path(image_path).name,
        "document_type": result["document_type"],
        "type_from": result.get("type_from"),
        "confidence": result.get("confidence", 1.0),
        "raw_response": result.get("raw_response", ""),
        "prompt_used": result.get("prompt_used", "detection"),
    }
    # The error and the time keep the places they have always had in the file.
    if result.get("error"):
        record["error"] = result["error"]
    record["processing_time"] = seconds
    return record


# --- extract --------------------------------------------------------------------------------------------


def read(
    page: Mapping[str, Any], *, reply: str | None, seconds: float, prompt: str, error: str | None
) -> dict[str, Any]:
    """The record of a page the extract stage asked the model about.

    Args:
        page: The page's classification record.
        reply: The model's reply, as it wrote it; None if the page could not be read.
        seconds: The page's share of the time its engine call took.
        prompt: The name of the prompt the page was read with.
        error: Why the page could not be read, or None.
    """
    record: Read = {
        "image_name": page["image_name"],
        "image_path": page["image_path"],
        "document_type": page["document_type"],
        "raw_response": reply if error is None and reply is not None else "",
        "processing_time": seconds,
        "prompt_used": prompt if error is None else "error",
        "error": error,
    }
    return cast(dict[str, Any], record)


def left_unread(page: Mapping[str, Any]) -> dict[str, Any]:
    """The record of a page the extract stage was set not to read, so that every page has a record."""
    record: Read = {
        "image_name": page.get("image_name", ""),
        "image_path": page.get("image_path", ""),
        "document_type": page.get("document_type", ""),
        "raw_response": "",
        "processing_time": 0.0,
        "prompt_used": "skipped",
        "error": None,
        "skipped": True,
        "skip_reason": page.get("document_type", ""),
        "fields": {},
    }
    return cast(dict[str, Any], record)


# --- clean ----------------------------------------------------------------------------------------------


def _cleaned(page: Read, fields: dict[str, str], expected: int, found: int) -> Cleaned:
    return {
        "image_name": page["image_name"],
        "image_path": page.get("image_path", ""),
        "document_type": page["document_type"],
        "extracted_data": fields,
        "field_count": expected,
        "extracted_fields_count": found,
        "processing_time": page.get("processing_time", 0.0),
    }


def cleaned(page: Read, fields: dict[str, str], *, expected: int, found: int) -> dict[str, Any]:
    """The record of a page whose reply was parsed and tidied.

    Args:
        page: The page's record from the extract stage.
        fields: The page's fields, cleaned.
        expected: How many fields the page's type has.
        found: How many of them the reply gave a value for.
    """
    return cast(dict[str, Any], _cleaned(page, fields, expected, found))


def cleaned_but_failed(page: Read) -> dict[str, Any]:
    """The record of a page the extract stage could not read: no fields, and why."""
    record = _cleaned(page, {}, 0, 0)
    record["error"] = page["error"] or ""
    return cast(dict[str, Any], record)


def cleaned_but_unread(page: Read) -> dict[str, Any]:
    """The record of a page the extract stage was set not to read: no fields, and marked as not read."""
    record = _cleaned(page, {}, 0, 0)
    record["skipped"] = True
    record["skip_reason"] = page.get("skip_reason", page["document_type"])
    return cast(dict[str, Any], record)


# --- evaluate -------------------------------------------------------------------------------------------


def scored(page: Cleaned, evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """The record of a page that was scored, or that had no answer in the key to be scored by."""
    return {"image_name": page["image_name"], "document_type": page["document_type"], **evaluation}


def scored_but_failed(page: Cleaned) -> dict[str, Any]:
    """The record of a page whose extraction failed: nothing to score, and why."""
    return {
        "image_name": page["image_name"],
        "document_type": page["document_type"],
        "error": page.get("error"),
        "overall_accuracy": 0,
        "median_f1": 0,
    }


def scored_but_unread(page: Cleaned) -> dict[str, Any]:
    """The record of a page the extract stage was set not to read: it is in no score."""
    return {
        "image_name": page["image_name"],
        "document_type": page["document_type"],
        "skipped": True,
        "skip_reason": page.get("skip_reason", page["document_type"]),
    }
