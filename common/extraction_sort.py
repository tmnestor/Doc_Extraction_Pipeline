"""The order pages are read in: one document type after another, so that the engine reuses its work on a prompt."""

from typing import Any


def sort_for_extraction(records: list[dict[str, Any]], type_order: list[str]) -> list[dict[str, Any]]:
    """The classification records with the types in ``type_order`` first, in that order.

    Within a type the records keep the order they came in, and a type that is not named comes last.
    Until 26 November 2026 a second sort, by the area of the page's image, could be asked for in the
    settings file; no run ever did.

    Args:
        records: Classification records, each with at least ``document_type``.
        type_order: The document types, the first to be read first.

    Returns:
        A new list: ``records`` is not changed.
    """
    place = {kind.lower(): position for position, kind in enumerate(type_order)}
    return sorted(
        records, key=lambda record: place.get(record.get("document_type", "").lower(), len(place))
    )


def filter_skip_labels(
    records: list[dict[str, Any]],
    skip_labels: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition records into (to_extract, skipped) based on skip labels.

    Args:
        records: Classification dicts with ``document_type`` key.
        skip_labels: Labels that should bypass extraction.

    Returns:
        Tuple of (records to process, records to skip).
    """
    if not skip_labels:
        return records, []

    skip_set = {label.lower() for label in skip_labels}
    to_extract: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for rec in records:
        doc_type = rec.get("document_type", "")
        if doc_type.lower() in skip_set:
            skipped.append(rec)
        else:
            to_extract.append(rec)

    return to_extract, skipped
