"""Stage 3: Parse and clean raw model responses (CPU).

Reads raw_extractions.jsonl, parses and cleans each raw
response to produce structured field dicts, and writes cleaned_extractions.jsonl.

No GPU needed -- this stage runs on CPU nodes.

Started by ``python -m stages`` with ``KFP_TASK`` naming it:

    KFP_TASK=clean output=<folder> bash entrypoint.sh
"""

import logging
from pathlib import Path
from typing import Any


from common.field_schema import NOT_FOUND, get_field_schema
from common.reply_fields import fields_from_reply

from common.stage_settings import StageSettings

from . import records as records_of
from .io import read_jsonl, start_afresh, write_jsonl

logger = logging.getLogger(__name__)


def run(
    raw_extractions_path: Path,
    output_path: Path,
) -> Path:
    """Parse, clean, and validate raw model responses.

    For each record in raw_extractions.jsonl:
      1. Look up the expected fields for the document type
      2. Parse and clean the reply (common.reply_fields.fields_from_reply)
      3. Write the cleaned field dict to cleaned_extractions.jsonl

    Args:
        raw_extractions_path: Path to raw_extractions.jsonl from Stage 2.
        output_path: Path to write cleaned_extractions.jsonl.

    Returns:
        Path to the written cleaned_extractions.jsonl.
    """
    # Read raw extractions
    raw_records = read_jsonl(raw_extractions_path)
    if not raw_records:
        msg = f"No records found in {raw_extractions_path}"
        raise FileNotFoundError(msg)

    logger.info("Read %d raw extractions from %s", len(raw_records), raw_extractions_path)

    schema = get_field_schema()

    # Build field lookup: doc_type -> list of expected fields
    all_fields = schema.get_all_doc_type_fields()

    cleaned_records: list[dict[str, Any]] = []

    for record in map(records_of.as_read, raw_records):
        image_name = record["image_name"]
        doc_type = record["document_type"]

        # A page the extract stage was set not to read (pipeline.extraction.skip_labels) has no reply.
        # It is passed on as not read. Until 25 November 2026 its empty reply was parsed like any
        # other, every field came out as not found, and the evaluate stage scored the page as all wrong.
        if records_of.not_read(record):
            cleaned_records.append(records_of.cleaned_but_unread(record))
            continue

        # A page the extract stage could not read has no reply either: its error is passed on.
        if records_of.failed(record):
            logger.warning("Skipping %s: extraction error: %s", image_name, record["error"])
            cleaned_records.append(records_of.cleaned_but_failed(record))
            continue

        # Resolve expected fields for this document type
        expected_fields = all_fields.get(doc_type.lower(), [])
        if not expected_fields:
            logger.warning("No field definitions for doc_type %r -- using empty list", doc_type)

        # Parse + clean + validate
        extracted_data = fields_from_reply(record.get("raw_response", ""), expected_fields)
        found_count = sum(1 for v in extracted_data.values() if v != NOT_FOUND)
        cleaned_records.append(
            records_of.cleaned(record, extracted_data, expected=len(expected_fields), found=found_count)
        )

        logger.debug(
            "%s: %d/%d fields extracted",
            image_name,
            found_count,
            len(expected_fields),
        )

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = write_jsonl(output_path, cleaned_records)
    logger.info("Wrote %d cleaned extractions to %s", count, output_path)

    # Summary
    total_fields = sum(r.get("field_count", 0) for r in cleaned_records)
    total_found = sum(r.get("extracted_fields_count", 0) for r in cleaned_records)
    error_count = sum(1 for r in cleaned_records if records_of.failed(r))
    logger.info(
        "Summary: %d records, %d/%d fields found, %d errors",
        len(cleaned_records),
        total_found,
        total_fields,
        error_count,
    )
    not_read = sum(1 for r in cleaned_records if records_of.not_read(r))
    if not_read:
        logger.info(
            "%d page(s) were not read, by pipeline.extraction.skip_labels, and are passed on as not read",
            not_read,
        )

    return output_path


BANNER = "Stage 3: clean, the model's replies parsed and tidied (CPU)"


def check(task: StageSettings) -> None:
    """Read and check everything the stage will use: the schema file. Nothing is written.

    Raises:
        ConfigError: With what is wrong, where, what is expected and how to mend it.
    """
    get_field_schema()


def start(task: StageSettings, checked: None) -> str:
    """Run the stage for a task, and give the line that says it is done."""
    start_afresh(task.clear_previous_output, task.cleaned_extractions)
    run(task.raw_extractions, task.cleaned_extractions)
    return "Cleaning complete."
