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
from .io import read_input, start_afresh, write_jsonl

logger = logging.getLogger(__name__)


def run(raw_records: list[dict[str, Any]], output_path: Path) -> Path:
    """Parse, clean, and validate raw model responses.

    For each record in raw_extractions.jsonl:
      1. Look up the expected fields for the document type
      2. Parse and clean the reply (common.reply_fields.fields_from_reply)
      3. Write the cleaned field dict to cleaned_extractions.jsonl

    Args:
        raw_records: The records of raw_extractions.jsonl from Stage 2: what ``checked`` gave.
        output_path: Path to write cleaned_extractions.jsonl.

    Returns:
        Path to the written cleaned_extractions.jsonl.
    """
    schema = get_field_schema()

    # Build field lookup: doc_type -> list of expected fields
    all_fields = schema.get_all_doc_type_fields()

    cleaned_records: list[dict[str, Any]] = []

    for record in map(records_of.as_read, raw_records):
        image_name = record["image_name"]
        doc_type = record["document_type"]

        # A page the extract stage could not read has no reply: its error is passed on.
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
    return output_path


BANNER = "Stage 3: clean, the model's replies parsed and tidied (CPU)"


def checked(raw_extractions_path: Path) -> list[dict[str, Any]]:
    """The model's replies, read, and the schema file checked.

    The one place the stage reads what it is run with. What it gives is what ``run`` is handed.

    Raises:
        ConfigError: With what is wrong, where, what is expected and how to mend it. A file of
            replies that is not there, or cannot be read, is one such: until 26 November 2026 it ended
            the task with a traceback.
    """
    get_field_schema()
    raw_records = read_input(raw_extractions_path, written_by="extract")
    logger.info("Read %d raw extractions from %s", len(raw_records), raw_extractions_path)
    return raw_records


def check(task: StageSettings) -> list[dict[str, Any]]:
    """Read and check everything the stage will use, and give it for ``start``. Nothing is written.

    Raises:
        ConfigError: With what is wrong, where, what is expected and how to mend it.
    """
    return checked(task.raw_extractions)


def start(task: StageSettings, checked: list[dict[str, Any]]) -> str:
    """Run the stage for a task with what ``check`` gave, and give the line that says it is done."""
    start_afresh(task.clear_previous_output, task.cleaned_extractions)
    run(checked, task.cleaned_extractions)
    return "Cleaning complete."
