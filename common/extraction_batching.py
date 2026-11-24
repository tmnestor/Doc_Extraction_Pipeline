"""Grouping classified documents into extraction batches.

Submitting several images in one ``engine.chat()`` call is what lets vLLM's
scheduler run them concurrently; calling generate() in a loop keeps exactly
one sequence in flight however ``max_num_seqs`` is set.

Two constraints shape the grouping, and both come from how the engine call
works rather than from taste:

* **One batch, one document type.** A call carries one completion budget, so
  every image in it shares a completion budget, and each type has its own
  prompt. classify runs first, so the type is already known here.
* **Consecutive runs only.** Classifications arrive sorted by
  ``sort_for_extraction`` (extraction_order, then secondary_sort), and the
  bank header cache depends on a bank's statements arriving together.
  Batching consecutive runs preserves that order exactly.

Every document type can share a call, a bank statement included.
"""

from pathlib import Path
from typing import Any

_SECTION = "pipeline.batching"

_CLASSIFY_KEY = "classify"

_EXAMPLE = """\
pipeline:
  batching:
    classify: 64
    receipt: 4
    invoice: 2
    bank_statement: 2
    default: 2"""


class BatchingConfigError(ValueError):
    """The pipeline.batching block is missing or invalid."""


def read_batch_sizes(raw_config: dict[str, Any], *, config_path: Path) -> dict[str, int]:
    """Read per-document-type batch sizes from the raw run_config mapping.

    Args:
        raw_config: Parsed run_config.yml contents.
        config_path: Path to that file, quoted in diagnostics.

    Returns:
        Batch size by lowercase document type, including ``default``.

    Raises:
        BatchingConfigError: If the block is missing, has no ``default``,
            or holds a value that is not a positive integer.
    """
    block = raw_config.get("pipeline", {}).get("batching")
    if not isinstance(block, dict):
        raise BatchingConfigError(
            f"What: the {_SECTION} configuration block is missing.\n"
            f"Where: {config_path}, under the {_SECTION} section.\n"
            f"Expected: a mapping of document type to batch size, including "
            f"a 'default' entry.\n"
            f"How to fix: add this block to {config_path}:\n\n{_EXAMPLE}"
        )

    if "default" not in block:
        raise BatchingConfigError(
            f"What: {_SECTION} has no 'default' entry, so an unlisted "
            f"document type has no batch size.\n"
            f"Where: {config_path}, under the {_SECTION} section.\n"
            f"Expected: default: a positive integer, e.g. 2\n"
            f"How to fix: add 'default: 2' to that block in {config_path}. "
            f"Set it to 1 to keep unlisted types unbatched."
        )

    from common.field_schema import get_field_schema

    types = set(get_field_schema().document_types)
    sizes = {}
    for key, value in block.items():
        if key == _CLASSIFY_KEY:
            continue  # pages per classify call, not a document type: read_classify_batch_size
        if key != "default" and str(key).lower() not in types:
            raise BatchingConfigError(
                f"What: {_SECTION}.{key} is not a document type, so its batch size would never be used.\n"
                f"Where: {config_path}, under the {_SECTION} section.\n"
                f"Expected: 'default', '{_CLASSIFY_KEY}', or a document type of the schema file: "
                f"{', '.join(sorted(types))}.\n"
                f"How to fix: correct the spelling of {key!r}, or give the type a block under "
                f"document_types in config/field_definitions.yaml."
            )
        # bool is an int subclass; a YAML `true` here is a config error.
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise BatchingConfigError(
                f"What: {_SECTION}.{key} is {value!r}, not a positive integer.\n"
                f"Where: {config_path}, under the {_SECTION} section.\n"
                f"Expected: {key}: a positive integer, e.g. 2. A size of 1 "
                f"means unbatched; 0 would produce no batches and extract "
                f"nothing.\n"
                f"How to fix: correct that value in {config_path}:\n\n{_EXAMPLE}"
            )
        sizes[str(key).lower()] = value

    return sizes


def read_classify_batch_size(raw_config: dict[str, Any], *, config_path: Path) -> int:
    """Read how many pages the classify stage submits in one engine call.

    Args:
        raw_config: Parsed run_config.yml contents.
        config_path: Path to that file, quoted in diagnostics.

    Returns:
        Pages per classify call. 1 means one call a page.

    Raises:
        BatchingConfigError: If the key is missing or not a positive integer.
    """
    block = raw_config.get("pipeline", {}).get("batching")
    value = block.get(_CLASSIFY_KEY) if isinstance(block, dict) else None
    # bool is an int subclass; a YAML `true` here is a config error.
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        found = "missing" if value is None else f"{value!r}, not a positive integer"
        raise BatchingConfigError(
            f"What: {_SECTION}.{_CLASSIFY_KEY} is {found}, so the classify stage "
            f"does not know how many pages to submit in one engine call.\n"
            f"Where: {config_path}, under the {_SECTION} section.\n"
            f"Expected: classify: 64 (a positive integer; 1 means one call a page).\n"
            f"How to fix: set that value in {config_path}:\n\n{_EXAMPLE}"
        )
    return value


def plan_extraction_batches(
    classifications: list[dict[str, Any]],
    *,
    batch_sizes: dict[str, int],
) -> list[list[dict[str, Any]]]:
    """Group classified documents into engine-call batches.

    Args:
        classifications: Classification records, already in extraction
            order. Each carries ``document_type``.
        batch_sizes: Documents per call, keyed by lowercase document type,
            with a ``default`` entry for unlisted types.

    Returns:
        Batches in input order, each holding one document type. Flattening
        the result returns the input unchanged, so nothing is dropped,
        duplicated or reordered.
    """
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_type: str | None = None

    for record in classifications:
        doc_type = record["document_type"]

        limit = batch_sizes.get(doc_type.lower(), batch_sizes["default"])

        if doc_type != current_type or len(current) >= limit:
            if current:
                batches.append(current)
            current, current_type = [], doc_type

        current.append(record)

    if current:
        batches.append(current)

    return batches
