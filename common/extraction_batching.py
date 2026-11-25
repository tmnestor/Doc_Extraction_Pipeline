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
  ``sort_for_extraction``. Batching consecutive runs preserves that order exactly.

Every document type can share a call, a bank statement included.
"""

from typing import Any

# How many pages go to the engine in one call, for every document type and for classify. A page is one
# independent request, so the engine runs up to the model's max_num_seqs of a call's pages at once and
# queues the rest. The size is several times that, so that the engine stays full as a call's short
# pages finish: measured on 55 statements with 8 in flight, 16 to a call took 484 s and one call of 55
# took 418 s. A call that fails is tried again one page at a time, so a size too large costs time and
# never a result. Until 26 November 2026 each type had a size of its own in the settings file, and by
# 25 November all of them were this.
PAGES_PER_CALL = 64


def plan_extraction_batches(classifications: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    """Group classified documents into engine calls.

    Args:
        classifications: Classification records, already in the order they are read in. Each carries
            ``document_type``.
        size: The most documents to a call.

    Returns:
        Calls in input order, each holding one document type. Flattening the result returns the input
        unchanged, so nothing is dropped, duplicated or reordered.
    """
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_type: str | None = None

    for record in classifications:
        doc_type = record["document_type"]
        if doc_type != current_type or len(current) >= size:
            if current:
                batches.append(current)
            current, current_type = [], doc_type
        current.append(record)

    if current:
        batches.append(current)

    return batches
