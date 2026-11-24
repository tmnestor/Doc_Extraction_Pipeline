"""Turning the model's replies to the SROIE receipts into a scored run.

The receipts are read in ``stages/sroie.py`` (``read_sroie``), which hands back the replies as the model
wrote them. Parsing happens here, once, however the engines were arranged.

A receipt that could not be read is recorded, not raised. Stopping at receipt 300 of 347 would discard the
work already done, and one receipt's error is not a fault of the data. The caller is expected to report a
run with errors as such, not as a clean measurement.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from common.sroie.ground_truth import SroieRecord
from common.sroie.parse import parse_sroie_response

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkRun:
    """Everything one pass over the split produced."""

    predictions: dict[str, dict[str, str]] = field(default_factory=dict)
    raw_responses: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


def benchmark_from_worker_responses(
    records: list[SroieRecord],
    responses: list[dict[str, Any]],
) -> BenchmarkRun:
    """Reassemble a run from data-parallel worker output.

    Parsing happens here and not where the receipts are read, so there is one parser.

    Args:
        records: Ground-truth records for the split, in scoring order.
        responses: Worker dicts carrying ``image_id``, ``raw_response``
            and ``error``.

    Returns:
        The run. A record absent from
        *responses* (a worker died part-way through its partition)
        becomes an explicit error, never a missing row.
    """
    by_id = {str(item["image_id"]): item for item in responses}
    run = BenchmarkRun()

    for record in records:
        item = by_id.get(record.image_id)
        if item is None:
            run.errors[record.image_id] = (
                "no response returned: the data-parallel worker holding this image did not report it"
            )
            run.raw_responses[record.image_id] = ""
            run.predictions[record.image_id] = {}
            continue

        error = item.get("error")
        response = item.get("raw_response") or ""
        run.raw_responses[record.image_id] = response
        if error:
            run.errors[record.image_id] = str(error)
            run.predictions[record.image_id] = {}
            continue
        run.predictions[record.image_id] = parse_sroie_response(response)

    return run
