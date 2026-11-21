"""The document type read from every recorded classification reply is the one read when the record was written.

See `tests/replay/wide_classification.py` for what the replies are and how the record is made.
"""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "wide_classification", Path(__file__).with_name("wide_classification.py")
)
assert spec is not None and spec.loader is not None
wide_classification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wide_classification)


def test_every_recorded_reply_is_read_as_it_was() -> None:
    if not wide_classification.RECORD.is_file():
        pytest.skip(f"No record at {wide_classification.RECORD}: run tests/replay/wide_classification.py")
    recorded = json.loads(wide_classification.RECORD.read_text())

    now = json.loads(json.dumps(wide_classification.read_types()))
    # The record also holds which heading played which column role. That was worked out on the way to
    # the type and never written to a classification record, and is no longer returned.
    for case in recorded:
        case["read"].pop("column_mapping", None)

    assert len(now) == len(recorded)
    differing = [new["case"] for new, old in zip(now, recorded, strict=True) if new != old]
    assert differing == []
