"""The scorer gives every recorded case the score it gave when the record was written.

See `tests/replay/wide_scoring.py` for what the cases are and how the record is made.
"""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("wide_scoring", Path(__file__).with_name("wide_scoring.py"))
assert spec is not None and spec.loader is not None
wide_scoring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wide_scoring)


def test_every_recorded_case_scores_as_it_did() -> None:
    if not wide_scoring.RECORD.is_file():
        pytest.skip(f"No record at {wide_scoring.RECORD}: run tests/replay/wide_scoring.py")
    recorded = json.loads(wide_scoring.RECORD.read_text())

    fields, whole = wide_scoring.cases()
    now = json.loads(json.dumps({"fields": fields, "whole": whole}))

    assert len(now["fields"]) == len(recorded["fields"])
    differing = [
        (new["field"], new["read"], new["truth"])
        for new, old in zip(now["fields"], recorded["fields"], strict=True)
        if new != old
    ]
    assert differing == []
    assert now["whole"] == recorded["whole"]
