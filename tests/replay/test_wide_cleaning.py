"""The clean stage makes of every recorded reply what it made when the record was written.

See `tests/replay/wide_cleaning.py` for what the replies are and how the record is made.
"""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("wide_cleaning", Path(__file__).with_name("wide_cleaning.py"))
assert spec is not None and spec.loader is not None
wide_cleaning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wide_cleaning)


def test_every_recorded_reply_is_cleaned_as_it_was() -> None:
    if not wide_cleaning.RECORD.is_file():
        pytest.skip(f"No record at {wide_cleaning.RECORD}: run tests/replay/wide_cleaning.py")
    recorded = json.loads(wide_cleaning.RECORD.read_text())

    now = wide_cleaning.cases()

    assert len(now) == len(recorded)
    differing = [new["image_name"] for new, old in zip(now, recorded, strict=True) if new != old]
    assert differing == []
