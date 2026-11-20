"""What each stage hands to the model must not change unless a change is meant.

`model_inputs.py` runs `classify`, `extract`, `split` and `sroie` on invented pages with a stand-in for the
engine, and returns what the engine was given: the arguments each engine was built with and the cards it
was put on, and for every call its requests in order, its sampling settings and its chat arguments. The
record for each model the pipeline runs is kept beside this file.

A change to the pipeline that is meant to alter no result must leave these records as they are. They are
not regenerated to make a test pass. When a change to what the model is sent is intended, the record is
written again on purpose, with `python tests/reference/test_model_inputs.py`, and the difference is read
in review like any other.
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from model_inputs import record_model_inputs  # noqa: E402

MODELS = ("gemma4-31b-w4a16-vllm", "gemma4-12b-unified-w4a16-vllm")


def recorded_path(model: str) -> Path:
    return HERE / f"model_inputs_{model}.json"


def as_text(record: dict[str, Any]) -> str:
    return json.dumps(record, indent=2, sort_keys=True) + "\n"


@pytest.mark.parametrize("model", MODELS)
def test_what_is_sent_to_the_model_has_not_changed(model: str, tmp_path: Path) -> None:
    recorded = json.loads(recorded_path(model).read_text())

    now = json.loads(as_text(record_model_inputs(model, tmp_path)))

    assert now["engines"] == recorded["engines"]
    assert now["prompts"] == recorded["prompts"]
    assert now["calls"] == recorded["calls"]


if __name__ == "__main__":
    for name in MODELS:
        with tempfile.TemporaryDirectory() as folder:
            recorded_path(name).write_text(as_text(record_model_inputs(name, Path(folder))))
        print(f"wrote {recorded_path(name).name}")
