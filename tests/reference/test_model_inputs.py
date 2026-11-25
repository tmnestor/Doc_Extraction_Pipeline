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


def as_asked(record: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    """A stage's calls with everything that is sent, and without what only says where in the record it is.

    The name of the stage is dropped, and so is the engine's running number, which counts the engines
    started so far and so differs between any two stages. In its place is what that engine was started
    with and the card it was put on.
    """
    return [
        {
            "engine": {
                key: value for key, value in record["engines"][call["engine"]].items() if key != "stage"
            },
            "sampling": call["sampling"],
            "others": call["others"],
            "requests": call["requests"],
        }
        for call in record["calls"]
        if call["stage"] == stage
    ]


@pytest.mark.parametrize("model", MODELS)
def test_the_dataset_s_receipts_are_asked_as_the_old_task_asks_them(model: str, tmp_path: Path) -> None:
    """The SROIE receipts read as a dataset, by classify and extract, beside the task that read them.

    The same engines with the same arguments on the same cards, the same receipts in the same calls in
    the same order, and for each the same prompt, image, completion budget, sampling and chat
    arguments. And the model is not asked what the receipts are: their dataset says.
    """
    record = json.loads(as_text(record_model_inputs(model, tmp_path)))

    new = as_asked(record, "dataset_sroie")
    assert new, "the dataset's receipts were not read"
    assert new == as_asked(record, "sroie")
    assert as_asked(record, "dataset_sroie_classify") == []
    assert not [engine for engine in record["engines"] if engine["stage"] == "dataset_sroie_classify"]


if __name__ == "__main__":
    for name in MODELS:
        with tempfile.TemporaryDirectory() as folder:
            recorded_path(name).write_text(as_text(record_model_inputs(name, Path(folder))))
        print(f"wrote {recorded_path(name).name}")
