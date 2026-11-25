"""What each stage hands to the model must not change unless a change is meant.

`model_inputs.py` runs `classify`, `extract` and `split` on invented pages, and `classify` and `extract`
again on a dataset whose pages are all of one type, with a stand-in for the
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


# The first twelve characters of the SHA-256 of the dataset's prompt, which is how the record names a prompt
# (tests/common/test_the_sroie_receipts_are_a_type_in_yaml.py holds the whole digest).
THE_DATASETS_PROMPT = "4faf339e27dd"


@pytest.mark.parametrize("model", MODELS)
def test_a_dataset_s_pages_are_asked_by_their_type_s_prompt_and_the_model_is_not_asked_what_they_are(
    model: str, tmp_path: Path
) -> None:
    """The receipts of the dataset: one request each, by the prompt of their type and its fixed budget.

    Until the task that read these receipts was deleted, this record held its requests too, and a test
    set the two beside each other: the same engines with the same arguments on the same cards, the same
    receipts in the same order, and for each the same prompt, image, completion budget, sampling and
    chat arguments. What is pinned here is what that showed.
    """
    record = json.loads(as_text(record_model_inputs(model, tmp_path)))

    calls = [call for call in record["calls"] if call["stage"] == "dataset_sroie"]
    requests = [request for call in calls for request in call["requests"]]
    assert len(requests) == 12
    assert all(call["sampling"] == {"max_tokens": 256, "temperature": 0} for call in calls)
    assert all(
        [(role, kind) for role, kind, _ in request] == [("user", "text"), ("user", "image_pil")]
        for request in requests
    )
    assert {request[0][2] for request in requests} == {THE_DATASETS_PROMPT}
    assert record["prompts"][THE_DATASETS_PROMPT].startswith(
        "Read this scanned receipt and report four fields."
    )
    # Each receipt once, in the order of its name, however the engines are arranged.
    assert sorted(request[1][2] for request in requests) == [f"image {100 + n}x150 RGB" for n in range(12)]
    # Their dataset says what they are: classify starts no engine and asks nothing.
    assert not [call for call in record["calls"] if call["stage"] == "dataset_sroie_classify"]
    assert not [engine for engine in record["engines"] if engine["stage"] == "dataset_sroie_classify"]


if __name__ == "__main__":
    for name in MODELS:
        with tempfile.TemporaryDirectory() as folder:
            recorded_path(name).write_text(as_text(record_model_inputs(name, Path(folder))))
        print(f"wrote {recorded_path(name).name}")
