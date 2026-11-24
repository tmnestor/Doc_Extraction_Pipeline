"""The stages that use no model must reproduce their recorded output exactly.

`clean` and `evaluate` need no GPU. So the model's stored replies from real runs can be put through them
on any machine, and what comes out compared with what came out before. A change to the pipeline that is
meant to alter no result must leave this output the same to the byte.

The recorded output lives outside the repository, because it is made from evaluation data. It is written
once by `tests/replay/make_reference.py`, from code known to be good. Where it is absent these tests are
skipped and say so.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from stages import clean, evaluate

DATA = Path.home() / "Desktop" / "evaluation_data"
REFERENCE = DATA / "refactor_reference_20261120"
CLEAN_PAGES_KEY = DATA / "snapshot_20261021" / "plain_20261021" / "ground_truth.jsonl"
CRUMPLED_KEY = DATA / "crumpled_20261021" / "ground_truth.jsonl"

# Run folder under DATA, and the answer key its pages are scored against.
RUNS: dict[str, Path] = {
    "output_gemma31b_all165_tensor_split_20261120": CLEAN_PAGES_KEY,
    "output_gemma12b_w4a16_all165_batched_20261120": CLEAN_PAGES_KEY,
    "output_gemma31b_crumpled_split_step_20261120": CRUMPLED_KEY,
}
NOT_COMPARED = ("missing_fields", "incorrect_fields")
"""Two lists in a page's record that are no longer what the reference holds.

When the reference was written they came from an older, looser comparison than the scores. From
21 November 2026 they follow the scores (see `tests/common/test_scoring.py`). Everything else in the
output, every score included, is compared exactly.
"""


def replay(run: str, into: Path) -> None:
    """Put a run's recorded model replies through `clean` and then `evaluate`.

    Args:
        run: Name of the run's folder under the reference.
        into: Empty folder to write `cleaned_extractions.jsonl` and `evaluation_results.jsonl` into.
    """
    raw = REFERENCE / run / "raw_extractions.jsonl"
    cleaned = clean.run(raw, into / "cleaned_extractions.jsonl")
    evaluate.run(cleaned, RUNS[run], into)


def comparable(path: Path) -> list[dict[str, Any]]:
    """A stage's output file as records, for comparing one replay with another."""
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        for name in NOT_COMPARED:
            record.pop(name, None)
    return records


def reference_or_skip(run: str) -> Path:
    folder = REFERENCE / run
    if not (folder / "evaluation_results.jsonl").is_file():
        pytest.skip(f"No reference at {folder}: run tests/replay/make_reference.py")
    return folder


@pytest.mark.parametrize("run", RUNS)
def test_clean_and_evaluate_reproduce_the_reference(run: str, tmp_path: Path) -> None:
    reference = reference_or_skip(run)

    replay(run, into=tmp_path)

    cleaned = "cleaned_extractions.jsonl"
    assert (tmp_path / cleaned).read_bytes() == (reference / cleaned).read_bytes()
    scored = "evaluation_results.jsonl"
    assert comparable(tmp_path / scored) == comparable(reference / scored)
