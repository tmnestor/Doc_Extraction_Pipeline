"""A stage that needs no model to make a record still carries on the one way.

Two stages can make a record with no model asked: ``classify`` when the pages' dataset says what they
are, and ``split`` when the photo split is off. Until 26 November 2026 each wrote its own carrying on
beside ``stages.carry_on``, and they did not keep to it: ``classify`` kept the record of a page an earlier
run could not classify, with its fallback type and its error, where every other path reads such a page
again.

Both now hand ``carry_on`` a way to make a record (``Work.tell``) in place of a reading function, and the
sequence is the same one: recover, take out what is to be done again, do what is left, put the file in
order.

The pages are blank images made by the bench, and every reply is invented.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from bench import ONE_ENGINE, Bench, run_settings, split_checked
from common.classification import BY_THE_DATASET
from stages import classify, records, split
from stages.carry_on import Work, carry_on
from stages.io import read_jsonl

STAGES_FOLDER = Path(classify.__file__).parent


def given(bench: Bench, output: Path, kind: str = "RECEIPT") -> list[dict[str, Any]]:
    classify.run(
        output, classify.checked(bench.pages, run_settings(ONE_ENGINE, bench.models)), given_type=kind
    )
    return read_jsonl(output)


def a_failed_record(bench: Bench, name: str) -> dict[str, Any]:
    return records.classified(
        str(bench.pages / name),
        {
            "document_type": "UNIVERSAL",
            "type_from": "fallback_not_asked",
            "error": "the engine did not answer",
        },
        0.0,
    )


# --- classify, when the dataset gives the type ----------------------------------------------------------


def test_a_page_an_earlier_run_could_not_classify_is_given_its_type_by_the_dataset(
    bench: Bench, tmp_path: Path
) -> None:
    output = tmp_path / "out" / "classifications.jsonl"
    output.parent.mkdir()
    failed = bench.names[2]
    output.write_text(json.dumps(a_failed_record(bench, failed)) + "\n")

    written = given(bench, output)

    assert [record["image_name"] for record in written] == bench.names
    assert all(record["document_type"] == "RECEIPT" for record in written)
    assert all(record["type_from"] == BY_THE_DATASET and "error" not in record for record in written)
    assert bench.engines_started == 0


def test_a_page_already_classified_keeps_its_record_and_its_place(bench: Bench, tmp_path: Path) -> None:
    output = tmp_path / "out" / "classifications.jsonl"
    bench.classify(output)
    asked = read_jsonl(output)

    assert given(bench, output) == asked


def test_what_a_stopped_run_had_classified_is_recovered_before_the_dataset_gives_the_rest(
    bench: Bench, tmp_path: Path
) -> None:
    output = tmp_path / "out" / "classifications.jsonl"
    output.parent.mkdir()
    journal = output.with_name(output.name + ".journal")
    recovered = records.classified(
        str(bench.pages / bench.names[4]),
        {"document_type": "INVOICE", "type_from": "rule", "raw_response": "a reply"},
        1.5,
    )
    journal.write_text(json.dumps(recovered) + "\n")

    written = given(bench, output)

    assert [record["image_name"] for record in written] == bench.names
    assert written[4] == recovered
    assert not journal.exists()


# --- split, when the photo split is off -----------------------------------------------------------------


def split_off(bench: Bench, output: Path) -> list[dict[str, Any]]:
    split.run(output, split_checked(ONE_ENGINE, bench.models, bench.pages, photo_split=False))
    return read_jsonl(output / "splits.jsonl")


def test_with_the_photo_split_off_every_photograph_is_passed_through_in_order(
    bench: Bench, tmp_path: Path
) -> None:
    written = split_off(bench, tmp_path / "split")

    assert [record["image_name"] for record in written] == bench.names
    assert all(record["action"] == "pass_through" for record in written)
    assert sorted(path.name for path in (tmp_path / "split" / "images").iterdir()) == sorted(bench.names)
    assert bench.engines_started == 0


def test_with_the_photo_split_off_a_second_run_does_nothing_again(bench: Bench, tmp_path: Path) -> None:
    first = split_off(bench, tmp_path / "split")
    held = (tmp_path / "split" / "splits.jsonl").read_bytes()

    assert split_off(bench, tmp_path / "split") == first
    assert (tmp_path / "split" / "splits.jsonl").read_bytes() == held


def test_with_the_photo_split_off_a_photograph_an_earlier_run_could_not_ask_about_is_passed_through(
    bench: Bench, tmp_path: Path
) -> None:
    output = tmp_path / "split"
    output.mkdir()
    name = bench.names[1]
    failed = {
        "image_name": name,
        "image_path": str(bench.pages / name),
        "outputs": [name],
        "error": "no answer",
    }
    (output / "splits.jsonl").write_text(json.dumps(failed) + "\n")

    written = split_off(bench, output)

    assert [record["image_name"] for record in written] == bench.names
    assert all("error" not in record for record in written)


# --- the one sequence ----------------------------------------------------------------------------------


@pytest.mark.parametrize("stage", ["classify", "split"])
def test_neither_stage_carries_on_by_itself(stage: str) -> None:
    """What carrying on is made of is used by ``carry_on`` alone, and not by these two stages."""
    source = (STAGES_FOLDER / f"{stage}.py").read_text()

    for part in ("read_completed_images", "take_out", "Journal", "in_the_order"):
        assert part not in source, part
    assert "carry_on(" in source and "tell=" in source


def work(tmp_path: Path, **how: Any) -> Work:
    return Work(stage="classify", output=tmp_path / "out.jsonl", items=["a"], name_of=str, **how)


def test_work_says_one_way_of_coming_by_a_record_and_not_two_or_none(tmp_path: Path) -> None:
    def read(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return []

    def tell(item: Any) -> dict[str, Any]:
        return {"image_name": item}

    with pytest.raises(ValueError, match="exactly one"):
        carry_on(None, work(tmp_path))
    with pytest.raises(ValueError, match="exactly one"):
        carry_on(None, work(tmp_path, read=read, tell=tell))


def test_records_made_with_no_model_start_no_engine_and_note_no_time(bench: Bench, tmp_path: Path) -> None:
    carried = carry_on(None, work(tmp_path, tell=lambda item: {"image_name": item}, timing=tmp_path))

    assert carried.records == [{"image_name": "a"}] and carried.arrangement is None
    assert read_jsonl(tmp_path / "out.jsonl") == [{"image_name": "a"}]
    assert not (tmp_path / ".stage_timing.jsonl").exists()
    assert bench.engines_started == 0
