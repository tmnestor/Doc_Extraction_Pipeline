"""The split stage starts its engines as every other stage does: one on each card where the model allows.

Until 25 November 2026 the split stage started one engine across the cards whatever the model. A model
that fits on one card, which every other stage gives an engine on each card, had here half the engines
it could have had, and a second arrangement of its own to go wrong.

Now the stage hands its photographs to the same function the other stages do. The model is asked about
them by one function, whichever way the engines are arranged, and each reply is acted on as it arrives.

The photographs are blank images made by the bench, and every reply is invented.
"""

import shutil
from pathlib import Path
from typing import Any

import pytest

from bench import AN_ENGINE_ON_EACH_CARD, ONE_ENGINE, Bench, Stopped
from bench import run_settings, split_checked
from common.engine_calls import StageStopped
from common.stage_settings import resolve
from common.stage_timing import read_stage_timings
from stages import split
from stages.io import read_jsonl

TWO_RECEIPTS = '[{"box_2d": [0, 0, 1000, 480], "label": "receipt"}, {"box_2d": [0, 520, 1000, 1000], "label": "receipt"}]'


def run(bench: Bench, output: Path, model: str, *, photo_split: bool = True) -> Path:
    return split.run(output, split_checked(model, bench.models, bench.pages, photo_split=photo_split))


def names_in(path: Path) -> list[str]:
    return [record["image_name"] for record in read_jsonl(path)]


def images_in(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir())


# --- how the engines are arranged -----------------------------------------------------------------------


@pytest.mark.parametrize(("model", "engines"), [(ONE_ENGINE, 1), (AN_ENGINE_ON_EACH_CARD, 2)])
def test_the_stage_starts_an_engine_on_each_card_for_a_model_that_fits_on_one(
    bench: Bench, tmp_path: Path, model: str, engines: int
) -> None:
    run(bench, tmp_path / "split", model)

    assert bench.engines_started == engines


def test_with_an_engine_on_each_card_every_photograph_is_asked_about_once(
    bench: Bench, tmp_path: Path
) -> None:
    output = tmp_path / "split"

    run(bench, output, AN_ENGINE_ON_EACH_CARD)

    # Six photographs, three to a card, two to an engine call: two calls a card.
    assert sorted(bench.processor.calls) == [["1 photographs"]] * 2 + [["2 photographs"]] * 2
    assert names_in(output / "splits.jsonl") == bench.names
    assert images_in(output / "images") == bench.names


def test_both_arrangements_write_the_same_records_and_the_same_images(bench: Bench, tmp_path: Path) -> None:
    def replies(images: list[Any], prompts: list[str], max_tokens: int) -> list[str]:
        return [TWO_RECEIPTS] * len(images)

    bench.processor.ask = replies  # type: ignore[method-assign]

    run(bench, tmp_path / "one" / "split", ONE_ENGINE)
    run(bench, tmp_path / "each" / "split", AN_ENGINE_ON_EACH_CARD)

    one, each = (read_jsonl(tmp_path / which / "split" / "splits.jsonl") for which in ("one", "each"))
    assert one == each
    assert all(record["action"] == "split" and len(record["outputs"]) == 2 for record in each)
    assert images_in(tmp_path / "one" / "split" / "images") == images_in(
        tmp_path / "each" / "split" / "images"
    )
    assert len(images_in(tmp_path / "each" / "split" / "images")) == 2 * len(bench.names)


def test_the_records_are_in_the_photographs_order_whatever_order_the_cards_answered_in(
    bench: Bench, tmp_path: Path
) -> None:
    """Card 0 answers about every other photograph before card 1 answers about any."""
    output = tmp_path / "split"

    run(bench, output, AN_ENGINE_ON_EACH_CARD)

    assert names_in(output / "splits.jsonl") == bench.names
    assert not list(output.glob("*.rewriting"))


def test_a_file_an_earlier_version_left_half_written_is_carried_on_from(
    bench: Bench, tmp_path: Path
) -> None:
    """Before 25 November 2026 the stage wrote each record straight to its file, with no journal, in
    the order the cards answered.
    """
    whole = tmp_path / "whole"
    run(bench, whole, ONE_ENGINE)
    lines = (whole / "splits.jsonl").read_text().splitlines(keepends=True)
    output = tmp_path / "split"
    (output / "images").mkdir(parents=True)
    (output / "splits.jsonl").write_text(lines[3] + lines[0])
    for name in (bench.names[3], bench.names[0]):
        shutil.copy2(whole / "images" / name, output / "images" / name)
    bench.processor.calls.clear()

    run(bench, output, ONE_ENGINE)

    assert bench.processor.calls == [["2 photographs"], ["2 photographs"]]
    assert (output / "splits.jsonl").read_text() == (whole / "splits.jsonl").read_text()
    assert images_in(output / "images") == bench.names


def two_receipts_in_each(bench: Bench) -> None:
    """Every photograph is answered as holding two receipts, by a call the bench still counts."""

    def replies(images: list[Any], prompts: list[str], max_tokens: int) -> list[str]:
        bench.processor._called([f"{len(images)} photographs"])
        return [TWO_RECEIPTS] * len(images)

    bench.processor.ask = replies  # type: ignore[method-assign]


def same_records_and_images(stopped: Path, straight: Path) -> None:
    assert (stopped / "splits.jsonl").read_text() == (straight / "splits.jsonl").read_text()
    assert images_in(stopped / "images") == images_in(straight / "images")
    for image in sorted((straight / "images").iterdir()):
        assert (stopped / "images" / image.name).read_bytes() == image.read_bytes()
    assert not (stopped / "splits.jsonl.journal").exists()


def test_split_stopped_and_carried_on_writes_the_records_and_images_of_a_straight_run(
    bench: Bench, tmp_path: Path
) -> None:
    two_receipts_in_each(bench)
    straight, stopped = tmp_path / "straight", tmp_path / "stopped"
    run(bench, straight, ONE_ENGINE)
    bench.processor.stop_at_call = len(bench.processor.calls) + 2
    with pytest.raises(Stopped):
        run(bench, stopped, ONE_ENGINE)
    bench.processor.stop_at_call = None

    run(bench, stopped, ONE_ENGINE)

    same_records_and_images(stopped, straight)


def test_with_an_engine_on_each_card_a_stage_whose_engine_stopped_carries_on_to_a_straight_runs_files(
    bench: Bench, tmp_path: Path
) -> None:
    """A stop that the bench stages truly with an engine on each card: the engine stops answering, the
    card's worker hands that to its parent, and the parent has by then kept what the cards had sent.
    """
    bench.stop_after(2)
    two_receipts_in_each(bench)
    straight, stopped = tmp_path / "straight", tmp_path / "stopped"
    run(bench, straight, AN_ENGINE_ON_EACH_CARD)
    bench.processor.dead_from_attempt = len(bench.processor.attempts) + 2
    with pytest.raises(StageStopped):
        run(bench, stopped, AN_ENGINE_ON_EACH_CARD)
    assert read_jsonl(stopped / "splits.jsonl.journal") and not (stopped / "splits.jsonl").exists()
    bench.processor.dead_from_attempt = None

    run(bench, stopped, AN_ENGINE_ON_EACH_CARD)

    same_records_and_images(stopped, straight)


def test_split_carries_on_though_a_photograph_it_had_finished_has_gone(
    bench: Bench, tmp_path: Path
) -> None:
    """The journal keeps the record made of a reply, so carrying on needs nothing of the photograph."""
    photographs, output = tmp_path / "photographs", tmp_path / "split"
    shutil.copytree(bench.pages, photographs)
    bench.processor.stop_at_call = 2
    with pytest.raises(Stopped):
        split.run(output, split_checked(ONE_ENGINE, bench.models, photographs))
    bench.processor.stop_at_call = None
    (photographs / bench.names[0]).unlink()

    split.run(output, split_checked(ONE_ENGINE, bench.models, photographs))

    # Its record is kept, after those of the photographs that are still there, and so is its image.
    assert names_in(output / "splits.jsonl") == [*bench.names[1:], bench.names[0]]
    assert images_in(output / "images") == bench.names
    assert not (output / "splits.jsonl.journal").exists()


def test_photo_split_switched_off_after_a_stopped_run_keeps_what_that_run_had_done(
    bench: Bench, tmp_path: Path
) -> None:
    two_receipts_in_each(bench)
    output = tmp_path / "split"
    bench.processor.stop_at_call = 2
    with pytest.raises(Stopped):
        run(bench, output, ONE_ENGINE)
    bench.processor.stop_at_call = None

    run(bench, output, ONE_ENGINE, photo_split=False)

    records = read_jsonl(output / "splits.jsonl")
    assert [record["image_name"] for record in records] == bench.names
    assert [record["action"] for record in records] == ["split"] * 2 + ["pass_through"] * 4
    # No photograph is there whole beside the images cut from it.
    images = images_in(output / "images")
    assert len(images) == 2 * 2 + 4 and not set(images) & set(bench.names[:2])
    assert not (output / "splits.jsonl.journal").exists()


# --- asking is apart from acting ------------------------------------------------------------------------


def test_asking_about_photographs_gives_the_replies_and_writes_nothing(
    bench: Bench, tmp_path: Path
) -> None:
    settings = run_settings(ONE_ENGINE, bench.models)
    photographs = [bench.pages / name for name in bench.names]
    seen: list[dict[str, Any]] = []

    class Sink:
        def write(self, record: dict[str, Any]) -> None:
            seen.append(record)

    asked = split.ask_about_photographs(bench.processor, photographs, settings, sink=Sink())

    assert asked == seen
    assert [record["image_name"] for record in asked] == bench.names
    assert all(
        set(record) == {"image_name", "image_path", "raw_response", "error", "processing_time"}
        and record["raw_response"] == "[]"
        and record["error"] is None
        for record in asked
    )
    assert list(tmp_path.glob("**/splits.jsonl")) == []


# --- stopped, and stopping ------------------------------------------------------------------------------


def test_stopped_part_way_with_an_engine_on_each_card_the_next_run_asks_only_about_the_rest(
    bench: Bench, tmp_path: Path
) -> None:
    output = tmp_path / "split"
    bench.processor.stop_at_call = 2

    with pytest.raises(Stopped):
        run(bench, output, AN_ENGINE_ON_EACH_CARD)

    bench.processor.stop_at_call = None
    bench.processor.calls.clear()
    run(bench, output, AN_ENGINE_ON_EACH_CARD)

    assert names_in(output / "splits.jsonl") == bench.names
    assert images_in(output / "images") == bench.names


def test_with_an_engine_on_each_card_the_stage_stops_when_the_engine_stops_answering(
    bench: Bench, tmp_path: Path
) -> None:
    bench.stop_after(2)
    output = tmp_path / "split"
    bench.processor.dead_from_attempt = 2

    with pytest.raises(StageStopped, match="2 requests in a row failed"):
        run(bench, output, AN_ENGINE_ON_EACH_CARD)

    # What was answered before the engine stopped was acted on, and is kept: the replies in the
    # stage's journal, which the card's worker handed its parent before the stop, and the images.
    kept = read_jsonl(output / "splits.jsonl.journal")
    assert kept and not any(record.get("error") for record in kept[:2])
    assert set(images_in(output / "images")) >= {record["image_name"] for record in kept[:2]}

    bench.processor.dead_from_attempt = None
    bench.processor.calls.clear()
    run(bench, output, AN_ENGINE_ON_EACH_CARD)

    records = read_jsonl(output / "splits.jsonl")
    assert [record["image_name"] for record in records] == bench.names
    assert not any(record.get("error") for record in records)
    assert not (output / "splits.jsonl.journal").exists()
    # The photographs answered about before the stop were not asked about again.
    asked_again = sum(int(call[0].split()[0]) for call in bench.processor.calls)
    assert asked_again == len(bench.names) - sum(1 for record in kept if not record.get("error"))


# --- the stage's time -----------------------------------------------------------------------------------


def test_a_run_of_split_given_the_runs_folder_notes_a_row_of_its_time(bench: Bench, tmp_path: Path) -> None:
    split.run(
        tmp_path / "out" / "split",
        split_checked(ONE_ENGINE, bench.models, bench.pages),
        timing=tmp_path / "out",
    )

    (row,) = read_stage_timings(tmp_path / "out")
    assert (row.stage, row.images, row.execution_mode) == ("split", 6, "single-engine")


def test_split_started_as_a_task_notes_its_row_in_the_runs_output_folder(
    bench: Bench, tmp_path: Path
) -> None:
    environ = {
        "model": AN_ENGINE_ON_EACH_CARD,
        "image_dir": str(bench.pages),
        "output": str(tmp_path / "out"),
        "DOC_EXTRACT_MODELS_DIR": str(bench.models),
        "photo_split": "true",
    }

    task = resolve("split", environ)
    split.start(task, split.check(task))

    (row,) = read_stage_timings(tmp_path / "out")
    assert (row.stage, row.images, row.execution_mode) == ("split", 6, "data-parallel (2 GPUs)")
    assert not (tmp_path / "out" / "split" / ".stage_timing.jsonl").exists()
