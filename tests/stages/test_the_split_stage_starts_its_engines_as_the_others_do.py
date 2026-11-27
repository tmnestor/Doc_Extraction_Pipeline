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
from bench import (
    AN_ENGINE_ON_EACH_CARD,
    ONE_ENGINE,
    Bench,
    Stopped,
    asks_which_way_up,
    run_settings,
    split_checked,
)
from PIL import Image

from common import photo_split
from common.engine_calls import StageStopped
from common.prompt_catalog import PromptCatalog
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
        if asks_which_way_up(prompts):
            return ["YES"] * len(images)
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
        set(record)
        == {"image_name", "image_path", "raw_response", "error", "processing_time", "upright_turn"}
        and record["raw_response"] == "[]"
        and record["error"] is None
        and record["upright_turn"] == 0
        for record in asked
    )
    assert list(tmp_path.glob("**/splits.jsonl")) == []


# --- a photograph is set the right way up before it is asked about ----------------------------------------


def printed_page() -> Image.Image:
    """A white page of print: rows of letter-sized marks, with a heading to tell its top from its bottom."""
    page = Image.new("RGB", (300, 600), "white")
    page.paste((20, 20, 20), (30, 20, 200, 50))
    for top in range(80, 560, 22):
        for left in range(24 + top % 9, 270, 9):
            if (top + left) % 7:
                page.paste((30, 30, 30), (left, top, left + 5, top + 10))
    return page


class ToldWhichWayUp:
    """Stands in for the processor: answers the first question as it is told to, and keeps what it was shown."""

    def __init__(self, upright: str) -> None:
        self.upright = upright
        self.shown: list[tuple[str, int, list[Image.Image]]] = []

    def load_document_image(self, image_path: str) -> Image.Image:
        return Image.open(image_path)

    def ask(self, images: list[Any], prompts: list[str], max_tokens: int) -> list[str]:
        self.shown.append((prompts[0], max_tokens, [image.copy() for image in images]))
        return [self.upright if asks_which_way_up(prompts) else "[]"] * len(images)


@pytest.mark.parametrize(
    ("stored_turned", "says", "turn_recorded", "ends_upright"),
    [
        # What a model that answers rightly says of a photograph stored each way up.
        (0, "YES", 0, True),
        (180, "NO", 180, True),
        (90, "NO", 270, True),
        (270, "YES", 90, True),
        # The model's answer is what decides between the right way up and upside down.
        (90, "YES", 90, False),
        (0, "NO", 180, False),
        (180, "I cannot say", 0, False),
    ],
)
def test_a_photograph_is_set_the_right_way_up_before_it_is_asked_where_its_receipts_are(
    bench: Bench, tmp_path: Path, stored_turned: int, says: str, turn_recorded: int, ends_upright: bool
) -> None:
    """Set on end by its print, then turned half way round when the model says it is upside down."""
    page = printed_page()
    photograph = tmp_path / "page.png"
    page.rotate(stored_turned, expand=True).save(photograph)
    processor = ToldWhichWayUp(says)

    (asked,) = split.ask_about_photographs(processor, [photograph], run_settings(ONE_ENGINE, bench.models))

    assert asked["upright_turn"] == turn_recorded
    (first, budget, _), (second, _, (boxes_asked_of,)) = processor.shown
    assert first == PromptCatalog().photo_split_prompt("upright")
    assert budget == photo_split.CONFIG.upright_max_tokens
    assert second == PromptCatalog().photo_split_prompt()
    as_stored = page.rotate(stored_turned, expand=True)
    assert boxes_asked_of.tobytes() == as_stored.rotate(turn_recorded, expand=True).tobytes()
    assert (boxes_asked_of.size == page.size and boxes_asked_of.tobytes() == page.tobytes()) is ends_upright


def test_a_photograph_the_model_could_not_be_asked_about_is_left_the_way_up_it_came(
    bench: Bench, tmp_path: Path
) -> None:
    photograph = tmp_path / "page.png"
    printed_page().rotate(90, expand=True).save(photograph)
    bench.processor.dead_from_attempt = 1

    (asked,) = split.ask_about_photographs(
        bench.processor, [photograph], run_settings(ONE_ENGINE, bench.models)
    )

    assert asked["error"] and asked["upright_turn"] == 0


def test_the_stage_writes_a_photograph_it_set_the_right_way_up_as_png_and_records_the_turn(
    bench: Bench, tmp_path: Path
) -> None:
    photographs, output = tmp_path / "photographs", tmp_path / "split"
    photographs.mkdir()
    page = printed_page()
    page.rotate(90, expand=True).save(photographs / "sideways.jpg", quality=95)
    page.save(photographs / "upright.jpg", quality=95)

    split.run(output, split_checked(ONE_ENGINE, bench.models, photographs))

    sideways, upright = read_jsonl(output / "splits.jsonl")
    assert (sideways["upright_turn"], sideways["outputs"]) == (90, ["sideways.png"])
    assert (upright["upright_turn"], upright["outputs"]) == (0, ["upright.jpg"])
    assert (output / "images" / "upright.jpg").read_bytes() == (photographs / "upright.jpg").read_bytes()
    with Image.open(output / "images" / "sideways.png") as written:
        # Set on end. The stand-in says YES to every photograph, so it is not turned further.
        assert written.size == page.size


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


# --- started afresh --------------------------------------------------------------------------------------


def started_as_a_task(bench: Bench, out: Path, **more: str) -> list[str]:
    """Start split as the runner does, and give the names of the images it leaves for the later stages."""
    environ = {
        "model": ONE_ENGINE,
        "image_dir": str(bench.pages),
        "output": str(out),
        "DOC_EXTRACT_MODELS_DIR": str(bench.models),
        **more,
    }
    task = resolve("split", environ)
    split.start(task, split.check(task))
    return sorted(path.name for path in (out / "split" / "images").iterdir())


def test_split_started_afresh_deletes_its_records_and_the_images_they_name(
    bench: Bench, tmp_path: Path
) -> None:
    """Until 26 November 2026 CLEAR_PREV_OUTPUT did nothing for split. An image an earlier run had cut
    would be read by the stages after it, beside whatever the new run made of the same photograph."""
    two_receipts_in_each(bench)
    out = tmp_path / "out"
    cut = started_as_a_task(bench, out, photo_split="true")
    assert len(cut) == 2 * len(bench.names) and all("__r" in name for name in cut)

    passed_through = started_as_a_task(bench, out, CLEAR_PREV_OUTPUT="true")

    assert passed_through == sorted(bench.names, key=str)
    records = read_jsonl(out / "split" / "splits.jsonl")
    assert [record["action"] for record in records] == ["pass_through"] * len(bench.names)


def test_split_not_started_afresh_keeps_what_it_had_done(bench: Bench, tmp_path: Path) -> None:
    two_receipts_in_each(bench)
    out = tmp_path / "out"
    cut = started_as_a_task(bench, out, photo_split="true")

    assert started_as_a_task(bench, out) == cut


def test_split_started_afresh_after_a_stopped_run_removes_what_that_run_had_cut(
    bench: Bench, tmp_path: Path
) -> None:
    """What a stopped run had done is in its journal, not yet in its records: those images go too."""
    two_receipts_in_each(bench)
    out = tmp_path / "out"
    bench.processor.stop_at_call = 2
    with pytest.raises(Stopped):
        started_as_a_task(bench, out, photo_split="true")
    bench.processor.stop_at_call = None
    assert any("__r" in path.name for path in (out / "split" / "images").iterdir())

    assert started_as_a_task(bench, out, CLEAR_PREV_OUTPUT="true") == sorted(bench.names, key=str)
    assert not (out / "split" / "splits.jsonl.journal").exists()
