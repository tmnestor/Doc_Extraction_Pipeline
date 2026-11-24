"""The split stage starts its engines as every other stage does: one on each card where the model allows.

Until 25 November 2026 the split stage started one engine across the cards whatever the model. A model
that fits on one card, which every other stage gives an engine on each card, had here half the engines
it could have had, and a second arrangement of its own to go wrong.

Now the stage hands its photographs to the same function the other stages do. The model is asked about
them by one function, whichever way the engines are arranged, and each reply is acted on as it arrives.

The photographs are blank images made by the bench, and every reply is invented.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from bench import AN_ENGINE_ON_EACH_CARD, ONE_ENGINE, Bench, Stopped
from common.engine_calls import StageStopped
from common.settings import Settings
from stages import split
from stages.io import read_jsonl

TWO_RECEIPTS = '[{"box_2d": [0, 0, 1000, 480], "label": "receipt"}, {"box_2d": [0, 520, 1000, 1000], "label": "receipt"}]'


def run(bench: Bench, output: Path, model: str) -> Path:
    return split.run(bench.pages, output, model_type=model, config_path=bench.config)


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


def test_a_file_already_in_the_photographs_order_is_left_as_it_is(tmp_path: Path) -> None:
    records = tmp_path / "splits.jsonl"
    # Written with spacing of its own, which a rewrite would not keep.
    records.write_text('{"image_name":   "a.png"}\n{"image_name": "b.png"}\n')
    before = records.read_text()

    split._in_the_photographs_order(records, [tmp_path / "a.png", tmp_path / "b.png"])

    assert records.read_text() == before


def test_records_out_of_order_are_put_in_order_with_nothing_lost(tmp_path: Path) -> None:
    records = tmp_path / "splits.jsonl"
    lines = [{"image_name": name, "action": "pass_through"} for name in ("c.png", "a.png", "b.png")]
    records.write_text("".join(json.dumps(line) + "\n" for line in lines))

    split._in_the_photographs_order(records, [tmp_path / name for name in ("a.png", "b.png", "c.png")])

    assert names_in(records) == ["a.png", "b.png", "c.png"]
    assert sorted(read_jsonl(records), key=str) == sorted(lines, key=str)


# --- asking is apart from acting ------------------------------------------------------------------------


def test_asking_about_photographs_gives_the_replies_and_writes_nothing(
    bench: Bench, tmp_path: Path
) -> None:
    settings = Settings.load({"model_type": ONE_ENGINE}, config_path=bench.config)
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
    bench.with_settings(
        lambda settings: settings["pipeline"]["processing"].update(stop_after_failed_requests=2)
    )
    output = tmp_path / "split"
    bench.processor.dead_from_attempt = 2

    with pytest.raises(StageStopped, match="2 requests in a row failed"):
        run(bench, output, AN_ENGINE_ON_EACH_CARD)

    # What was answered before the engine stopped was acted on, and is kept.
    kept = read_jsonl(output / "splits.jsonl")
    assert kept and not any(record.get("error") for record in kept[:2])

    bench.processor.dead_from_attempt = None
    run(bench, output, AN_ENGINE_ON_EACH_CARD)

    records = read_jsonl(output / "splits.jsonl")
    assert [record["image_name"] for record in records] == bench.names
    assert not any(record.get("error") for record in records)
