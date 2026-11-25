"""A stage that uses the cards carries on from what an earlier run left, and all three do it one way.

Until 25 November 2026 ``split``, ``classify`` and ``extract`` each wrote this out for itself, and the
three did not agree. ``split`` put its records in the photographs' order whenever it finished. The other
two added what a run had recovered, then what it read, then what it read again, each after what was
there: a classify stopped between its two cards' first calls left its file as pages 0, 2, 4, 1, 3, 5, and
extract then put 158 of 165 pages in an engine call with other pages than a run straight through would.

The first part of this file is what a stage's files are put through (``stages/io.py``). The second is
the one function the three stages carry on by (``stages/carry_on.py``).

The pages are blank images made by the bench, and every reply is invented.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

from bench import AN_ENGINE_ON_EACH_CARD, ONE_ENGINE, Bench, Stopped
from common.settings import Settings
from common.stage_timing import read_stage_timings
from stages.carry_on import Work, carry_on
from stages.io import Journal, in_the_order, read_jsonl, write_jsonl


def names_in(path: Path) -> list[str]:
    return [record["image_name"] for record in read_jsonl(path)]


# --- an output is put in its pages' order -----------------------------------------------------------------


def test_records_are_put_in_the_order_given_and_each_line_is_left_as_written(tmp_path: Path) -> None:
    output = tmp_path / "records.jsonl"
    # Written with spacing of its own, which writing the records out again would not keep.
    output.write_text('{"image_name":   "c.png"}\n{"image_name": "a.png"}\n{"image_name":  "b.png"}\n')

    in_the_order(output, ["a.png", "b.png", "c.png"])

    assert output.read_text() == (
        '{"image_name": "a.png"}\n{"image_name":  "b.png"}\n{"image_name":   "c.png"}\n'
    )
    assert not list(tmp_path.glob("*.rewriting"))


def test_a_file_already_in_order_is_not_touched(tmp_path: Path) -> None:
    output = tmp_path / "records.jsonl"
    output.write_text('{"image_name":   "a.png"}\n{"image_name": "b.png"}\n')
    before = output.stat().st_mtime_ns

    in_the_order(output, ["a.png", "b.png"])
    in_the_order(tmp_path / "not_there.jsonl", ["a.png"])

    assert output.stat().st_mtime_ns == before and not (tmp_path / "not_there.jsonl").exists()


def test_a_page_that_is_not_one_of_this_runs_is_kept_after_those_that_are(tmp_path: Path) -> None:
    """The folder, or the selection of pages, may have changed since the record was made."""
    output = tmp_path / "records.jsonl"
    output.write_text(
        "".join(json.dumps({"image_name": n}) + "\n" for n in ("z.png", "b.png", "y.png", "a.png"))
    )

    in_the_order(output, ["a.png", "b.png"])

    assert names_in(output) == ["a.png", "b.png", "z.png", "y.png"]


def test_a_page_in_the_output_twice_keeps_both_lines_next_to_each_other(tmp_path: Path) -> None:
    output = tmp_path / "records.jsonl"
    lines = [{"image_name": "b.png", "n": 1}, {"image_name": "a.png"}, {"image_name": "b.png", "n": 2}]
    output.write_text("".join(json.dumps(line) + "\n" for line in lines))

    in_the_order(output, ["a.png", "b.png"])

    assert read_jsonl(output) == [lines[1], lines[0], lines[2]]


def test_a_line_that_cannot_be_read_is_dropped_and_said(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A run stopped while writing leaves one. No reader could have used it."""
    output = tmp_path / "records.jsonl"
    output.write_text('{"image_name": "a.png"}\n{"image_name": "b.png", "raw_resp')

    with caplog.at_level(logging.WARNING):
        in_the_order(output, ["a.png", "b.png"])

    assert output.read_text() == '{"image_name": "a.png"}\n'
    assert "records.jsonl" in caplog.text and "1 line" in caplog.text


# --- an output is put in place whole -----------------------------------------------------------------------


class _CannotBeWritten:
    """Stands in for a run killed while a file is being written: it stops the write part way."""

    def __str__(self) -> str:
        raise Stopped


def test_an_output_being_written_leaves_the_earlier_one_whole(tmp_path: Path) -> None:
    output = tmp_path / "records.jsonl"
    output.write_text('{"image_name": "old.png"}\n')

    with pytest.raises(Stopped):
        write_jsonl(output, [{"image_name": "new.png"}, {"image_name": _CannotBeWritten()}])

    assert output.read_text() == '{"image_name": "old.png"}\n'


def test_an_output_written_whole_is_what_it_always_was(tmp_path: Path) -> None:
    output = tmp_path / "records.jsonl"

    assert write_jsonl(output, [{"image_name": "a.png", "n": 1.5}, {"image_name": "b.png"}]) == 2

    assert output.read_text() == '{"image_name": "a.png", "n": 1.5}\n{"image_name": "b.png"}\n'
    assert not list(tmp_path.glob("*.rewriting"))


# --- what a journal held is made into the stage's record -------------------------------------------------


def test_what_is_recovered_is_made_into_the_stages_record_before_it_is_added(tmp_path: Path) -> None:
    output = tmp_path / "records.jsonl"
    journal = Journal(output)
    with journal:
        journal.write({"image_name": "a.png", "raw_response": "a reply", "processing_time": 2.0})

    recovered = journal.recover(
        ["a.png"], made_of=lambda asked: {"image_name": asked["image_name"], "made": True}
    )

    assert recovered == [{"image_name": "a.png", "raw_response": "a reply", "processing_time": 2.0}]
    assert read_jsonl(output) == [{"image_name": "a.png", "made": True}]
    assert not journal.path.exists()


# --- the one way a stage carries on -----------------------------------------------------------------------

THINGS = [f"page_{n}.png" for n in range(6)]
# The stand-in for a reading function fails when it comes to one of these.
STOP_BEFORE: set[str] = set()
BOTH = [ONE_ENGINE, AN_ENGINE_ON_EACH_CARD]


def read_names(
    processor: Any, items: list[str], settings: Any, *, sink: Any = None
) -> list[dict[str, Any]]:
    """Stands in for a stage's reading function: one record an item, written as it is made.

    It fails as a card that falls over does. In the bench a worker runs inside the test's own process,
    so this is the stop that can be staged truly with an engine on each card: the worker hands the
    failure to its parent, which has by then written every record the cards had sent.
    """
    made = []
    for item in items:
        if item in STOP_BEFORE:
            raise RuntimeError("the card fell over")
        record = {"image_name": item, "raw_response": f"a reply about {item}", "processing_time": 1.5}
        made.append(record)
        if sink is not None:
            sink.write(record)
    return made


def work(output: Path, **more: Any) -> Work:
    described: dict[str, Any] = {
        "stage": "classify",
        "output": output,
        "items": list(THINGS),
        "name_of": str,
        "read": read_names,
        "timing": output.parent,
        **more,
    }
    return Work(**described)


def settings_of(bench: Bench, model: str) -> Settings:
    return Settings.load({"model_type": model}, config_path=bench.config)


def without_card(path: Path) -> list[dict[str, Any]]:
    """The records, whichever card read each: a page left to a second run may go to the other card."""
    return [{k: v for k, v in record.items() if k != "gpu_id"} for record in read_jsonl(path)]


def stop_before(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    monkeypatch.setattr(sys.modules[__name__], "STOP_BEFORE", set(names))


@pytest.mark.parametrize("model", BOTH)
def test_stopped_part_way_and_carried_on_writes_the_file_a_straight_run_writes(
    bench: Bench, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    """With an engine on each card, card 0 has finished pages 0, 2 and 4 and card 1 only page 1."""
    straight, stopped = tmp_path / "straight" / "r.jsonl", tmp_path / "stopped" / "r.jsonl"
    carry_on(settings_of(bench, model), work(straight))

    stop_before(monkeypatch, THINGS[3])
    with pytest.raises(RuntimeError, match="the card fell over"):
        carry_on(settings_of(bench, model), work(stopped))
    assert not stopped.exists()
    journalled = (
        ["page_0.png", "page_1.png", "page_2.png"]
        if model == ONE_ENGINE
        else ["page_0.png", "page_2.png", "page_4.png", "page_1.png"]
    )
    assert names_in(Journal(stopped).path) == journalled

    stop_before(monkeypatch)
    carried = carry_on(settings_of(bench, model), work(stopped))

    assert names_in(stopped) == THINGS
    assert without_card(stopped) == without_card(straight)
    assert not Journal(stopped).path.exists()
    assert carried.recovered == len(journalled) and len(carried.records) == len(THINGS) - len(journalled)


@pytest.mark.parametrize("model", BOTH)
def test_what_is_written_is_what_the_reading_function_gave(
    bench: Bench, tmp_path: Path, model: str
) -> None:
    output = tmp_path / "out" / "r.jsonl"

    carried = carry_on(settings_of(bench, model), work(output))

    assert read_jsonl(output) == carried.read == carried.records
    assert names_in(output) == THINGS
    assert carried.arrangement == ("single-engine" if model == ONE_ENGINE else "data-parallel (2 GPUs)")
    assert carried.model_seconds == pytest.approx(9.0 if model == ONE_ENGINE else 4.5)


def test_a_record_to_be_read_again_is_taken_out_read_and_put_back_in_its_place(
    bench: Bench, tmp_path: Path
) -> None:
    output = tmp_path / "out" / "r.jsonl"
    carry_on(settings_of(bench, ONE_ENGINE), work(output))
    spoilt = [{**r, "error": "refused"} if r["image_name"] == THINGS[2] else r for r in read_jsonl(output)]
    write_jsonl(output, spoilt)
    told: list[dict[str, Any]] = []

    carried = carry_on(settings_of(bench, ONE_ENGINE), work(output, taken_out=told.extend))

    assert [r["image_name"] for r in told] == [THINGS[2]] and carried.taken_out == 1
    assert [r["image_name"] for r in carried.records] == [THINGS[2]]
    assert names_in(output) == THINGS and not any(r.get("error") for r in read_jsonl(output))
    # The row of the stage's time counts the page once, though it was read twice.
    (row,) = read_stage_timings(output.parent)
    assert row.images == len(THINGS)


def test_with_nothing_left_no_engine_is_started_and_the_output_is_not_touched(
    bench: Bench, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    output = tmp_path / "out" / "r.jsonl"
    carry_on(settings_of(bench, ONE_ENGINE), work(output))
    started, before = bench.engines_started, output.stat().st_mtime_ns
    rows = read_stage_timings(output.parent)

    with caplog.at_level(logging.INFO):
        carried = carry_on(settings_of(bench, ONE_ENGINE), work(output))

    assert carried.arrangement is None and carried.records == [] and carried.already_done == 6
    assert bench.engines_started == started and output.stat().st_mtime_ns == before
    assert read_stage_timings(output.parent) == rows
    assert "Every page is already done: nothing to do" in caplog.text


def test_a_journal_that_holds_every_page_gives_the_output_and_starts_no_engine(
    bench: Bench, tmp_path: Path
) -> None:
    """A run that was stopped when it had read everything and written nothing."""
    output = tmp_path / "out" / "r.jsonl"
    output.parent.mkdir()
    with Journal(output) as journal:
        for n in (4, 0, 5, 1, 3, 2):
            journal.write({"image_name": THINGS[n], "processing_time": 1.5})

    carried = carry_on(settings_of(bench, ONE_ENGINE), work(output))

    assert names_in(output) == THINGS and not Journal(output).path.exists()
    assert bench.engines_started == 0 and carried.recovered == 6 and carried.arrangement is None


def test_a_run_stopped_after_its_records_were_added_adds_nothing_twice(
    bench: Bench, tmp_path: Path
) -> None:
    """The journal goes last. A run stopped between the output and that leaves both."""
    output = tmp_path / "out" / "r.jsonl"
    carry_on(settings_of(bench, ONE_ENGINE), work(output))
    before = output.read_text()
    with Journal(output) as journal:
        for record in read_jsonl(output):
            journal.write(record)

    carried = carry_on(settings_of(bench, ONE_ENGINE), work(output))

    assert output.read_text() == before and not Journal(output).path.exists()
    assert carried.recovered == 0 and bench.engines_started == 1


def test_the_stages_record_is_written_and_the_reply_is_what_the_journal_keeps(
    bench: Bench, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """For a stage whose output is what was done about a reply: the reply can be acted on again."""
    output = tmp_path / "out" / "r.jsonl"

    def made_of(asked: dict[str, Any]) -> dict[str, Any]:
        return {"image_name": asked["image_name"], "made": True}

    stop_before(monkeypatch, THINGS[3])
    with pytest.raises(RuntimeError, match="the card fell over"):
        carry_on(settings_of(bench, ONE_ENGINE), work(output, made_of=made_of))
    assert all("raw_response" in line and "made" not in line for line in read_jsonl(Journal(output).path))

    stop_before(monkeypatch)
    carried = carry_on(settings_of(bench, ONE_ENGINE), work(output, made_of=made_of))

    assert read_jsonl(output) == [{"image_name": name, "made": True} for name in THINGS]
    assert carried.records == [{"image_name": name, "made": True} for name in THINGS[3:]]
    assert all(record["processing_time"] == 1.5 for record in carried.read)
    # The time of the pages recovered is in the row with the rest: the journal kept it.
    (row,) = read_stage_timings(output.parent)
    assert row.images == 6 and row.inference_seconds == pytest.approx(9.0)


def test_with_no_folder_for_the_row_of_time_none_is_written(bench: Bench, tmp_path: Path) -> None:
    output = tmp_path / "out" / "r.jsonl"

    carry_on(settings_of(bench, ONE_ENGINE), work(output, timing=None))

    assert names_in(output) == THINGS and not (output.parent / ".stage_timing.jsonl").exists()


def test_a_run_that_carried_on_adds_to_the_row_it_began(
    bench: Bench, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out" / "r.jsonl"
    stop_before(monkeypatch, THINGS[2])
    with pytest.raises(RuntimeError):
        carry_on(settings_of(bench, ONE_ENGINE), work(output))
    stop_before(monkeypatch)

    carry_on(settings_of(bench, ONE_ENGINE), work(output))

    (row,) = read_stage_timings(output.parent)
    assert row.stage == "classify" and row.images == 6
    assert row.execution_mode == "a run that stopped, then single-engine"


def test_an_output_left_out_of_order_by_an_earlier_version_is_put_in_order(
    bench: Bench, tmp_path: Path
) -> None:
    output = tmp_path / "out" / "r.jsonl"
    output.parent.mkdir()
    write_jsonl(output, [{"image_name": THINGS[n]} for n in (0, 2, 4, 1, 3, 5)])

    carry_on(settings_of(bench, ONE_ENGINE), work(output))

    assert names_in(output) == THINGS and bench.engines_started == 0


def test_a_page_added_to_the_folder_since_is_read_and_put_in_its_place(
    bench: Bench, tmp_path: Path
) -> None:
    """Pages arrive over time: a second run reads only the new ones."""
    output = tmp_path / "out" / "r.jsonl"
    carry_on(settings_of(bench, ONE_ENGINE), work(output, items=[THINGS[0], THINGS[2], THINGS[5]]))

    carried = carry_on(settings_of(bench, ONE_ENGINE), work(output))

    assert [r["image_name"] for r in carried.records] == [THINGS[1], THINGS[3], THINGS[4]]
    assert names_in(output) == THINGS
