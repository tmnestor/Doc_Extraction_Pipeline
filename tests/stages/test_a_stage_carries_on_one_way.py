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
from pathlib import Path

import pytest

from bench import Stopped
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
