"""A page a stage could not read is read again by the next run, and a stage whose engine has stopped
answering stops.

Until 25 November 2026 a stage carried on from its earlier output by the names of the pages in it, and a
page recorded as failed was among them: it was never read again. And every failure was taken for a
page's own. So an engine that stopped answering part way left every page after it recorded as failed, in
seconds, the stage ended with exit code 0, and the next run found nothing to do.

Now:

- when request after request fails with none answered between them, the stage stops with its reason,
  at ``common.engine_calls.STOP_AFTER_FAILED_REQUESTS``;
- a stage that carries on takes the failed pages' records out of its output and reads those pages again;
- the extract stage also reads again a page that has been given another document type since.

The pages are blank images made by the bench, and every reply is invented.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from bench import AN_ENGINE_ON_EACH_CARD, KINDS, ONE_ENGINE, Bench
from bench import split_checked
from common.engine_calls import StageStopped
from common.stage_timing import read_stage_timings
from stages import __main__ as starter
from stages import split
from stages.io import read_jsonl, take_out
from stages.records import failed

BOTH = [ONE_ENGINE, AN_ENGINE_ON_EACH_CARD]


def records_by_name(path: Path) -> dict[str, dict[str, Any]]:
    records = read_jsonl(path)
    names = [record["image_name"] for record in records]
    assert len(names) == len(set(names)), "a page is in the output twice"
    return {record["image_name"]: record for record in records}


def in_reading_order(bench: Bench) -> list[str]:
    """The bench's pages in the order extract reads them: one type after another."""
    return sorted(bench.names, key=lambda name: (KINDS.index(name[8:-4]), name))


def stop_after(bench: Bench, requests: int) -> None:
    bench.stop_after(requests)


def classified(bench: Bench, folder: Path, types: dict[str, str] | None = None) -> Path:
    """A classifications file for the bench's pages, each given its own type unless ``types`` says."""
    records = []
    for name in bench.names:
        kind = next(k for k in ("bank_statement", "invoice", "receipt") if k in name).upper()
        records.append(
            {
                "image_name": name,
                "image_path": str(bench.pages / name),
                "document_type": (types or {}).get(name, kind),
            }
        )
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "classifications.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


# --- taking records out of an output ----------------------------------------------------------------------


def test_the_records_of_failed_pages_are_taken_out_and_the_rest_left_as_written(tmp_path: Path) -> None:
    output = tmp_path / "raw_extractions.jsonl"
    lines = [
        json.dumps({"image_name": "quill.png", "error": None, "raw_response": "TOTAL: 4"}),
        json.dumps({"image_name": "feather.png", "error": "the engine refused it", "raw_response": ""}),
        json.dumps({"image_name": "ink.png", "raw_response": "TOTAL: 9"}),
    ]
    output.write_text("\n".join(lines) + "\n")

    taken = take_out(output, failed)

    assert [record["image_name"] for record in taken] == ["feather.png"]
    assert output.read_text() == lines[0] + "\n" + lines[2] + "\n"
    assert not list(tmp_path.glob("*.rewriting"))


def test_an_output_with_nothing_to_take_out_is_not_touched(tmp_path: Path) -> None:
    output = tmp_path / "classifications.jsonl"
    # Written with spacing of its own, which a rewrite would not keep.
    output.write_text('{"image_name":   "quill.png"}\n{"image_name": "ink.png", "error": null}\n')
    before = output.read_text()

    assert take_out(output, failed) == []
    assert output.read_text() == before
    assert take_out(tmp_path / "not_there.jsonl", failed) == []


def test_a_line_a_stopped_run_left_half_written_is_dropped_with_them(tmp_path: Path) -> None:
    output = tmp_path / "raw_extractions.jsonl"
    whole = json.dumps({"image_name": "quill.png", "raw_response": "TOTAL: 4"})
    output.write_text(whole + '\n{"image_name": "feather.png", "raw_resp')

    assert take_out(output, failed) == []
    assert output.read_text() == whole + "\n"


# --- a page that could not be read is read again --------------------------------------------------------


@pytest.mark.parametrize("model", BOTH)
def test_a_page_classify_could_not_read_is_asked_about_again_by_the_next_run(
    bench: Bench, tmp_path: Path, model: str, caplog: pytest.LogCaptureFixture
) -> None:
    output = tmp_path / "out" / "classifications.jsonl"
    spoilt = bench.names[3]
    bench.processor.cannot_read = {spoilt}

    with caplog.at_level(logging.WARNING):
        bench.classify(output, model)

    first = records_by_name(output)
    assert set(first) == set(bench.names)
    assert (
        first[spoilt]["error"] == f"cannot read {spoilt}" and first[spoilt]["document_type"] == "UNIVERSAL"
    )
    assert [name for name, record in first.items() if record.get("error")] == [spoilt]
    assert "1 of 6 page(s) could not be classified" in caplog.text

    bench.processor.cannot_read = set()
    bench.processor.calls.clear()
    bench.classify(output, model)

    assert bench.processor.calls == [[spoilt]]
    second = records_by_name(output)
    assert set(second) == set(bench.names)
    assert second[spoilt]["document_type"] == "BANK_STATEMENT" and "error" not in second[spoilt]
    assert all(second[name] == first[name] for name in bench.names if name != spoilt)
    # The page read again is put back in its place, and not added after the others.
    assert list(second) == bench.names
    # The stage's row of timing covers every page of the output once.
    (row,) = [t for t in read_stage_timings(output.parent) if t.stage == "classify"]
    assert row.images == len(bench.names)


@pytest.mark.parametrize("model", BOTH)
def test_a_page_extract_could_not_read_is_read_again_by_the_next_run(
    bench: Bench, tmp_path: Path, model: str, caplog: pytest.LogCaptureFixture
) -> None:
    out = tmp_path / "out"
    classifications, output = classified(bench, out), out / "raw_extractions.jsonl"
    spoilt = bench.names[4]
    bench.processor.cannot_read = {spoilt}

    with caplog.at_level(logging.WARNING):
        bench.extract(classifications, output, model)

    first = records_by_name(output)
    assert first[spoilt]["error"] == f"cannot read {spoilt}" and first[spoilt]["raw_response"] == ""
    assert [name for name, record in first.items() if record["error"]] == [spoilt]
    assert "1 of 6 page(s) could not be read" in caplog.text

    bench.processor.cannot_read = set()
    bench.processor.calls.clear()
    bench.extract(classifications, output, model)

    assert bench.processor.calls == [[spoilt]]
    second = records_by_name(output)
    assert set(second) == set(bench.names)
    assert second[spoilt]["error"] is None
    assert second[spoilt]["raw_response"] == f"TOTAL_AMOUNT: read from {spoilt}"
    assert all(second[name] == first[name] for name in bench.names if name != spoilt)
    # The page read again is put back in its place, and not added after the others.
    assert list(second) == in_reading_order(bench)
    (row,) = [t for t in read_stage_timings(out) if t.stage == "extract"]
    assert row.images == len(bench.names)


def test_a_page_that_still_cannot_be_read_is_recorded_once_and_tried_each_run(
    bench: Bench, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    classifications, output = classified(bench, out), out / "raw_extractions.jsonl"
    spoilt = bench.names[0]
    bench.processor.cannot_read = {spoilt}

    bench.extract(classifications, output)
    bench.processor.attempts.clear()
    bench.extract(classifications, output)

    assert bench.processor.attempts == [[spoilt]]
    records = records_by_name(output)
    assert set(records) == set(bench.names) and records[spoilt]["error"]


def test_a_run_with_no_failed_page_and_nothing_new_starts_no_engine(bench: Bench, tmp_path: Path) -> None:
    out = tmp_path / "out"
    classifications, output = classified(bench, out), out / "raw_extractions.jsonl"
    bench.extract(classifications, output)
    before = output.read_text()

    bench.extract(classifications, output)

    assert bench.engines_started == 1
    assert output.read_text() == before


def test_a_page_given_another_type_since_it_was_read_is_read_again(bench: Bench, tmp_path: Path) -> None:
    """Its record is of the fields of the type it had: a page classified again must not keep it."""
    out = tmp_path / "out"
    output = out / "raw_extractions.jsonl"
    retyped = bench.names[2]
    bench.extract(classified(bench, out, {retyped: "UNIVERSAL"}), output)
    assert records_by_name(output)[retyped]["document_type"] == "UNIVERSAL"

    bench.processor.calls.clear()
    bench.extract(classified(bench, out), output)

    assert bench.processor.calls == [[retyped]]
    records = records_by_name(output)
    assert set(records) == set(bench.names)
    assert records[retyped]["document_type"] == "RECEIPT" and records[retyped]["prompt_used"] == "receipt"
    assert list(records) == in_reading_order(bench)


# --- an engine that has stopped answering ---------------------------------------------------------------


@pytest.mark.parametrize("model", BOTH)
def test_classify_stops_when_its_engine_stops_answering_and_the_next_run_reads_every_page(
    bench: Bench, tmp_path: Path, model: str
) -> None:
    stop_after(bench, 2)
    output = tmp_path / "out" / "classifications.jsonl"
    # The first call is answered, and nothing after it.
    bench.processor.dead_from_attempt = 2

    with pytest.raises(StageStopped) as stopped:
        bench.classify(output, model)

    message = str(stopped.value)
    assert all(part in message for part in ("What:", "Where:", "Expected:", "How to fix:"))
    assert "2 requests in a row failed" in message and "the engine has stopped answering" in message
    assert "STOP_AFTER_FAILED_REQUESTS" in message
    # The stage did not go on to record every page left as failed, and did not write its output.
    assert not output.exists()
    kept = read_jsonl(output.with_name(output.name + ".journal"))
    # What was read before the engine stopped is kept. A page that failed before the number allowed
    # was reached is kept too, with its error: fewer of those than the number allowed.
    assert 1 <= len(kept) < len(bench.names)
    assert sum(1 for record in kept if record.get("error")) < 2
    assert any(not record.get("error") for record in kept)

    bench.processor.dead_from_attempt = None
    bench.classify(output, model)

    records = records_by_name(output)
    assert set(records) == set(bench.names)
    assert not any(record.get("error") for record in records.values())
    assert [records[name]["document_type"] for name in bench.names] == [
        "BANK_STATEMENT",
        "INVOICE",
        "RECEIPT",
    ] * 2


def test_pages_recorded_as_failed_before_the_stage_stopped_are_read_again(
    bench: Bench, tmp_path: Path
) -> None:
    """With one page to a call a failed page is recorded before the next is tried, so some are on disk."""
    stop_after(bench, 3)
    bench.pages_to_a_call(1)
    out = tmp_path / "out"
    classifications, output = classified(bench, out), out / "raw_extractions.jsonl"
    bench.processor.dead_from_attempt = 2

    with pytest.raises(StageStopped, match="3 requests in a row failed"):
        bench.extract(classifications, output)

    kept = read_jsonl(output.with_name(output.name + ".journal"))
    assert [bool(record["error"]) for record in kept] == [False, True, True]

    bench.processor.dead_from_attempt = None
    bench.extract(classifications, output)

    records = records_by_name(output)
    assert set(records) == set(bench.names)
    assert not any(record["error"] for record in records.values())


def test_a_page_that_fails_among_pages_that_are_answered_does_not_stop_the_stage(
    bench: Bench, tmp_path: Path
) -> None:
    stop_after(bench, 2)
    out = tmp_path / "out"
    classifications, output = classified(bench, out), out / "raw_extractions.jsonl"
    # Two pages fail, each read on its own after the call it shared failed, and a page is answered
    # between them.
    bench.processor.cannot_read = {bench.names[0], bench.names[4]}

    bench.extract(classifications, output)

    records = records_by_name(output)
    assert sorted(name for name, record in records.items() if record["error"]) == [
        bench.names[0],
        bench.names[4],
    ]


def test_as_many_unreadable_pages_in_a_row_as_the_number_allowed_stop_the_stage(
    bench: Bench, tmp_path: Path
) -> None:
    """The rule cannot tell them from an engine that has stopped: the number allowed is the line."""
    stop_after(bench, 2)
    out = tmp_path / "out"
    # The two statements are read first, in one call, and neither can be read.
    bench.processor.cannot_read = {bench.names[0], bench.names[3]}

    with pytest.raises(StageStopped, match="2 requests in a row failed"):
        bench.extract(classified(bench, out), out / "raw_extractions.jsonl")

    stop_after(bench, 3)
    bench.extract(classified(bench, out), out / "raw_extractions.jsonl")

    records = records_by_name(out / "raw_extractions.jsonl")
    assert sorted(name for name, record in records.items() if record["error"]) == [
        bench.names[0],
        bench.names[3],
    ]


def test_a_job_whose_engine_stops_answering_ends_with_its_reason_and_exit_code_one(
    bench: Bench, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stop_after(bench, 2)
    bench.processor.dead_from_attempt = 1
    environ = {
        "KFP_TASK": "classify",
        "model": AN_ENGINE_ON_EACH_CARD,
        "image_dir": str(bench.pages),
        "output": str(tmp_path / "out"),
        "DOC_EXTRACT_LOG_DIR": str(tmp_path / "logs"),
    }
    monkeypatch.setattr(starter, "CONFIG_PATH", bench.config)
    monkeypatch.setattr("common.settings.DEFAULT_CONFIG", bench.config)
    monkeypatch.setattr(starter.os, "environ", environ)

    code = starter.main()

    printed = capsys.readouterr().out
    assert code == 1
    assert "FATAL: the task was stopped part way." in printed
    assert "2 requests in a row failed" in printed and "How to fix:" in printed
    assert not (tmp_path / "out" / "classifications.jsonl").exists()


# --- the split stage ------------------------------------------------------------------------------------

TWO_RECEIPTS = '[{"box_2d": [0, 0, 1000, 480], "label": "receipt"}, {"box_2d": [0, 520, 1000, 1000], "label": "receipt"}]'


def test_a_photograph_split_could_not_ask_about_is_asked_about_again(
    bench: Bench, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    stop_after(bench, 100)
    output = tmp_path / "out" / "split"
    bench.processor.dead_from_attempt = 1

    with caplog.at_level(logging.WARNING):
        split.run(output, split_checked(ONE_ENGINE, bench.config, bench.pages))

    assert "The model could not be asked about 6 of 6 photograph(s)" in caplog.text
    first = records_by_name(output / "splits.jsonl")
    assert set(first) == set(bench.names) and all(record["error"] for record in first.values())
    assert sorted(path.name for path in (output / "images").iterdir()) == bench.names

    # This time the model answers, and says each photograph holds two receipts.
    bench.processor.dead_from_attempt = None
    bench.processor.ask = lambda images, prompts, max_tokens: [TWO_RECEIPTS] * len(images)  # type: ignore[method-assign]
    split.run(output, split_checked(ONE_ENGINE, bench.config, bench.pages))

    second = records_by_name(output / "splits.jsonl")
    assert set(second) == set(bench.names) and not any(record.get("error") for record in second.values())
    written = sorted(path.name for path in (output / "images").iterdir())
    assert written == sorted(name for record in second.values() for name in record["outputs"])
    # Each photograph was cut in two, and the image it was passed through as the first time is gone:
    # only its cuts are there.
    assert all(record["action"] == "split" and len(record["outputs"]) == 2 for record in second.values())
    assert not set(written) & set(bench.names)


def test_split_stops_when_its_engine_stops_answering(bench: Bench, tmp_path: Path) -> None:
    stop_after(bench, 2)
    output = tmp_path / "out" / "split"
    bench.processor.dead_from_attempt = 2

    with pytest.raises(StageStopped):
        split.run(output, split_checked(ONE_ENGINE, bench.config, bench.pages))

    # The first call's two photographs were dealt with and are kept: the model's replies in the
    # stage's journal, and the images written.
    assert sorted(records_by_name(output / "splits.jsonl.journal")) == bench.names[:2]
    assert sorted(path.name for path in (output / "images").iterdir()) == bench.names[:2]
