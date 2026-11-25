"""A stage that stops part way keeps what it had read, however its engines are arranged.

Until 24 November 2026 only the extract stage with one engine wrote each record as it was made. The
classify stage, the split stage, and the extract stage with an engine on each card wrote when they had
read every page, so a run that stopped part way lost the whole stage: on the quantised 12B, which runs an
engine on each card, that was every run.

Now each record goes into a journal beside the stage's output the moment it is made. A journal found
when a stage starts is what an earlier run had read: it is added to the output, in the pages' own order,
and the stage carries on from there.

The pages are blank images made here, and every reply is invented.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reference"))

from model_inputs import test_config  # noqa: E402

from common import vllm_dp  # noqa: E402
from common.stage_settings import resolve  # noqa: E402
from common.stage_timing import read_stage_timings  # noqa: E402
from stages import __main__ as starter  # noqa: E402
from stages import split  # noqa: E402
from stages.classify import classify_pages  # noqa: E402
from stages.io import Journal, read_jsonl  # noqa: E402

test_config.__test__ = False  # type: ignore[attr-defined]

from bench import AN_ENGINE_ON_EACH_CARD, KINDS, ONE_ENGINE, Bench, Stopped, split_checked  # noqa: E402


def names_in(path: Path) -> list[str]:
    return [record["image_name"] for record in read_jsonl(path)]


def without_times(path: Path) -> list[dict[str, Any]]:
    return [{k: v for k, v in record.items() if k != "processing_time"} for record in read_jsonl(path)]


# --- the journal --------------------------------------------------------------------------------------


def test_a_record_is_on_disk_as_soon_as_it_is_written(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "classifications.jsonl")

    with journal:
        journal.write({"image_name": "page_2.png"})
        assert json.loads(journal.path.read_text()) == {"image_name": "page_2.png"}

    # Closing it does not remove it: only the stage does, once its output is written.
    assert journal.path == tmp_path / "classifications.jsonl.journal" and journal.path.is_file()
    journal.discard()
    assert not journal.path.exists()


def test_what_an_interrupted_run_had_read_is_added_to_the_output_in_the_pages_order(tmp_path: Path) -> None:
    output = tmp_path / "classifications.jsonl"
    output.write_text(json.dumps({"image_name": "page_1.png", "from": "the output"}) + "\n")
    journal = Journal(output)
    journal.path.write_text(
        "".join(
            json.dumps({"image_name": name, "from": "the journal"}) + "\n"
            for name in ("page_4.png", "page_1.png", "page_2.png", "page_4.png", "page_9.png")
        )
        + '{"image_name": "page_3.png", "from": "the jour'  # the line being written when the run stopped
    )

    recovered = journal.recover(["page_1.png", "page_2.png", "page_3.png", "page_4.png"])

    # In the pages' order, each once, without the page the output already held or the half-written line.
    # A page that is not one of this run's goes last.
    assert [record["image_name"] for record in recovered] == ["page_2.png", "page_4.png", "page_9.png"]
    assert names_in(output) == ["page_1.png", "page_2.png", "page_4.png", "page_9.png"]
    assert read_jsonl(output)[0]["from"] == "the output"
    assert not journal.path.exists()


def test_with_no_journal_nothing_is_recovered_and_the_output_is_not_touched(tmp_path: Path) -> None:
    output = tmp_path / "classifications.jsonl"

    assert Journal(output).recover(["page_1.png"]) == []
    assert not output.exists()


# --- one engine ---------------------------------------------------------------------------------------


def test_classify_stopped_part_way_keeps_its_pages_and_the_next_run_reads_only_the_rest(
    bench: Bench, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    output = tmp_path / "out" / "classifications.jsonl"
    bench.processor.stop_at_call = 3

    with pytest.raises(Stopped):
        bench.classify(output)

    # Two calls of two pages were finished. They are in the journal, and the output is not yet written.
    assert not output.exists()
    assert names_in(Journal(output).path) == bench.names[:4]

    bench.processor.stop_at_call = None
    bench.processor.calls.clear()
    with caplog.at_level(logging.INFO):
        bench.classify(output)

    assert bench.processor.asked_about == bench.names[4:]
    assert names_in(output) == bench.names
    assert not Journal(output).path.exists()
    assert "Recovered 4 page(s)" in caplog.text
    # The row of the stage's time covers every page, the recovered ones too, so it is not stale when scored.
    (row,) = [row for row in read_stage_timings(output.parent) if row.stage == "classify"]
    assert row.images == len(bench.names) and "a run that stopped" in row.execution_mode


def test_a_run_that_was_stopped_and_carried_on_gives_what_a_run_straight_through_gives(
    bench: Bench, tmp_path: Path
) -> None:
    straight = bench.classify(tmp_path / "straight" / "classifications.jsonl")
    bench.extract(straight, tmp_path / "straight" / "raw_extractions.jsonl")

    stopped = tmp_path / "stopped"
    bench.processor.stop_at_call = len(bench.processor.calls) + 2
    with pytest.raises(Stopped):
        bench.classify(stopped / "classifications.jsonl")
    bench.processor.stop_at_call = None
    classified = bench.classify(stopped / "classifications.jsonl")
    bench.processor.stop_at_call = len(bench.processor.calls) + 3
    with pytest.raises(Stopped):
        bench.extract(classified, stopped / "raw_extractions.jsonl")
    assert not (stopped / "raw_extractions.jsonl").exists()
    assert len(names_in(Journal(stopped / "raw_extractions.jsonl").path)) == 4
    bench.processor.stop_at_call = None
    bench.extract(classified, stopped / "raw_extractions.jsonl")

    def moved(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The same records, whichever of the two folders their pages' paths were read from."""
        return [{k: v for k, v in record.items() if k != "image_path"} for record in records]

    for file in ("classifications.jsonl", "raw_extractions.jsonl"):
        assert moved(without_times(stopped / file)) == moved(without_times(tmp_path / "straight" / file))
        assert not Journal(stopped / file).path.exists()


def test_extract_run_straight_through_leaves_its_output_and_no_journal(
    bench: Bench, tmp_path: Path
) -> None:
    classified = bench.classify(tmp_path / "out" / "classifications.jsonl")
    output = bench.extract(classified, tmp_path / "out" / "raw_extractions.jsonl")

    # One type after another, in the order the settings give, each page once.
    assert names_in(output) == sorted(bench.names, key=lambda name: (KINDS.index(name[8:-4]), name))
    assert not Journal(output).path.exists()


# --- an engine on each card ---------------------------------------------------------------------------


def test_with_an_engine_on_each_card_the_output_is_in_the_pages_order(bench: Bench, tmp_path: Path) -> None:
    """The cards finish in their own order. The output's order decides how the next stage groups pages."""
    output = bench.classify(tmp_path / "out" / "classifications.jsonl", model=AN_ENGINE_ON_EACH_CARD)

    assert names_in(output) == bench.names
    assert [record["gpu_id"] for record in read_jsonl(output)] == [0, 1, 0, 1, 0, 1]
    assert not Journal(output).path.exists()


def extract_calls(bench: Bench, classifications: Path, into: Path) -> list[list[str]]:
    """The engine calls extract makes of a classifications file, each as the pages it held."""
    before = len(bench.processor.calls)
    bench.extract(classifications, into / "raw_extractions.jsonl", model=AN_ENGINE_ON_EACH_CARD)
    return bench.processor.calls[before:]


def test_classify_stopped_between_its_cards_leaves_the_file_a_straight_run_leaves(
    bench: Bench, tmp_path: Path
) -> None:
    """Card 0 had finished and card 1 had not. Until 25 November 2026 the file came out as pages 0, 2,
    4, 1, 3, 5, and extract then put an invoice on the other card than a run straight through would.

    The bench runs a worker inside the test's own process, so a run killed between the cards cannot be
    staged by stopping it. What such a stop leaves is put there instead: card 0's share in the journal.
    """
    straight = bench.classify(tmp_path / "straight" / "classifications.jsonl", model=AN_ENGINE_ON_EACH_CARD)
    output = tmp_path / "stopped" / "classifications.jsonl"
    output.parent.mkdir()
    with Journal(output) as journal:
        for record in read_jsonl(straight)[0::2]:
            journal.write(record)
    bench.processor.calls.clear()

    bench.classify(output, model=AN_ENGINE_ON_EACH_CARD)

    assert sorted(bench.processor.asked_about) == sorted(bench.names[1::2])
    assert names_in(output) == bench.names
    assert extract_calls(bench, output, tmp_path / "b") == extract_calls(bench, straight, tmp_path / "a")


def read_and_then_fall_over(
    processor: Any, pages: list[Path], settings: Any, *, sink: Any = None
) -> list[dict[str, Any]]:
    """Stands in for a card whose engine dies: card 1 reads its first call's pages and then fails."""
    if any("CASE001_receipt" in page.name for page in pages):
        classify_pages(processor, pages[:2], settings, sink=sink)
        raise RuntimeError("the card ran out of memory")
    return classify_pages(processor, pages, settings, sink=sink)


def test_when_a_card_fails_what_every_card_had_read_is_kept(bench: Bench, tmp_path: Path) -> None:
    from common.settings import Settings

    output = tmp_path / "out" / "classifications.jsonl"
    settings = Settings.load(
        {
            "output_dir": str(output.parent),
            "model_type": AN_ENGINE_ON_EACH_CARD,
        },
        config_path=bench.config,
    ).for_stage("classify")
    pages = [bench.pages / name for name in bench.names]
    journal = Journal(output)

    with journal, pytest.raises(RuntimeError, match="the card ran out of memory"):
        vllm_dp.run_dp(
            num_gpus=2, items=pages, read=read_and_then_fall_over, settings=settings, sink=journal
        )

    # Card 0 read its three pages, and card 1 two of its three before it failed.
    kept = read_jsonl(journal.path)
    assert sorted(record["image_name"] for record in kept) == sorted(
        [*bench.names[0::2], *bench.names[1::2][:2]]
    )
    assert all(record["gpu_id"] in (0, 1) and record["processing_time"] >= 0.0 for record in kept)

    # The next run adds them to the output in the pages' order, and has one page left to read.
    bench.classify(output, model=AN_ENGINE_ON_EACH_CARD)

    assert names_in(output) == [*sorted(r["image_name"] for r in kept), bench.names[5]]
    assert bench.processor.asked_about.count(bench.names[5]) == 1
    assert not journal.path.exists()


# --- the split stage ----------------------------------------------------------------------------------


def test_split_stopped_part_way_keeps_the_photographs_it_had_finished(bench: Bench, tmp_path: Path) -> None:
    output = tmp_path / "out" / "split"
    bench.processor.stop_at_call = 3

    with pytest.raises(Stopped):
        split.run(bench.pages, output, split_checked(ONE_ENGINE, bench.config))

    # Two calls of two photographs were asked and acted on before the run stopped. Their replies are in
    # the journal, their images are written, and the stage's own file is not yet there.
    assert not (output / "splits.jsonl").exists()
    assert names_in(output / "splits.jsonl.journal") == bench.names[:4]
    assert "raw_response" in read_jsonl(output / "splits.jsonl.journal")[0]
    assert sorted(path.name for path in (output / "images").iterdir()) == bench.names[:4]

    bench.processor.stop_at_call = None
    bench.processor.calls.clear()
    split.run(bench.pages, output, split_checked(ONE_ENGINE, bench.config))

    assert bench.processor.calls == [["2 photographs"]]
    assert names_in(output / "splits.jsonl") == bench.names
    assert not (output / "splits.jsonl.journal").exists()


# --- starting afresh ----------------------------------------------------------------------------------


def test_a_stage_started_afresh_has_its_journal_deleted_with_its_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise the pages of the run that stopped would come back into the run meant to replace it."""
    out = tmp_path / "out"
    out.mkdir()
    for name in ("classifications.jsonl", "classifications.jsonl.journal", "raw_extractions.jsonl.journal"):
        (out / name).write_text("{}\n")
    monkeypatch.setattr("stages.classify.run", lambda *args, **kwargs: None)
    monkeypatch.setattr("stages.classify.check", lambda task: None)
    environ = {
        "model": ONE_ENGINE,
        "image_dir": str(tmp_path / "pages"),
        "output": str(out),
        "DOC_EXTRACT_LOG_DIR": str(tmp_path / "logs"),
        "CLEAR_PREV_OUTPUT": "true",
    }
    config = Path(__file__).resolve().parents[2] / "config" / "run_config.yml"

    starter.run_task(resolve("classify", environ, config))

    assert not (out / "classifications.jsonl").exists()
    assert not (out / "classifications.jsonl.journal").exists()
    # Another stage's journal is that stage's, and is left.
    assert (out / "raw_extractions.jsonl.journal").exists()
