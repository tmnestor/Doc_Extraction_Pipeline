"""A run's time on the cards is kept one way: a row for each stage, written by the stage.

Until 24 November 2026 it was kept three ways (a file of whole seconds written round each stage, a row
written by extract alone, and the pages' own times), with a rule to choose between them. The summary's
one figure, "Inference Time", was extract's model time on one run and both stages' whole time on another.

The times are invented.
"""

import json
import logging
from pathlib import Path

import pytest

from common import stage_timing
from common.stage_timing import StageTiming, note_stage_timing, read_stage_timings, run_time
from stages import evaluate


def pages(count: int, seconds: float, card: int | None = None) -> list[dict]:
    records = [{"image_name": f"page_{n}.png", "processing_time": seconds} for n in range(count)]
    if card is not None:
        for record in records:
            record["gpu_id"] = card
    return records


def test_one_engine_takes_the_sum_of_its_pages_times() -> None:
    assert stage_timing.inference_seconds_from_records(pages(4, 2.5), fallback=99.0) == 10.0


def test_engines_working_at_once_take_the_slowest_cards_time() -> None:
    """Adding the cards' times would give the work done, and understate the rate by the number of cards."""
    records = [*pages(4, 2.5, card=0), *pages(4, 3.0, card=1)]

    assert stage_timing.inference_seconds_from_records(records, fallback=99.0) == 12.0


def test_records_with_no_times_give_the_callers_own_clock_and_never_nothing() -> None:
    assert stage_timing.inference_seconds_from_records([{"image_name": "page.png"}], fallback=7.0) == 7.0
    assert stage_timing.inference_seconds_from_records([], fallback=7.0) == 7.0


def test_each_stage_notes_a_row_of_its_own(tmp_path: Path) -> None:
    note_stage_timing(
        tmp_path,
        stage="classify",
        records=pages(6, 1.0),
        wall_clock=40.0,
        execution_mode="single-engine",
        carried_on=False,
    )
    note_stage_timing(
        tmp_path,
        stage="extract",
        records=[*pages(3, 5.0, card=0), *pages(3, 6.0, card=1)],
        wall_clock=70.0,
        execution_mode="data-parallel (2 GPUs)",
        carried_on=False,
    )

    assert read_stage_timings(tmp_path) == [
        StageTiming("classify", 40.0, 6.0, "single-engine", 6),
        StageTiming("extract", 70.0, 18.0, "data-parallel (2 GPUs)", 6),
    ]
    assert not (tmp_path / ".inference_elapsed").exists()


def test_a_stage_run_afresh_writes_its_row_anew(tmp_path: Path) -> None:
    """Otherwise a second run into a folder would leave two rows, adding up to double the time."""
    for seconds in (5.0, 2.0):
        note_stage_timing(
            tmp_path,
            stage="extract",
            records=pages(4, seconds),
            wall_clock=60.0,
            execution_mode="single-engine",
            carried_on=False,
        )

    assert read_stage_timings(tmp_path) == [StageTiming("extract", 60.0, 8.0, "single-engine", 4)]


def test_a_stage_run_again_keeps_its_place_among_the_rows(tmp_path: Path) -> None:
    for stage in ("classify", "extract", "classify"):
        note_stage_timing(
            tmp_path,
            stage=stage,
            records=pages(2, 1.0),
            wall_clock=9.0,
            execution_mode="single-engine",
            carried_on=False,
        )

    assert [row.stage for row in read_stage_timings(tmp_path)] == ["classify", "extract"]


def test_a_stage_that_carried_on_adds_to_the_row_it_began(tmp_path: Path) -> None:
    """The row then covers every page the output holds, and is not set aside as stale when scored."""
    note_stage_timing(
        tmp_path,
        stage="extract",
        records=pages(4, 5.0),
        wall_clock=60.0,
        execution_mode="single-engine",
        carried_on=False,
    )
    note_stage_timing(
        tmp_path,
        stage="extract",
        records=pages(2, 5.0),
        wall_clock=45.0,
        execution_mode="single-engine",
        carried_on=True,
    )

    assert read_stage_timings(tmp_path) == [StageTiming("extract", 105.0, 30.0, "single-engine", 6)]


def test_a_stage_that_carried_on_with_no_earlier_row_writes_what_it_has(tmp_path: Path) -> None:
    note_stage_timing(
        tmp_path,
        stage="extract",
        records=pages(2, 5.0),
        wall_clock=45.0,
        execution_mode="single-engine",
        carried_on=True,
    )

    assert read_stage_timings(tmp_path) == [StageTiming("extract", 45.0, 10.0, "single-engine", 2)]


def test_a_note_that_cannot_be_written_is_logged_and_does_not_end_the_stage(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    in_the_way = tmp_path / "a_file"
    in_the_way.write_text("not a folder")

    with caplog.at_level(logging.WARNING):
        note_stage_timing(
            in_the_way,
            stage="extract",
            records=pages(2, 5.0),
            wall_clock=45.0,
            execution_mode="single-engine",
            carried_on=False,
        )

    assert "Could not record extract stage timing" in caplog.text


def test_a_row_that_cannot_be_read_is_passed_over(tmp_path: Path) -> None:
    good = {
        "stage": "extract",
        "wall_clock": 60.0,
        "inference_seconds": 20.0,
        "execution_mode": "x",
        "images": 4,
    }
    (tmp_path / ".stage_timing.jsonl").write_text(
        f"not json\n{json.dumps(good)}\n{json.dumps({'stage': 1})}\n"
    )

    assert read_stage_timings(tmp_path) == [StageTiming("extract", 60.0, 20.0, "x", 4)]
    assert read_stage_timings(tmp_path / "no_such_folder") == []


def test_the_runs_time_is_the_stages_model_times_together() -> None:
    rows = [
        StageTiming("classify", 120.0, 31.9, "data-parallel (2 GPUs)", 165),
        StageTiming("extract", 288.9, 200.4, "data-parallel (2 GPUs)", 165),
    ]

    time = run_time(rows, per_image_total=396.7, pages=165, read=165)

    assert time.model_seconds == pytest.approx(232.3)
    assert time.source == "stage timing (this run)"
    assert time.wall_clock == pytest.approx(408.9) and time.startup_seconds == pytest.approx(176.6)
    assert time.execution_mode == "data-parallel (2 GPUs)"


def test_rows_for_another_number_of_pages_are_stale_and_the_pages_own_times_are_used(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rows = [StageTiming("extract", 60.0, 20.0, "single-engine", 12)]

    with caplog.at_level(logging.WARNING):
        time = run_time(rows, per_image_total=396.7, pages=165, read=165)

    assert time.model_seconds == 396.7 and time.stages == ()
    assert "unreliable under data parallelism" in time.source
    assert "stale" in caplog.text


def test_with_no_record_the_time_is_not_recorded_and_is_never_nothing() -> None:
    time = run_time([], per_image_total=0.0, pages=165, read=165)

    assert time.model_seconds is None and time.source == "not recorded"


@pytest.mark.parametrize("not_read", [0, 1, 2, 5])
def test_pages_left_unread_do_not_make_the_rows_stale(not_read: int) -> None:
    """Classify counts every page and extract those it read. Until 25 November 2026 the mean of the two
    was set beside the pages read, so two pages left unread made the rows of a whole run stale.
    """
    rows = [
        StageTiming("classify", 100.0, 30.0, "single-engine", 165),
        StageTiming("extract", 300.0, 200.0, "single-engine", 165 - not_read),
    ]

    time = run_time(rows, per_image_total=230.0, pages=165, read=165 - not_read)

    assert time.source == "stage timing (this run)" and time.model_seconds == pytest.approx(230.0)


@pytest.mark.parametrize(
    "rows",
    [
        [StageTiming("classify", 100.0, 30.0, "x", 160), StageTiming("extract", 300.0, 200.0, "x", 165)],
        [StageTiming("classify", 100.0, 30.0, "x", 165), StageTiming("extract", 300.0, 200.0, "x", 170)],
    ],
)
def test_a_row_that_covers_another_number_than_its_own_stage_read_is_stale(
    rows: list[StageTiming],
) -> None:
    """The mean of these two rows is right, or near it. Each is judged by itself."""
    assert "unreliable" in run_time(rows, per_image_total=230.0, pages=165, read=165).source


def test_a_row_of_another_stage_is_kept_apart_and_judged_by_nothing() -> None:
    rows = [
        StageTiming("split", 90.0, 21.0, "data-parallel (2 GPUs)", 57),
        StageTiming("classify", 100.0, 30.0, "single-engine", 165),
        StageTiming("extract", 300.0, 200.0, "single-engine", 165),
    ]

    time = run_time(rows, per_image_total=230.0, pages=165, read=165)

    assert [t.stage for t in time.stages] == ["classify", "extract"]
    assert [t.stage for t in time.others] == ["split"]
    assert time.model_seconds == pytest.approx(230.0) and time.wall_clock == pytest.approx(400.0)


def test_stages_arranged_differently_are_each_named() -> None:
    rows = [
        StageTiming("classify", 10.0, 5.0, "single-engine", 4),
        StageTiming("extract", 20.0, 9.0, "data-parallel (2 GPUs)", 4),
    ]

    assert run_time(rows, per_image_total=0.0, pages=4, read=4).execution_mode == (
        "classify=single-engine, extract=data-parallel (2 GPUs)"
    )


# --- the summary's rows ---------------------------------------------------------------------------


def test_the_summary_says_which_stages_every_time_is_of() -> None:
    rows = [
        StageTiming("classify", 120.0, 30.0, "single-engine", 60),
        StageTiming("extract", 300.0, 90.0, "single-engine", 60),
    ]

    shown = dict(evaluate.time_rows(run_time(rows, per_image_total=0.0, pages=60, read=60), 60))

    assert shown["Classify, model time"] == "30.0s"
    assert shown["Extract, model time"] == "90.0s"
    assert shown["Model Time (classify + extract)"] == "120.0s"
    # Pages a minute through both stages: 60 pages in 120 seconds.
    assert shown["Throughput"] == "30.00 images/min"
    assert shown["Timing Source"] == "stage timing (this run)"
    assert shown["Total Wall Clock"] == "420.0s" and shown["Engine Startup"] == "300.0s"
    # The old label named no stage, and meant two different things.
    assert "Inference Time" not in shown


def test_an_older_runs_folder_with_extracts_row_alone_says_so() -> None:
    rows = [StageTiming("extract", 300.0, 90.0, "single-engine", 60)]

    shown = dict(evaluate.time_rows(run_time(rows, per_image_total=0.0, pages=60, read=60), 60))

    assert shown["Model Time (extract)"] == "90.0s"
    assert shown["Throughput"] == "40.00 images/min"


def test_a_run_with_no_time_recorded_shows_none_and_no_rate() -> None:
    shown = dict(evaluate.time_rows(run_time([], per_image_total=0.0, pages=60, read=60), 60))

    assert shown == {
        "Model Time": "not recorded",
        "Throughput": "not recorded",
        "Timing Source": "not recorded",
    }


def test_the_summary_shows_splits_time_on_a_row_of_its_own_and_not_in_the_rate() -> None:
    """A photograph is not a page: the rate is of the stages that read pages, as it was for every
    recorded run.
    """
    rows = [
        StageTiming("split", 90.0, 21.0, "data-parallel (2 GPUs)", 57),
        StageTiming("classify", 100.0, 30.0, "single-engine", 165),
        StageTiming("extract", 300.0, 200.0, "single-engine", 165),
    ]

    shown = dict(evaluate.time_rows(run_time(rows, per_image_total=0.0, pages=165, read=165), 165))

    assert shown["Split, model time (57 photographs, not in the rate)"] == "21.0s"
    assert shown["Model Time (classify + extract)"] == "230.0s"
    assert shown["Throughput"] == "43.04 images/min"
    assert shown["Total Wall Clock"] == "400.0s"


def test_a_run_with_no_row_of_split_shows_none() -> None:
    rows = [StageTiming("classify", 100.0, 30.0, "x", 60), StageTiming("extract", 300.0, 90.0, "x", 60)]

    shown = dict(evaluate.time_rows(run_time(rows, per_image_total=0.0, pages=60, read=60), 60))

    assert not [label for label in shown if label.startswith("Split")]
