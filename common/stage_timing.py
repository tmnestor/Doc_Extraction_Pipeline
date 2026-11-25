"""A run's time on the cards: one row for each stage that uses them, kept in the output folder.

``classify`` and ``extract`` each note what they spent in ``<output>/.stage_timing.jsonl``, and the
evaluate stage reads the rows for its summary. That file is the one record of a run's time.

A row holds two times, because they answer different questions:

* ``inference_seconds`` is the model's time: reading pages, apart from starting the engine. On a short
  run starting the engine is a large share of the whole (41% on one run of 347 receipts), so a rate
  worked out from the whole understates the model badly. With an engine on each card the cards work at
  once, so this is the SLOWEST card's time and not the sum: the sum is the work done, and would
  understate the rate by about the number of cards.
* ``wall_clock`` is the whole of the stage, engine start and all.

Until 24 November 2026 a run's time was kept three ways: a file of whole seconds written round each
stage, this file, written by extract alone, and the pages' own times, with a rule to choose between
them. So the summary's one figure meant extract's model time on one run and both stages' whole time on
another.

Timing is a measurement about a run and never a reason to fail one: a row that cannot be written or read
is logged and passed over.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_FILENAME = ".stage_timing.jsonl"


@dataclass(frozen=True)
class StageTiming:
    """What one stage spent on the cards."""

    stage: str
    wall_clock: float
    inference_seconds: float
    execution_mode: str
    images: int


def inference_seconds_from_records(records: list[dict], *, fallback: float) -> float:
    """The model's time for a stage, from its records' own times.

    Records that carry ``gpu_id`` came from engines working at once, one on each card, so the time taken
    is the slowest card's total. Without ``gpu_id`` the pages were read by one engine and the sum is the
    time taken.

    Args:
        records: The stage's records, each with ``processing_time``.
        fallback: What to give when no record was timed, such as the caller's own clock. Never 0.0,
            which would turn into an absurd rate.
    """
    per_card: dict[object, float] = {}
    timed = False
    for record in records:
        elapsed = record.get("processing_time")
        if elapsed is None:
            continue
        timed = True
        per_card[record.get("gpu_id")] = per_card.get(record.get("gpu_id"), 0.0) + float(elapsed)
    return max(per_card.values()) if timed else fallback


def note_stage_timing(
    output_dir: Path,
    *,
    stage: str,
    records: list[dict],
    wall_clock: float,
    execution_mode: str,
    carried_on: bool,
    read_again: int = 0,
) -> None:
    """Note what a stage has just spent. Never raises: the stage's work is already done.

    Args:
        output_dir: The run's output folder.
        stage: The stage's name.
        records: The records the stage has just made, each with its ``processing_time``.
        wall_clock: The whole time the stage took, engine start and all.
        execution_mode: How the engines were arranged.
        carried_on: Whether the stage carried on from an earlier run's output. Its time is then added
            to that run's row, so that the row covers every page the output holds. Otherwise the row is
            written anew.
        read_again: How many of ``records`` are of pages the earlier run had recorded as failed, whose
            records were taken out of the output. The earlier row counts them already, so they are
            not counted twice; the time first spent on them stays in the row.
    """
    try:
        now = StageTiming(
            stage=stage,
            wall_clock=wall_clock,
            inference_seconds=inference_seconds_from_records(records, fallback=wall_clock),
            execution_mode=execution_mode,
            images=len(records),
        )
        earlier = [t for t in read_stage_timings(output_dir) if t.stage == stage]
        if carried_on and earlier:
            before = earlier[-1]
            modes = (
                now.execution_mode
                if before.execution_mode == now.execution_mode
                else f"{before.execution_mode}, then {now.execution_mode}"
            )
            now = StageTiming(
                stage=stage,
                wall_clock=before.wall_clock + now.wall_clock,
                inference_seconds=before.inference_seconds + now.inference_seconds,
                execution_mode=modes,
                images=before.images + now.images - read_again,
            )
        write_stage_timing(output_dir, now)
    except Exception as err:  # noqa: BLE001 - timing must not break the stage
        logger.warning("Could not record %s stage timing: %s", stage, err)


def note_recovered(output_dir: Path, *, stage: str, records: list[dict]) -> None:
    """Note the time of pages recovered from a run that stopped before it could note its own.

    The run that stopped left no row, or left one that does not cover these pages. Their own times are
    added, so that the stage's row covers every page its output holds. Of that run's whole time nothing is
    known but the model's.
    """
    note_stage_timing(
        output_dir,
        stage=stage,
        records=records,
        wall_clock=inference_seconds_from_records(records, fallback=0.0),
        execution_mode="a run that stopped",
        carried_on=True,
    )


def write_stage_timing(output_dir: Path, timing: StageTiming) -> Path:
    """Write one stage's row, in the place of any earlier row for that stage. Returns the file."""
    path = output_dir / _FILENAME
    rows = read_stage_timings(output_dir)
    if any(t.stage == timing.stage for t in rows):
        rows = [timing if t.stage == timing.stage else t for t in rows]
    else:
        rows.append(timing)
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(t)) + "\n" for t in rows), encoding="utf-8")
    return path


def read_stage_timings(output_dir: Path) -> list[StageTiming]:
    """Every stage's row for a run, in the order written. Empty when there is no file."""
    path = output_dir / _FILENAME
    if not path.is_file():
        return []
    timings = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            timings.append(StageTiming(**json.loads(line)))
        except (json.JSONDecodeError, TypeError) as err:
            logger.warning("Skipping unreadable stage timing row: %s", err)
    return timings


@dataclass(frozen=True)
class RunTime:
    """A run's time on the cards, for the evaluate stage's summary, with where it came from.

    ``source`` is shown beside the figure: a time can come from two places of very different quality,
    and a figure with no label invites the reader to trust the worse.
    """

    # The model's time over the stages, or None if nothing was recorded. Never 0.0.
    model_seconds: float | None
    source: str
    # The rows of the stages that read pages, when they are what the time is from; empty otherwise.
    stages: tuple[StageTiming, ...]
    # The rows of any other stage, such as split, which reads photographs. They are in no figure above:
    # a photograph is not a page, and a rate of pages a minute is of the stages that read pages.
    others: tuple[StageTiming, ...] = ()

    @property
    def wall_clock(self) -> float:
        return sum(t.wall_clock for t in self.stages)

    @property
    def startup_seconds(self) -> float:
        """The whole time less the model's: starting the engines. Never below nothing."""
        return max(0.0, self.wall_clock - (self.model_seconds or 0.0))

    @property
    def execution_mode(self) -> str:
        modes = {t.execution_mode for t in self.stages}
        if len(modes) == 1:
            return modes.pop()
        return ", ".join(f"{t.stage}={t.execution_mode}" for t in self.stages)


def run_time(timings: list[StageTiming], *, per_image_total: float, pages: int, read: int) -> RunTime:
    """A run's time on the cards, from the best record there is.

    1. **The stages' own rows.** The model's time, apart from engine start, and right for either
       arrangement of engines. Set aside as stale when a row covers another number of pages than its
       own stage dealt with, which is what an output folder scored again after a part run would give.
       Each row is judged by itself: ``classify`` gives every page a type, and ``extract`` reads those
       the settings do not say to leave. Until 25 November 2026 the mean of the two rows was set beside
       the pages read, so two pages left unread made the rows of a whole run stale. A row of any other
       stage is judged by nothing here and kept apart (``RunTime.others``).
    2. **The sum of the pages' own times.** With an engine on each card this is the work done and not
       the time taken, so it understates the rate by about the number of cards. It is labelled so.

    With neither, the time is not recorded: never 0.0.

    Args:
        timings: The rows of the run's output folder.
        per_image_total: The sum of the scored pages' own times.
        pages: How many pages the run has a record of: what ``classify`` dealt with.
        read: How many of them were read: what ``extract`` dealt with.
    """
    covers = {"classify": pages, "extract": read}
    of_the_pages = [t for t in timings if t.stage in covers]
    others = tuple(t for t in timings if t.stage not in covers)
    if of_the_pages and sum(t.inference_seconds for t in of_the_pages) > 0:
        wrong = [t for t in of_the_pages if t.images != covers[t.stage]]
        if not wrong:
            return RunTime(
                sum(t.inference_seconds for t in of_the_pages),
                "stage timing (this run)",
                tuple(of_the_pages),
                others,
            )
        logger.warning(
            "Stage timing is stale and is set aside: %s",
            "; ".join(
                f"the row of {t.stage} covers {t.images} images where {covers[t.stage]} are being evaluated"
                for t in wrong
            ),
        )
    if per_image_total > 0:
        return RunTime(per_image_total, "per-image sum (unreliable under data parallelism)", (), others)
    return RunTime(None, "not recorded", (), others)
