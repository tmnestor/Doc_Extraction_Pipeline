"""One way for a stage that uses the cards to carry on from what an earlier run left.

``split``, ``classify`` and ``extract`` each read items with the model and write a record for each. A run
may be the first, or may follow one that was stopped, or one that left pages it could not read, or one
after which more pages arrived. Whichever it is, the stage describes its work (``Work``) and calls
``carry_on``, and its output comes out as a run straight through would have left it: one record for each
item, in the stage's own order.

Until 25 November 2026 each of the three wrote this out for itself, and the three did not agree. A fix
to carrying on was a fix written three times, and ``split`` was the last to get each. Two of them added
what a run recovered, then what it read, then what it read again, each after what was there, so a
classify stopped between its cards left its pages in another order, and extract then grouped them into
other engine calls than a run straight through would.

This is a function a stage calls. It knows nothing of pages, types or photographs: what an item is, how
it is read and what is made of the reply are the stage's, handed over in ``Work``.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.stage_settings import StageSettings
from common.stage_timing import inference_seconds_from_records, note_recovered, note_stage_timing
from common.vllm_dp import Reading, read_on_the_cards

from . import records
from .io import Journal, in_the_order, read_completed_images, take_out

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Work:
    """What a stage has to read, and where its records go."""

    # "split", "classify" or "extract": the name of its row of time.
    stage: str
    # The stage's file of records.
    output: Path
    # Every item of this run, in the order the output holds them. An item is whatever the reading
    # function is given: a page, a photograph, a page's classification record.
    items: list[Any]
    # An item's name: the ``image_name`` of its record.
    name_of: Callable[[Any], str]
    # How an item's record is come by: exactly one of these two.
    # The stage's reading function, at the top of its module (``common.vllm_dp.Reading``): the items
    # are read on the cards.
    read: Reading | None = None
    # Makes an item's record with no model asked, for a stage that can say what an item is without
    # one: a page whose dataset gives its type, a photograph that is passed through. No engine is
    # started, and there is no time on the cards to note. Until 26 November 2026 each such stage
    # wrote its own carrying on, and one of them kept the record of a page an earlier run had failed.
    tell: Callable[[Any], dict[str, Any]] | None = None
    # Says of a record in the output whether its item is to be read again.
    read_again: Callable[[dict[str, Any]], bool] = records.failed
    # Told of the records taken out to be read again, before anything is read.
    taken_out: Callable[[list[dict[str, Any]]], None] | None = None
    # Makes the stage's record of what the reading function gave, for a stage whose output holds what
    # was done about a reply and not the reply. It is called once for each reply, when the reply is
    # given, and never by a run that carries on. None: the output holds what the reading function gave.
    made_of: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    # The run's output folder, where the row of the stage's time is noted. None: no row.
    timing: Path | None = None
    # What an item is called in the log.
    thing: str = "page"


@dataclass(frozen=True)
class Carried:
    """What carrying on gave."""

    # The records this run added to the output, in the output's order.
    records: list[dict[str, Any]]
    # What the reading function gave for them: the same, unless the stage makes its record of a reply.
    read: list[dict[str, Any]]
    # How the engines were arranged. None when no engine was started: nothing was left to read, or
    # the stage's records are made with no model asked (``Work.tell``).
    arrangement: str | None
    # The model's time, apart from starting the engine, and the whole time of the reading.
    model_seconds: float
    wall_clock: float
    # How many items the output held before this run read any, those recovered among them.
    already_done: int
    # How many of those were recovered from the journal of a run that stopped.
    recovered: int
    # How many records were taken out of the output to be read again.
    taken_out: int


class _Kept:
    """Keeps each record as it is made, in the journal.

    For a stage that makes its own record of a reply, the reply is acted on first and the two are
    kept together: a run that carries on then has the record, and does nothing about the reply again.
    """

    def __init__(
        self, journal: Journal, made_of: Callable[[dict[str, Any]], dict[str, Any]] | None
    ) -> None:
        self.made: list[dict[str, Any]] = []
        self._journal, self._made_of = journal, made_of

    def write(self, record: dict[str, Any]) -> None:
        made = self._made_of(record) if self._made_of is not None else None
        self._journal.write(record, made)
        self.made.append(record if made is None else made)


def carry_on(settings: StageSettings | None, work: Work) -> Carried:
    """Read what a stage has left to read, and leave its output as a run straight through would.

    In this order:

    1. What an earlier run had read before it stopped is recovered from the journal into the output.
    2. The records of this run's items that are to be read again are taken out of the output.
    3. The items whose names the output does not hold are what is left to read.
    4. With nothing left, the output is put in order and no engine is started.
    5. Otherwise they are read on the cards. Each record goes into the journal the moment it is made.
       A stage whose records are made with no model asked (``Work.tell``) has each of them made
       here instead: no engine, no journal, and no time to note.
    6. The output is put in place whole, with the records added and in the stage's order, and only
       then is the journal removed: a run stopped anywhere before leaves the earlier output as it
       was and the next run able to recover.
    7. The stage's time is noted.

    Args:
        settings: The run's settings. None is allowed only for work whose records are made with no model asked.
        work: What the stage has to read, and where its records go.

    Returns:
        What this run added, and how.

    Raises:
        ConfigError, StageStopped: As ``read_on_the_cards`` raises them. The journal is kept.
        ValueError: If the work gives both a reading function and a way to make a record without
            one, or neither.
    """
    if (work.read is None) == (work.tell is None):
        msg = (
            "Work takes exactly one of read (the items are read on the cards) and tell (no model is asked)"
        )
        raise ValueError(msg)
    order = [work.name_of(item) for item in work.items]
    journal = Journal(work.output)

    recovered = journal.recover(order)
    if recovered:
        logger.info(
            "Recovered %d %s(s) an earlier run had finished before it stopped", len(recovered), work.thing
        )
        if work.timing is not None:
            note_recovered(work.timing, stage=work.stage, records=recovered)

    # Only a record of one of this run's pages is taken out: a page that is no longer among them
    # cannot be read again, and taking its record out would only lose it.
    of_this_run = set(order)
    taken = take_out(
        work.output, lambda record: record.get("image_name") in of_this_run and work.read_again(record)
    )
    if taken and work.taken_out is not None:
        work.taken_out(taken)

    done = read_completed_images(work.output)
    left = [item for item in work.items if work.name_of(item) not in done]
    if done:
        logger.info(
            "Resuming: %d already done, %d remaining (of %d total)", len(done), len(left), len(work.items)
        )
    if not left:
        in_the_order(work.output, order)
        logger.info("Every %s is already done: nothing to do", work.thing)
        return Carried([], [], None, 0.0, 0.0, len(done), len(recovered), len(taken))

    if work.tell is not None:
        told = [work.tell(item) for item in left]
        in_the_order(work.output, order, adding=told)
        return Carried(told, [], None, 0.0, 0.0, len(done), len(recovered), len(taken))
    assert work.read is not None and settings is not None

    kept = _Kept(journal, work.made_of)
    started = time.time()
    with journal:
        read = read_on_the_cards(settings, left, work.read, sink=kept)
    wall_clock = time.time() - started

    # With an engine on each card the records arrive as the cards make them: they are added in the
    # order of the items.
    place = {work.name_of(item): position for position, item in enumerate(left)}
    made = sorted(kept.made, key=lambda record: place.get(record.get("image_name", ""), len(place)))
    in_the_order(work.output, order, adding=made)
    journal.discard()

    if work.timing is not None:
        note_stage_timing(
            work.timing,
            stage=work.stage,
            records=read.records,
            wall_clock=wall_clock,
            execution_mode=read.arrangement,
            carried_on=bool(done),
            read_again=len(taken),
        )
    return Carried(
        records=made,
        read=read.records,
        arrangement=read.arrangement,
        model_seconds=inference_seconds_from_records(read.records, fallback=wall_clock),
        wall_clock=wall_clock,
        already_done=len(done),
        recovered=len(recovered),
        taken_out=len(taken),
    )
