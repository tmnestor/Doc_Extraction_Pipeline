"""Where the engine is started: one engine across the cards, or one whole engine on each card.

A stage that reads pages calls ``read_on_the_cards`` with its pages and its own reading function. This
module decides the arrangement and starts the engine or engines; the reading function is the same either
way, so the two arrangements cannot come to differ in what they ask.

With an engine on each card, one process is started for each card and pinned to it with
``CUDA_VISIBLE_DEVICES``, and each is given every n-th page. There is no NCCL, no /dev/shm and nothing
passed between the processes while they read.

Nothing here knows what a page is or what is asked of it.
"""

import logging
import multiprocessing as mp
import os
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from queue import Empty
from typing import Any, Protocol

from common.cards import cards_to_use
from common.config_schema import ConfigError
from common.engine_calls import StageStopped
from common.settings import Settings

logger = logging.getLogger(__name__)

# A stage's reading function: given the document processor, its pages and the run's settings, returns
# one record for each page, in order. It must be a function at the top of a module, since a worker's
# process finds it by name.
type Reading = Callable[..., list[dict[str, Any]]]


# How long to wait for a finished worker's process to end. There is no limit on the work itself.
JOIN_TIMEOUT_SECONDS = 60

# The kinds of message a worker's process sends its parent.
_A_RECORD, _FINISHED, _FAILED = "record", "finished", "failed"


class RecordSink(Protocol):
    """Anything a stage can write its records to, one at a time: a file written as it goes, or a list."""

    def write(self, record: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class CardsRead:
    """What a stage's reading gave, and how the engines were arranged for it."""

    records: list[dict[str, Any]]
    # How many engines were started, one on each card; None for one engine across the cards. It is the
    # number started and not the number of cards: one page left to read starts one engine.
    engines: int | None

    @property
    def arrangement(self) -> str:
        if not self.engines:
            return "single-engine"
        return f"data-parallel ({self.engines} GPU{'' if self.engines == 1 else 's'})"


def read_on_the_cards(
    settings: Settings, items: list[Any], read: Reading, *, sink: RecordSink | None = None
) -> CardsRead:
    """Read a stage's pages with one engine on each card where the model allows it, else with one engine.

    The one place the arrangement is chosen. Either way the pages are read by ``read``.

    Args:
        settings: The run's settings; for a stage with its own engine settings, the copy made by
            ``for_stage``.
        items: What ``read`` reads: pages, or the records of pages, in order.
        read: The stage's reading function.
        sink: Where each record is written the moment it is made, if anywhere, so that a run that
            stops part way keeps what it had read. With an engine on each card the records arrive
            in the order the cards finish them, not the order of the items.

    Returns:
        One record for each item, in the order given, and the arrangement.
    """
    from common.pipeline_ops import check_processor, open_processor

    # Before any engine is started, and here, not in a worker's process: what would stop a processor
    # being built then ends the run with its reason, and costs no engine's start.
    check_processor(settings)
    engines = resolve_dp_gpus(settings.model_type)
    if engines:
        records = run_dp(num_gpus=engines, items=items, read=read, settings=settings, sink=sink)
        return CardsRead(records, max(1, min(engines, len(items))))
    with open_processor(settings) as processor:
        if sink is not None:
            return CardsRead(read(processor, items, settings, sink=sink), None)
        return CardsRead(read(processor, items, settings), None)


def resolve_dp_gpus(model_type: str) -> int | None:
    """How many engines to start, one on each card, or None for one engine across the cards.

    None for one of two reasons: the run has one card, or the model cannot hold a whole engine on each
    card (a quantised 31B still wants the entire card once its cache and its vision activations are
    counted, so an engine on each would run out of memory). The second is logged: an operator who has
    two cards and got one slower engine would have no way to tell from the output.
    """
    from models.registry import supports_data_parallel

    gpus = cards_to_use()
    if gpus <= 1:
        return None
    if not supports_data_parallel(model_type):
        logger.info(
            "%s cannot run one engine per GPU; using the single-engine path despite "
            "%d GPUs being available",
            model_type,
            gpus,
        )
        return None
    return gpus


def _shares(items: list[Any], n: int) -> list[list[Any]]:
    """Deal the items round-robin into at most n shares.

    Pages arrive sorted by document type, the slowest type first. Cutting the list into runs would
    put every page of the slowest type on card 0, which would still be reading them when the other
    cards had finished. Dealing round-robin (``items[i::n]``) gives every card about the same number of
    each type.

    Each card still reads its pages one type after another: a share keeps the order of the whole.
    """
    n = min(n, len(items))
    return [items[i::n] for i in range(n)]


def _in_the_order_given(by_card: dict[int, list[dict[str, Any]]], cards: int) -> list[dict[str, Any]]:
    """The cards' records put back in the order of the items.

    Card ``g`` holds the items at ``g, g + cards, g + 2 * cards, ...`` and returns its records in that
    order.
    """
    placed: list[tuple[int, dict[str, Any]]] = []
    for card in range(cards):
        for position, record in enumerate(by_card[card]):
            placed.append((position * cards + card, record))
    placed.sort(key=lambda entry: entry[0])
    return [record for _, record in placed]


def run_dp(
    *,
    num_gpus: int,
    items: list[Any],
    read: Reading,
    settings: Settings,
    sink: RecordSink | None = None,
) -> list[dict[str, Any]]:
    """Start an engine on each card, read each card's share, and give the records in the items' order.

    There is no time limit on the work: a card grinding through dense pages is never stopped for being
    slow. The loop looks every ``poll_interval`` seconds for a finished card, and fails only
    if a worker's process has died (out of memory, killed) without handing anything back.

    Args:
        num_gpus: How many engines to start, one on each card.
        items: What ``read`` reads, in order; they are dealt out to the cards.
        read: The stage's reading function. It crosses a process boundary by name, as do the items and
            the settings by value, so all three must pickle.
        settings: The run's settings.
        sink: Where each record is written as soon as its card has made it, if anywhere. Each
            worker then sends its records one at a time as well as all together at the end.

    Returns:
        One record for each item, in the order given. Each carries ``gpu_id``, its card's number, so
        that the slowest card's time can be reported: the cards work at once, and adding their times
        would give the work done, not the time taken.
    """
    poll_interval = 10  # seconds between liveness checks
    shares = _shares(items, num_gpus)
    cards = len(shares)

    logger.info("vLLM DP: distributing %d images across %d GPUs (TP=1 per GPU)", len(items), cards)

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    procs: list[Any] = []  # SpawnProcess (mp.Process subtype)
    for gpu_id, share in enumerate(shares):
        p = ctx.Process(
            target=_worker, args=(gpu_id, share, read, settings, result_queue, sink is not None)
        )
        p.start()
        procs.append(p)

    # A worker puts three kinds of message on the queue: each record as it is made (when there is a
    # sink), all its records when it finishes, and an error it hit in Python. The case left is a worker
    # killed by the system without a last message, which is found by is_alive() and the exit code, so
    # that a slow but healthy worker is never mistaken for a dead one.
    by_card: dict[int, list[dict]] = {}
    try:
        remaining = cards
        while remaining > 0:
            try:
                gpu_id, kind, payload = result_queue.get(timeout=poll_interval)
            except Empty:
                for i, p in enumerate(procs):
                    if not p.is_alive() and p.exitcode != 0 and i not in by_card:
                        msg = f"GPU {i} worker died (exit code {p.exitcode}) without returning results"
                        raise RuntimeError(msg) from None
                continue
            if kind == _A_RECORD:
                if sink is not None:
                    sink.write(payload)
                continue
            if kind == _FAILED:
                raise payload
            by_card[gpu_id] = payload
            remaining -= 1
            logger.info("GPU %d finished: %d images", gpu_id, len(payload))
    except Exception:
        for p in procs:
            if p.is_alive():
                p.terminate()
        raise
    finally:
        for p in procs:
            p.join(timeout=JOIN_TIMEOUT_SECONDS)

    for i, p in enumerate(procs):
        if p.exitcode and p.exitcode != 0:
            msg = f"GPU {i} worker exited with code {p.exitcode}"
            raise RuntimeError(msg)

    return _in_the_order_given(by_card, cards)


class _ToTheParent:
    """A worker's sink: each record goes to the parent's process as it is made, marked with its card."""

    def __init__(self, gpu_id: int, result_queue: Any) -> None:
        self._gpu_id, self._queue = gpu_id, result_queue

    def write(self, record: dict[str, Any]) -> None:
        record["gpu_id"] = self._gpu_id
        # A copy: the queue pickles it a moment later, and the record may be added to meanwhile.
        self._queue.put((self._gpu_id, _A_RECORD, dict(record)))


def _worker(
    gpu_id: int, share: list[Any], read: Reading, settings: Settings, result_queue: Any, streams: bool
) -> None:
    """A worker's process: pin its card, start the engine on it, read its share, hand the records back."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        logging.basicConfig(
            level=logging.INFO,
            format=f"%(levelname)s [GPU {gpu_id}] %(name)s: %(message)s",
        )
        from common.pipeline_ops import open_processor

        with open_processor(settings) as processor:
            if streams:
                records = read(processor, share, settings, sink=_ToTheParent(gpu_id, result_queue))
            else:
                records = read(processor, share, settings)
        for record in records:
            record["gpu_id"] = gpu_id
        result_queue.put((gpu_id, _FINISHED, records))
    except (ConfigError, StageStopped) as err:
        # A setting that cannot be used, or an engine that has stopped answering, is handed to the
        # parent as it is, so that the run ends with its reason, as it would have had the parent met it.
        result_queue.put((gpu_id, _FAILED, err))
    except Exception:
        tb = traceback.format_exc()
        result_queue.put((gpu_id, _FAILED, RuntimeError(f"GPU {gpu_id} worker failed:\n{tb}")))
