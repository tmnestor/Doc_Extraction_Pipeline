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

from common.settings import Settings

logger = logging.getLogger(__name__)

# A stage's reading function: given the document processor, its pages and the run's settings, returns
# one record for each page, in order. It must be a function at the top of a module, since a worker's
# process finds it by name.
type Reading = Callable[..., list[dict[str, Any]]]


class RecordSink(Protocol):
    """Anything a stage can write its records to, one at a time: a file written as it goes, or a list."""

    def write(self, record: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class CardsRead:
    """What a stage's reading gave, and how the engines were arranged for it."""

    records: list[dict[str, Any]]
    # How many engines were started, one on each card; None for one engine across the cards.
    engines: int | None

    @property
    def arrangement(self) -> str:
        return f"data-parallel ({self.engines} GPUs)" if self.engines else "single-engine"


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
        sink: Where each record is written, if anywhere. One engine writes each record as it is made, so
            that a run that stops part way can be carried on. With an engine on each card the records
            come back when the cards have finished, and are written then.

    Returns:
        One record for each item, in the order given, and the arrangement.
    """
    from common.pipeline_ops import open_processor

    engines = resolve_dp_gpus(settings, settings.model_type)
    if engines:
        records = run_dp(num_gpus=engines, items=items, read=read, settings=settings)
        if sink is not None:
            for record in records:
                sink.write(record)
        return CardsRead(records, engines)
    with open_processor(settings) as processor:
        if sink is not None:
            return CardsRead(read(processor, items, settings, sink=sink), None)
        return CardsRead(read(processor, items, settings), None)


def resolve_gpu_count(config: Settings) -> int:
    """Resolve effective GPU count for vLLM DP.

    Priority: data_parallel_size > num_gpus > auto-detect.
    """
    if config.data_parallel_size is not None:
        return config.data_parallel_size
    if config.num_gpus > 0:
        return config.num_gpus
    try:
        import torch

        return max(1, torch.cuda.device_count())
    except ImportError:
        return 1


def resolve_dp_gpus(config: Settings, model_type: str) -> int | None:
    """Resolve the DP rank count, or None when the DP fast path must not run.

    ``None`` means "use the single-engine path", for one of two reasons: only one
    GPU is available, or *model_type* cannot hold one whole engine per GPU (a
    quantised 31B still wants the entire card once KV cache and vision
    activations are counted, so N engines would OOM).

    The second case is logged: an operator who asked for N GPUs and silently got
    a slower single-engine run would have no way to tell from the output.

    Args:
        config: Pipeline config carrying the GPU/DP counts.
        model_type: Registered model type, used to look up the capability.

    Returns:
        Number of DP ranks to spawn, or None to skip the data-parallel path.
    """
    from models.registry import supports_data_parallel

    gpus = resolve_gpu_count(config)
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


def run_dp(*, num_gpus: int, items: list[Any], read: Reading, settings: Settings) -> list[dict[str, Any]]:
    """Start an engine on each card, read each card's share, and give the records in the items' order.

    There is no time limit on the work: a card grinding through dense pages is never stopped for being
    slow. The loop looks every ``poll_interval`` seconds for a finished card, and fails only
    if a worker's process has died (out of memory, killed) without handing anything back.

    Args:
        num_gpus: How many engines to start, one on each card.
        items: What ``read`` reads, in order; they are dealt out to the cards.
        read: The stage's reading function. It crosses a process boundary by name, as do the items and
            the settings by value, so all three must pickle.
        settings: The run's settings. How long to wait for a finished worker's process to end is
            ``resources.infrastructure.dp_join_timeout``.

    Returns:
        One record for each item, in the order given. Each carries ``gpu_id``, its card's number, so
        that the slowest card's time can be reported: the cards work at once, and adding their times
        would give the work done, not the time taken.
    """
    join_timeout = int(settings.get_infra("dp_join_timeout"))
    poll_interval = 10  # seconds between liveness checks
    shares = _shares(items, num_gpus)
    cards = len(shares)

    logger.info("vLLM DP: distributing %d images across %d GPUs (TP=1 per GPU)", len(items), cards)

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    procs: list[Any] = []  # SpawnProcess (mp.Process subtype)
    # Each worker's process sees one card only.
    one_card = settings.on_one_card()

    for gpu_id, share in enumerate(shares):
        p = ctx.Process(target=_worker, args=(gpu_id, share, read, one_card, result_queue))
        p.start()
        procs.append(p)

    # A worker that finishes puts its records on the queue; one that hits an error in Python puts the
    # error. The case left is a worker killed by the system without writing anything, which is found by
    # is_alive() and the exit code, so that a slow but healthy worker is never mistaken for a dead one.
    by_card: dict[int, list[dict]] = {}
    try:
        remaining = cards
        while remaining > 0:
            try:
                gpu_id, payload = result_queue.get(timeout=poll_interval)
            except Empty:
                for i, p in enumerate(procs):
                    if not p.is_alive() and p.exitcode != 0 and i not in by_card:
                        msg = f"GPU {i} worker died (exit code {p.exitcode}) without returning results"
                        raise RuntimeError(msg) from None
                continue
            if isinstance(payload, Exception):
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
            p.join(timeout=join_timeout)

    for i, p in enumerate(procs):
        if p.exitcode and p.exitcode != 0:
            msg = f"GPU {i} worker exited with code {p.exitcode}"
            raise RuntimeError(msg)

    return _in_the_order_given(by_card, cards)


def _worker(
    gpu_id: int, share: list[Any], read: Reading, settings: Settings, result_queue: mp.Queue
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
            records = read(processor, share, settings)
        for record in records:
            record["gpu_id"] = gpu_id
        result_queue.put((gpu_id, records))
    except Exception:
        tb = traceback.format_exc()
        result_queue.put((gpu_id, RuntimeError(f"GPU {gpu_id} worker failed:\n{tb}")))
