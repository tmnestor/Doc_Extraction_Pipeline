"""Engine calls: several requests to a call, and one at a time when a call fails.

Every stage that uses the model sends its requests several to an engine call, so that the engine can run
them at once. Every stage also wants the same thing when a call fails: each request of it tried again on
its own, so that one bad page costs its group only the shared call, and a page that still fails recorded
with its error, so that it never ends the run. That is written here once.

What is asked, how the requests are grouped and what a stage records are the stage's business. This
module is given the groups and a function that makes one call.
"""

import logging
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Outcome[Item, Reply]:
    """What came of one request."""

    item: Item
    # The reply, or None if the request failed in a shared call and again on its own.
    reply: Reply | None
    # Why it failed, or None.
    error: str | None
    # The request's share of the time its call took. Inside a shared call the requests overlap, so one
    # request's own time is not separable: each is charged an equal share. A request tried again on its
    # own is charged that call's time and its share of the call that failed.
    seconds: float


def groups_of[Item](items: Sequence[Item], size: int) -> list[list[Item]]:
    """The items in order, ``size`` to a group; the last group holds what is left."""
    return [list(items[start : start + size]) for start in range(0, len(items), size)]


def _asked[Item, Reply](ask: Callable[[list[Item]], Sequence[Reply]], group: list[Item]) -> Sequence[Reply]:
    """One call's replies, refused unless there is one for each request.

    Replies are matched to requests by position, so a list that is short would give one page another's
    answer.
    """
    replies = ask(group)
    if len(replies) != len(group):
        raise ValueError(
            f"the engine returned {len(replies)} replies for {len(group)} requests; matching by "
            f"position is unsafe"
        )
    return replies


def in_engine_calls[Item, Reply](
    groups: Iterable[list[Item]],
    ask: Callable[[list[Item]], Sequence[Reply]],
    *,
    what: str = "requests",
) -> Iterator[list[Outcome[Item, Reply]]]:
    """Make one engine call for each group, and give each group's outcomes as soon as it is done.

    A call that fails is tried again one request at a time. A request that still fails is given back
    with its error and no reply. Nothing is raised for a failed call.

    Args:
        groups: The requests, grouped as they are to be sent. An empty group is passed over.
        ask: Makes one call: given a group, returns a reply for each of its requests, in order.
        what: What the requests are, for the log ("pages", "receipts").

    Yields:
        For each group, in order, one outcome for each of its requests, in order.
    """
    for group in groups:
        if not group:
            continue
        started = time.time()
        try:
            replies = _asked(ask, group)
        except Exception as err:  # noqa: BLE001 - one call, or one request, must not end the run
            failed_for = time.time() - started
            if len(group) == 1:
                logger.error("A request failed (%s: %s)", type(err).__name__, err)
                yield [Outcome(group[0], None, str(err), failed_for)]
                continue
            logger.warning(
                "A call of %d %s failed (%s: %s); retrying one at a time",
                len(group),
                what,
                type(err).__name__,
                err,
            )
            yield [_alone(ask, item, failed_for / len(group)) for item in group]
            continue
        share = (time.time() - started) / len(group)
        yield [Outcome(item, reply, None, share) for item, reply in zip(group, replies, strict=True)]


def _alone[Item, Reply](
    ask: Callable[[list[Item]], Sequence[Reply]], item: Item, already: float
) -> Outcome[Item, Reply]:
    """One request on its own, after the call it shared failed."""
    started = time.time()
    try:
        (reply,) = _asked(ask, [item])
    except Exception as err:  # noqa: BLE001 - recorded on the request, not raised
        logger.error("A request failed on its own (%s: %s)", type(err).__name__, err)
        return Outcome(item, None, str(err), already + time.time() - started)
    return Outcome(item, reply, None, already + time.time() - started)


def flat[Item, Reply](calls: Iterable[list[Outcome[Item, Reply]]]) -> list[Outcome[Item, Reply]]:
    """Every outcome of every call, in order."""
    return [outcome for outcomes in calls for outcome in outcomes]
