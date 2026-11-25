"""Several requests to an engine call, and one at a time when a call fails.

Until 24 November 2026 this was written out in each stage that uses the model: three times with small
differences, and in the split stage not at all, where one photograph the engine refused ended the stage.

The items are invented words; nothing here is a page.
"""

from typing import Any

import pytest

from common.engine_calls import EngineWatch, Outcome, StageStopped, flat, groups_of, in_engine_calls


class Engine:
    """Stands in for one engine call: answers in capitals, and notes the size of every call."""

    def __init__(self, *, refuses_shared: bool = False, refuses: tuple[str, ...] = (), short: bool = False):
        self.calls: list[list[str]] = []
        self._refuses_shared, self._refuses, self._short = refuses_shared, refuses, short

    def __call__(self, group: list[str]) -> list[str]:
        self.calls.append(list(group))
        if self._refuses_shared and len(group) > 1:
            raise RuntimeError("the call was too large")
        if any(item in self._refuses for item in group):
            raise ValueError(f"cannot read {group[0]}")
        replies = [item.upper() for item in group]
        return replies[:-1] if self._short and len(group) > 1 else replies


def in_calls(groups: Any, ask: Any, **more: Any) -> Any:
    """The engine calls of these groups, with a watch that never stops them."""
    return in_engine_calls(groups, ask, watch=EngineWatch(10_000), **more)


WORDS = ["quill", "feather", "ink", "blotter", "nib"]


def test_items_are_grouped_in_order_and_the_last_group_holds_what_is_left() -> None:
    assert groups_of(WORDS, 2) == [["quill", "feather"], ["ink", "blotter"], ["nib"]]
    assert groups_of(WORDS, 5) == [WORDS]
    assert groups_of(WORDS, 64) == [WORDS]
    assert groups_of(WORDS, 1) == [[word] for word in WORDS]
    assert groups_of([], 3) == []


def test_each_group_is_one_call_and_every_item_has_its_reply_in_order() -> None:
    engine = Engine()

    outcomes = flat(in_calls(groups_of(WORDS, 2), engine))

    assert engine.calls == [["quill", "feather"], ["ink", "blotter"], ["nib"]]
    assert [(outcome.item, outcome.reply, outcome.error) for outcome in outcomes] == [
        (word, word.upper(), None) for word in WORDS
    ]
    assert all(outcome.seconds >= 0.0 for outcome in outcomes)


def test_a_groups_outcomes_come_as_soon_as_its_call_is_done() -> None:
    """A stage writes each group's records before the next call starts, so a run can be carried on."""
    engine = Engine()

    calls = in_calls(groups_of(WORDS, 2), engine)

    assert [outcome.item for outcome in next(calls)] == ["quill", "feather"]
    assert engine.calls == [["quill", "feather"]]


def test_a_call_that_fails_is_tried_again_one_item_at_a_time() -> None:
    engine = Engine(refuses_shared=True)

    outcomes = flat(in_calls([WORDS[:3]], engine, what="words"))

    assert engine.calls == [["quill", "feather", "ink"], ["quill"], ["feather"], ["ink"]]
    assert [outcome.reply for outcome in outcomes] == ["QUILL", "FEATHER", "INK"]
    assert all(outcome.error is None for outcome in outcomes)


def test_an_item_that_fails_on_its_own_is_given_back_with_its_error_and_the_rest_are_read() -> None:
    engine = Engine(refuses=("feather",))

    outcomes = flat(in_calls(groups_of(WORDS, 3), engine))

    assert [outcome.reply for outcome in outcomes] == ["QUILL", None, "INK", "BLOTTER", "NIB"]
    assert [outcome.error for outcome in outcomes] == [None, "cannot read feather", None, None, None]
    # The second group was not touched by the first group's failure.
    assert engine.calls[-1] == ["blotter", "nib"]


def test_a_group_of_one_that_fails_is_not_asked_twice() -> None:
    engine = Engine(refuses=("ink",))

    (outcome,) = flat(in_calls([["ink"]], engine))

    assert engine.calls == [["ink"]]
    assert outcome == Outcome("ink", None, "cannot read ink", outcome.seconds)


def test_a_call_with_a_reply_missing_is_treated_as_failed() -> None:
    """Replies are matched to requests by position: a short list would give one page another's answer."""
    engine = Engine(short=True)

    outcomes = flat(in_calls([WORDS[:2]], engine))

    assert engine.calls == [["quill", "feather"], ["quill"], ["feather"]]
    assert [outcome.reply for outcome in outcomes] == ["QUILL", "FEATHER"]


def test_nothing_is_raised_however_the_engine_fails() -> None:
    def always_refuses(group: list[str]) -> list[str]:
        raise MemoryError("the card is full")

    outcomes = flat(in_calls(groups_of(WORDS, 2), always_refuses))

    assert [outcome.error for outcome in outcomes] == ["the card is full"] * 5


def test_an_empty_group_is_passed_over() -> None:
    engine = Engine()

    (outcome,) = flat(in_calls([[], ["nib"], []], engine))

    assert (outcome.item, outcome.reply, outcome.error) == ("nib", "NIB", None)
    assert engine.calls == [["nib"]]


def test_the_time_of_a_failed_call_is_charged_to_the_items_tried_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sum of the items' times is the time the engine was busy, failed call and all."""
    from common import engine_calls

    clock = iter(float(tick) for tick in range(100))
    monkeypatch.setattr(engine_calls.time, "time", lambda: next(clock))

    shared = flat(in_calls([["quill", "feather"]], Engine()))
    retried = flat(in_calls([["quill", "feather"]], Engine(refuses_shared=True)))

    # One tick for the shared call, halved between the two.
    assert [outcome.seconds for outcome in shared] == [0.5, 0.5]
    # One tick for the failed call, halved, and one tick for each item's own call.
    assert [outcome.seconds for outcome in retried] == [1.5, 1.5]


def test_a_setting_that_cannot_be_used_is_raised_and_the_requests_are_not_tried_again() -> None:
    """It would fail each request the same way, and record a run of failed pages for one mistake."""
    from common.errors import ConfigError

    calls: list[list[str]] = []

    def ask(group: list[str]) -> list[str]:
        calls.append(list(group))
        raise ConfigError("What: a rule cannot be used.")

    with pytest.raises(ConfigError, match="a rule cannot be used"):
        flat(in_calls(groups_of(WORDS, 3), ask))

    assert calls == [["quill", "feather", "ink"]]


def test_a_setting_that_cannot_be_used_is_raised_from_a_request_tried_on_its_own() -> None:
    from common.errors import ConfigError

    calls: list[list[str]] = []

    def ask(group: list[str]) -> list[str]:
        calls.append(list(group))
        if len(group) > 1:
            raise RuntimeError("the call was too large")
        raise ConfigError("What: a rule cannot be used.")

    with pytest.raises(ConfigError):
        flat(in_calls(groups_of(WORDS, 3), ask))

    assert calls == [["quill", "feather", "ink"], ["quill"]]


# --- requests that fail one after another ---------------------------------------------------------------


def test_a_stage_is_stopped_when_as_many_requests_fail_in_a_row_as_are_allowed() -> None:
    engine = Engine(refuses=tuple(WORDS))
    calls = in_engine_calls(groups_of(WORDS, 1), engine, watch=EngineWatch(3))

    assert [outcome.error for outcome in next(calls)] == ["cannot read quill"]
    assert [outcome.error for outcome in next(calls)] == ["cannot read feather"]
    with pytest.raises(StageStopped) as stopped:
        next(calls)

    message = str(stopped.value)
    assert all(part in message for part in ("What:", "Where:", "Expected:", "How to fix:"))
    assert "3 requests in a row failed" in message and "cannot read ink" in message
    assert "STOP_AFTER_FAILED_REQUESTS" in message
    assert engine.calls == [["quill"], ["feather"], ["ink"]]


def test_the_requests_of_a_failed_shared_call_are_counted_one_by_one() -> None:
    engine = Engine(refuses=tuple(WORDS))

    with pytest.raises(StageStopped, match="3 requests in a row failed"):
        flat(in_engine_calls([WORDS], engine, watch=EngineWatch(3)))

    # The shared call, and then three of its five requests on their own.
    assert engine.calls == [WORDS, ["quill"], ["feather"], ["ink"]]


def test_a_request_that_is_answered_starts_the_count_again() -> None:
    engine = Engine(refuses=("quill", "feather", "blotter", "nib"))

    outcomes = flat(in_engine_calls(groups_of(WORDS, 1), engine, watch=EngineWatch(3)))

    assert [outcome.error is None for outcome in outcomes] == [False, False, True, False, False]


def test_a_shared_call_that_is_answered_starts_the_count_again() -> None:
    engine = Engine(refuses=("quill", "nib"))

    outcomes = flat(in_engine_calls([["quill"], ["feather", "ink"], ["nib"]], engine, watch=EngineWatch(2)))

    assert [outcome.error is None for outcome in outcomes] == [False, True, True, False]
