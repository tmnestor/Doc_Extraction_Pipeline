"""Every size of engine call is several times what an engine runs at once, so the engine stays full.

A call hands the engine its pages together, and the engine runs up to ``max_num_seqs`` of them at once
and queues the rest. A call no larger than that leaves the engine with fewer to run as its short pages
finish, and the next call does not start until the slowest page of this one is done.

Receipts and invoices went 10 to a call from 14 August 2026, the largest size tried that day, when the
quantised 12B ran 10 at once and that number was taken for a ceiling on the size of a call. It is not:
on 16 November, 55 statements with 8 in flight took 484 s at 16 to a call and 418 s as one call of 55,
and statements and classify were given 64. Receipts and invoices were not looked at again, and by
25 November a call of 10 left 6 of an engine's 16 places empty.
"""

from typing import Any

import yaml

from common.settings import DEFAULT_CONFIG

SEVERAL = 3


def shipped() -> dict[str, Any]:
    return yaml.safe_load(DEFAULT_CONFIG.read_text())


def most_at_once(raw: dict[str, Any]) -> int:
    """The most requests any engine of the shipped file runs at once."""
    engines = raw["inference"]["vllm"]
    return max(
        [engines["defaults"]["max_num_seqs"]]
        + [own["max_num_seqs"] for own in engines["models"].values() if "max_num_seqs" in own]
    )


def test_every_size_of_call_is_several_times_what_an_engine_runs_at_once() -> None:
    raw = shipped()
    least = SEVERAL * most_at_once(raw)

    too_small = {name: size for name, size in raw["pipeline"]["batching"].items() if size < least}

    assert too_small == {}, f"each must be at least {least}"


def test_no_type_is_left_with_a_size_of_its_own_that_differs_from_the_rest() -> None:
    """One rule for every type: a size that stands apart is a size someone has to explain."""
    sizes = shipped()["pipeline"]["batching"]

    assert len(set(sizes.values())) == 1, sizes
