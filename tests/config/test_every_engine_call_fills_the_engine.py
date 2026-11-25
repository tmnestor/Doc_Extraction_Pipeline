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

from common.extraction_batching import PAGES_PER_CALL
from models.registry import get_model, list_models

SEVERAL = 3


def most_at_once() -> int:
    """The most requests any registered model's engine runs at once."""
    return max(get_model(model).max_num_seqs for model in list_models())


def test_a_call_is_several_times_what_any_engine_runs_at_once() -> None:
    """One size for every type and for classify: a size that stands apart is one someone has to explain."""
    assert PAGES_PER_CALL == 64
    assert PAGES_PER_CALL >= SEVERAL * most_at_once()
