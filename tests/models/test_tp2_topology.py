"""The 31B must refuse the data-parallel path on 2 GPUs, and the quantised 12B must take it.

Gemma 4 31B W4A16 is 21.67 GiB; an L4 has 24 GiB, about 22.5 usable. It cannot hold one whole engine on
a card, so it must run as a single engine split across both (tp=2). If ``resolve_dp_gpus`` ever returns
a count for it, ``run_dp`` starts two whole engines and runs out of memory at load.
"""

import pytest

from common.vllm_dp import resolve_dp_gpus
from models.registry import get_model

GEMMA = "gemma4-31b-w4a16-vllm"


@pytest.fixture
def two_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    """A process that sees two cards. Nothing else is read by the choice of arrangement."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")


def test_two_gpus_still_take_the_single_engine_path(two_cards):
    assert resolve_dp_gpus(GEMMA) is None


def test_the_12b_still_does_take_the_data_parallel_path(two_cards):
    # Control: the capability is per model, not a blanket disable. The QAT 12B
    # is 9.56 GiB and genuinely does fit one engine per card.
    assert resolve_dp_gpus("gemma4-12b-unified-w4a16-vllm") == 2


def test_gemma31b_description_states_the_real_size_and_topology():
    # The committed string said "~18 GB, 1xL40S tp=1". All three were wrong: the
    # checkpoint is 23,265,352,448 B = 21.67 GiB and the hardware is 2xL4.
    description = get_model(GEMMA).description
    assert "21.67" in description
    assert "2xL4" in description
    assert "tp=2" in description
    assert "L40S" not in description
