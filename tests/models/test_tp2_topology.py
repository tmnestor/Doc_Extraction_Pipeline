"""The 31B must refuse the data-parallel path on 2 GPUs, and the quantised 12B must take it.

Gemma 4 31B W4A16 is 21.67 GiB; an L4 has 24 GiB, about 22.5 usable. It cannot hold one whole engine on
a card, so it must run as a single engine split across both (tp=2). If ``resolve_dp_gpus`` ever returns
a count for it, ``run_dp`` starts two whole engines and runs out of memory at load.
"""

import pytest

from common.pipeline_config import PipelineConfig
from common.vllm_dp import resolve_dp_gpus
from models.registry import get_model

GEMMA = "gemma4-31b-w4a16-vllm"


@pytest.fixture
def two_gpu_config(tmp_path) -> PipelineConfig:
    """A config reporting two GPUs. Paths are required but unused by the gate."""
    return PipelineConfig(
        data_dir=tmp_path,
        output_dir=tmp_path,
        num_gpus=2,
        data_parallel_size=None,
    )


def test_two_gpus_still_take_the_single_engine_path(two_gpu_config):
    assert resolve_dp_gpus(two_gpu_config, GEMMA) is None


def test_the_12b_still_does_take_the_data_parallel_path(two_gpu_config):
    # Control: the capability is per model, not a blanket disable. The QAT 12B
    # is 9.56 GiB and genuinely does fit one engine per card.
    assert resolve_dp_gpus(two_gpu_config, "gemma4-12b-unified-w4a16-vllm") == 2


def test_gemma31b_description_states_the_real_size_and_topology():
    # The committed string said "~18 GB, 1xL40S tp=1". All three were wrong: the
    # checkpoint is 23,265,352,448 B = 21.67 GiB and the hardware is 2xL4.
    description = get_model(GEMMA).description
    assert "21.67" in description
    assert "2xL4" in description
    assert "tp=2" in description
    assert "L40S" not in description
