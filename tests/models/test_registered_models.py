"""The pipeline runs two models, and refuses any other name before doing any work.

Gemma 4 31B runs as one engine across both cards. Gemma 4 12B, quantised by its maker, runs as one engine
on each card. A model that was once registered and is no longer is a plain mistake in the settings, and
the message must say which names are allowed.
"""

from pathlib import Path
from typing import Any

import pytest

from models.registry import IMAGE_TOKENS, get_model, list_models

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_CONFIG = REPO_ROOT / "config" / "run_config.yml"
MODELS = ["gemma4-12b-unified-w4a16-vllm", "gemma4-31b-w4a16-vllm"]


def test_exactly_two_models_are_registered() -> None:
    assert list_models() == MODELS


@pytest.mark.parametrize("removed", ["internvl3-vllm", "qwen38-27b-fp8-vllm", "gemma4-12b-unified-vllm"])
def test_a_removed_model_is_refused_with_a_full_diagnostic(removed: str) -> None:
    with pytest.raises(ValueError) as raised:
        get_model(removed)

    message = str(raised.value)
    assert removed in message
    assert all(model in message for model in MODELS)
    assert all(part in message for part in ("What:", "Where:", "Expected:", "How to fix:"))
    assert "internvl" not in message.replace(removed, "")


# --- what a model's engine is started with is in its registration ---------------------------------------
# Until 26 November 2026 these were a block for each model in the settings file, with defaults under
# them, overrides for one stage over them, and the number of image tokens written three times a model.


@pytest.mark.parametrize("model", MODELS)
def test_a_models_engine_numbers_are_in_its_registration(model: str) -> None:
    spec = get_model(model)

    # The values every run of either model has used. The settings file shipped the 12B with 16384 and
    # 10, and every run sheet changed them to these.
    assert (spec.max_model_len, spec.max_num_seqs, spec.gpu_memory_utilization) == (8192, 16, 0.90)
    assert spec.chat_template is None


def numbers_in(block: Any) -> list[int]:
    if isinstance(block, dict):
        return [number for value in block.values() for number in numbers_in(value)]
    return [block]


@pytest.mark.parametrize("model", MODELS)
def test_a_models_image_tokens_are_one_number_wherever_it_is_named(model: str) -> None:
    """The processor and the model's own configuration each name it, and they must agree."""
    spec = get_model(model)

    assert IMAGE_TOKENS == 560
    assert numbers_in(spec.mm_processor_kwargs) == [IMAGE_TOKENS]
    assert numbers_in(spec.hf_overrides) and set(numbers_in(spec.hf_overrides)) == {IMAGE_TOKENS}


def test_each_model_names_its_image_tokens_as_its_own_configuration_does() -> None:
    assert get_model("gemma4-31b-w4a16-vllm").hf_overrides == {
        "vision_soft_tokens_per_image": 560,
        "vision_config": {"default_output_length": 560},
    }
    assert get_model("gemma4-12b-unified-w4a16-vllm").hf_overrides == {
        "vision_config": {"num_soft_tokens": 560}
    }
