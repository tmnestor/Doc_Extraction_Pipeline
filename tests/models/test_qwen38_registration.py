"""Registration and capability tests for the Qwen 3.8 27B FP8 vLLM spec."""

import pytest

from models.registry import get_model, is_vllm_model, list_models, supports_data_parallel

QWEN = "qwen38-27b-fp8-vllm"


def test_qwen38_is_registered():
    assert QWEN in list_models()
    assert get_model(QWEN).model_type == QWEN


def test_qwen38_is_a_vllm_model():
    assert is_vllm_model(QWEN) is True


def test_qwen38_shares_the_common_prompt_file():
    # No model here has its own prompt file; one shared prompt definition is
    # what makes the comparison a comparison.
    assert get_model(QWEN).prompt_file == "internvl3_prompts.yaml"


def test_qwen38_refuses_data_parallel():
    # 25.87 GiB cannot replicate onto a 24 GiB L4; run_dp would start two whole
    # engines and OOM at load.
    assert supports_data_parallel(QWEN) is False


def test_unknown_model_type_fails_with_a_diagnostic():
    with pytest.raises(ValueError) as excinfo:
        get_model("qwen38-27b-vllm")  # the -fp8 suffix omitted: a plausible typo
    assert QWEN in str(excinfo.value)
