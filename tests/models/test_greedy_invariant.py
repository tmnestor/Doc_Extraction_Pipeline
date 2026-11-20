"""Greedy decoding is a hardcoded invariant of the vLLM path, not a config value.

``VllmBackend`` builds ``SamplingParams(temperature=0)`` at the call site, and vLLM honours per-request
params over whatever a checkpoint's own ``generation_config.json`` asks for. So a model whose checkpoint
wants sampling still decodes greedily, with no override needed.

The claim that a difference between two runs is real signal rests on this, so it gets a guard: if that
hardcoded zero ever became configurable, a model could start sampling without anything saying so.
"""

import inspect

import pytest
import yaml

from common.app_config import _build_generation_registry
from models.backends import vllm_backend

MODELS_NORMALISED = ("gemma4-31b", "gemma4-12b-unified")


@pytest.fixture(scope="module")
def generation_registry(run_config_path) -> dict[str, dict]:
    """The generation registry with `defaults` merged into each model entry."""
    return _build_generation_registry(yaml.safe_load(run_config_path.read_text()))


def test_the_vllm_extraction_path_hardcodes_temperature_zero():
    source = inspect.getsource(vllm_backend.VllmBackend)
    assert "temperature=0" in source
    # No sampler may be threaded in from config on the extraction path.
    assert "do_sample" not in source


@pytest.mark.parametrize("model", MODELS_NORMALISED)
def test_generation_settings_are_greedy_for_both_models(generation_registry, model):
    assert generation_registry[model]["temperature"] == 0.0
    assert generation_registry[model]["do_sample"] is False


def test_both_models_have_the_same_token_budget(generation_registry):
    # The same completion budget for both, or a comparison of the two is confounded: a model given fewer
    # tokens can be cut off mid-answer on the dense tables of a bank statement.
    for key, value in (("max_new_tokens_base", 512), ("max_new_tokens_per_field", 64)):
        assert {generation_registry[model][key] for model in MODELS_NORMALISED} == {value}
