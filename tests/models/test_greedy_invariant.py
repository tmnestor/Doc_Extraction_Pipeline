"""Greedy decoding is a hardcoded invariant of the vLLM path, not a config value.

Qwen 3.8's checkpoint asks for sampling: ``generation_config.json`` ships
``do_sample: true``, ``temperature: 1.0``, ``top_k: 20``, ``top_p: 0.95``. That
never reaches the sampler, because ``VllmBackend`` builds
``SamplingParams(temperature=0)`` at the call site and vLLM honours per-request
params over the checkpoint file. So no override is needed and none is added.

Qwen is nevertheless the FIRST model here whose checkpoint actively wants
sampling, which makes it the model that would silently start sampling if that
hardcoded zero ever became configurable. The benchmark's claim that any
run-to-run delta is real signal rests on this, so it gets a guard.
"""

import inspect

import pytest
import yaml

from common.app_config import _build_generation_registry
from models.backends import vllm_backend

QWEN_NORMALISED = "qwen38-27b-fp8"
GEMMA_NORMALISED = "gemma4-31b"


@pytest.fixture(scope="module")
def generation_registry(run_config_path) -> dict[str, dict]:
    """The generation registry with `defaults` merged into each model entry."""
    return _build_generation_registry(yaml.safe_load(run_config_path.read_text()))


def test_the_vllm_extraction_path_hardcodes_temperature_zero():
    source = inspect.getsource(vllm_backend.VllmBackend)
    assert "temperature=0" in source
    # No sampler may be threaded in from config on the extraction path: if this
    # ever becomes configurable, Qwen is what starts sampling.
    assert "do_sample" not in source


def test_generation_defaults_are_greedy(generation_registry):
    # Every entry inherits `generation.defaults`, so checking one proves the
    # default rather than a per-model override.
    assert generation_registry[GEMMA_NORMALISED]["temperature"] == 0.0
    assert generation_registry[GEMMA_NORMALISED]["do_sample"] is False


def test_qwen38_inherits_greedy_without_a_per_model_override(generation_registry):
    qwen = generation_registry[QWEN_NORMALISED]
    assert qwen["temperature"] == 0.0
    assert qwen["do_sample"] is False


def test_qwen38_token_budget_matches_gemma31b(generation_registry):
    qwen = generation_registry[QWEN_NORMALISED]
    gemma = generation_registry[GEMMA_NORMALISED]
    # Same completion budget both sides, or the comparison is confounded: a
    # model given fewer tokens can be cut off mid-answer on the dense tables
    # that are the only discriminating document type.
    assert qwen["max_new_tokens_base"] == gemma["max_new_tokens_base"] == 512
    assert qwen["max_new_tokens_per_field"] == gemma["max_new_tokens_per_field"] == 64
