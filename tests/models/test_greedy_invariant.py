"""Greedy decoding is a hardcoded invariant of the vLLM path, not a config value.

``VllmBackend`` builds ``SamplingParams(temperature=0)`` at the call site, and vLLM honours per-request
params over whatever a checkpoint's own ``generation_config.json`` asks for. So a model whose checkpoint
wants sampling still decodes greedily, with no override needed.

The claim that a difference between two runs is real signal rests on this, so it gets a guard: if that
hardcoded zero ever became configurable, a model could start sampling without anything saying so.
"""

import inspect

from common.config_schema import KNOWN_KEYS
from models.backends import vllm_backend


def test_the_vllm_extraction_path_hardcodes_temperature_zero():
    source = inspect.getsource(vllm_backend.VllmBackend)
    assert "temperature=0" in source
    # No sampler may be threaded in from config on the extraction path.
    assert "do_sample" not in source


def test_no_setting_offers_a_sampler():
    """Until 24 November 2026 the settings file held temperature, do_sample and top_p, which nothing read.

    They read as if sampling could be switched on there. No key may name a sampler: the zero is fixed
    where the request is made.
    """
    sampler_words = ("temperature", "do_sample", "top_p", "top_k", "generation")
    assert [key for key in KNOWN_KEYS if any(word in key for word in sampler_words)] == []


def test_both_models_have_the_same_token_budget():
    # The same completion budget for both, or a comparison of the two is confounded: a model given fewer
    # tokens can be cut off mid-answer on the dense tables of a bank statement. The budgets are constants
    # beside the requests, keyed by what is asked and never by model.
    from models import orchestrator

    assert (
        orchestrator.CLASSIFY_TOKENS,
        orchestrator.EXTRACT_BASE_TOKENS,
        orchestrator.EXTRACT_TOKENS_PER_FIELD,
    ) == (150, 512, 64)
    assert "model" not in "".join(name.lower() for name in dir(orchestrator) if name.endswith("_TOKENS"))
