"""enable_thinking=False must reach engine.chat(), or suppression is a no-op.

Gemma's chat template reasons by default. Unsuppressed, extraction spends its completion budget reasoning
and STILL returns a parseable-looking response. The failure is silent, which is why it is guarded here
rather than left to be noticed in a score.

``VllmBackend`` takes its engine as a plain argument, so this runs without a GPU.
"""

import pytest

from models.backends.vllm_backend import VllmBackend
from models.model_loader import _VLLM_SPECS

# Importing the registry is what RUNS the register_vllm_model calls that populate
# _VLLM_SPECS. Without this the map is empty when this file is run on its own, and
# only fills by accident when another test module happens to import the registry
# first, so the import is load-bearing, not tidiness.
from models.registry import list_models  # noqa: E402

MODELS = ("gemma4-31b-w4a16-vllm", "gemma4-12b-unified-w4a16-vllm")


@pytest.mark.parametrize("model", MODELS)
def test_the_backend_forwards_thinking_suppression(model):
    backend = VllmBackend(
        engine=None,
        model_type_key=model,
        chat_template_kwargs={"enable_thinking": False},
    )
    assert backend._chat_kwargs() == {"chat_template_kwargs": {"enable_thinking": False}}


def test_a_backend_given_no_kwargs_sends_none():
    backend = VllmBackend(engine=None, model_type_key=MODELS[0])
    assert backend._chat_kwargs() == {}


@pytest.mark.parametrize("model", MODELS)
def test_each_model_actually_sets_the_flag(model):
    # Guards the gap between "the backend CAN forward it" and "the spec DOES set it".
    assert _VLLM_SPECS[model].chat_template_kwargs == {"enable_thinking": False}


def test_every_registered_spec_is_recorded():
    # The map must not drift from the registry, or a future model could be
    # registered without its capabilities being inspectable.
    assert set(_VLLM_SPECS) == set(list_models())
