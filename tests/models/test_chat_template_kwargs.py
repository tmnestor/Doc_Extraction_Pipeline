"""enable_thinking=False must reach engine.chat(), or suppression is a no-op.

Qwen 3.8's chat template treats an undefined ``enable_thinking`` as true and
defaults ``reasoning_effort`` to ``xhigh``, then injects a system message telling
the model to reason at length before opening a ``<think>`` block. Unsuppressed,
extraction spends its completion budget reasoning and STILL returns a
parseable-looking response. The failure is silent, which is why it is guarded
here rather than left to be noticed in a score.

``chat_template_kwargs`` was a declared-but-never-read field until the Gemma work
wired it through. ``VllmBackend`` takes its engine as a plain argument, so this
runs without a GPU.
"""

from models.backends.vllm_backend import VllmBackend
from models.model_loader import _VLLM_SPECS

# Importing the registry is what RUNS the register_vllm_model calls that populate
# _VLLM_SPECS. Without this the map is empty when this file is run on its own, and
# only fills by accident when another test module happens to import the registry
# first, so the import is load-bearing, not tidiness.
from models.registry import list_models  # noqa: E402

QWEN = "qwen38-27b-fp8-vllm"
GEMMA = "gemma4-31b-w4a16-vllm"


def test_qwen_backend_forwards_thinking_suppression():
    backend = VllmBackend(
        engine=None,
        model_type_key=QWEN,
        chat_template_kwargs={"enable_thinking": False},
    )
    assert backend._chat_kwargs() == {"chat_template_kwargs": {"enable_thinking": False}}


def test_a_model_without_kwargs_sends_none():
    # InternVL drives thinking from the prompt, not a flag, so its call site must
    # stay byte-identical to the pre-Gemma behaviour.
    backend = VllmBackend(engine=None, model_type_key="internvl3-vllm")
    assert backend._chat_kwargs() == {}


def test_both_comparison_specs_actually_set_the_flag():
    # Guards the gap between "the backend CAN forward it" and "the spec DOES set
    # it". Both models in the comparison must suppress, or one of them spends its
    # budget reasoning and the throughput half of the result is meaningless.
    for model_type in (QWEN, GEMMA):
        spec = _VLLM_SPECS[model_type]
        assert spec.chat_template_kwargs == {"enable_thinking": False}


def test_every_registered_spec_is_recorded():
    # The map must not drift from the registry, or a future model could be
    # registered without its capabilities being inspectable.
    assert set(_VLLM_SPECS) == set(list_models())
