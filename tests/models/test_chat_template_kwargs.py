"""The chat-template arguments a model is registered with must reach ``LLM.chat``.

For Gemma 4 the argument is ``enable_thinking: False``. The model's own chat template treats a request
without the argument as thinking off too (rendered both ways on 21 November 2026 and found the same), so
the argument states what the default already does. It is still passed and still guarded: a template that
thought by default would spend its completion budget reasoning and return a reply that looks parseable,
and nothing in a score would say why.

``VllmBackend`` takes its engine as a plain argument, so this runs without a GPU.
"""

import sys
import types

import pytest

from models.backends.vllm_backend import VllmBackend
from models.model_loader import _VLLM_SPECS

# Importing the registry is what RUNS the register_vllm_model calls that populate
# _VLLM_SPECS. Without this the map is empty when this file is run on its own, and
# only fills by accident when another test module happens to import the registry
# first, so the import is load-bearing, not tidiness.
from models.registry import list_models  # noqa: E402

MODELS = ("gemma4-31b-w4a16-vllm", "gemma4-12b-unified-w4a16-vllm")


class KeepsCalls:
    """Stands in for the engine: keeps the arguments of each chat call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def chat(self, **arguments):
        self.calls.append(arguments)
        return [
            types.SimpleNamespace(
                outputs=[types.SimpleNamespace(text="reply", token_ids=[1])],
                prompt_token_ids=[1],
                num_cached_tokens=0,
            )
            for _ in arguments["messages"]
        ]


@pytest.fixture
def blank(monkeypatch):
    from PIL import Image

    monkeypatch.setitem(
        sys.modules, "vllm", types.SimpleNamespace(SamplingParams=lambda **settings: settings)
    )
    return Image.new("RGB", (8, 8), "white")


@pytest.mark.parametrize("model", MODELS)
def test_the_backend_forwards_thinking_suppression(model, blank):
    engine = KeepsCalls()
    backend = VllmBackend(
        engine=engine, model_type_key=model, chat_template_kwargs={"enable_thinking": False}
    )

    backend.chat([blank], ["read this"], 50)

    assert engine.calls[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_a_backend_given_no_kwargs_sends_none(blank):
    engine = KeepsCalls()

    VllmBackend(engine=engine, model_type_key=MODELS[0]).chat([blank], ["read this"], 50)

    assert "chat_template_kwargs" not in engine.calls[0]


@pytest.mark.parametrize("model", MODELS)
def test_each_model_actually_sets_the_flag(model):
    # Guards the gap between "the backend CAN forward it" and "the spec DOES set it".
    assert _VLLM_SPECS[model].chat_template_kwargs == {"enable_thinking": False}


def test_every_registered_spec_is_recorded():
    # The map must not drift from the registry, or a future model could be
    # registered without its capabilities being inspectable.
    assert set(_VLLM_SPECS) == set(list_models())
