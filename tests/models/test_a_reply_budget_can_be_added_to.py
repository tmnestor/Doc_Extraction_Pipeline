"""A model that deliberates before it answers is given room for both, and the others are sent what they were.

On 22 September 2026 the Qwen 3.8 27B, with thinking off, still deliberated in plain text on 107 of 165
pages. On four receipts the deliberation used up the reply's budget before the answer was written, and
ten or more of their fourteen fields came back not found. A model's registration can now add tokens to
the budget of every reply. For the two models the pipeline runs it adds none, so what they are sent is
unchanged.

The engine is a stand-in: no model is loaded.
"""

import dataclasses
import sys
import types
from typing import Any

import pytest
from PIL import Image

from models.backends.vllm_backend import VllmBackend
from models.registry import get_model, list_models

THE_MODELS_THE_PIPELINE_RUNS = ("gemma4-31b-w4a16-vllm", "gemma4-12b-unified-w4a16-vllm")
THE_EXPERIMENT = "qwen38-27b-fp8-vllm"


class KeepsCalls:
    """Stands in for the engine: keeps the arguments of each chat call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat(self, **arguments: Any) -> list[Any]:
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
def blank(monkeypatch: pytest.MonkeyPatch) -> Image.Image:
    monkeypatch.setitem(
        sys.modules, "vllm", types.SimpleNamespace(SamplingParams=lambda **settings: settings)
    )
    return Image.new("RGB", (8, 8), "white")


def test_the_tokens_a_registration_adds_are_added_to_every_reply_s_budget(blank: Image.Image) -> None:
    engine = KeepsCalls()

    VllmBackend(engine=engine, model_type_key=THE_EXPERIMENT, reply_tokens_added=2048).chat(
        [blank, blank], ["which type", "read this"], 150
    )

    assert engine.calls[0]["sampling_params"] == {"max_tokens": 150 + 2048, "temperature": 0}


def test_a_backend_told_of_none_asks_for_the_budget_it_was_given(blank: Image.Image) -> None:
    engine = KeepsCalls()

    VllmBackend(engine=engine, model_type_key=THE_MODELS_THE_PIPELINE_RUNS[0]).chat(
        [blank], ["read this"], 150
    )

    assert engine.calls[0]["sampling_params"] == {"max_tokens": 150, "temperature": 0}


@pytest.mark.parametrize("model", THE_MODELS_THE_PIPELINE_RUNS)
def test_the_models_the_pipeline_runs_add_nothing(model: str) -> None:
    assert get_model(model).reply_tokens_added == 0


def test_the_experiment_is_registered_as_it_was_run_in_september_with_room_to_deliberate() -> None:
    spec = get_model(THE_EXPERIMENT)

    assert spec.weights_folder == "Qwen3.8-27B-FP8"
    assert (spec.max_model_len, spec.max_num_seqs, spec.gpu_memory_utilization) == (16384, 8, 0.90)
    assert (
        spec.mm_processor_kwargs == {"min_pixels": 65536, "max_pixels": 16777216}
        and spec.hf_overrides == {}
    )
    assert spec.chat_template_kwargs == {"enable_thinking": False} and spec.chat_template is None
    assert spec.supports_data_parallel is False and spec.default_image_first is True
    assert spec.prompt_file == "extraction_prompts.yaml"
    # Room for the longest deliberation of September, which was cut short at 1,408 tokens, and the answer.
    assert spec.reply_tokens_added == 2048


# --- the same model with its images held to a size ------------------------------------------------------
# This model gives an image a token for each 1,024 pixels, and its time goes on the image. Each of these
# differs from the model above in the one thing its name says.


@pytest.mark.parametrize(
    ("name", "pixels", "tokens_of_image_at_most"),
    [("4m", 4_194_304, 4096), ("2m", 2_097_152, 2048), ("1m", 1_048_576, 1024), ("05m", 524_288, 512)],
)
def test_a_limit_on_the_image_is_the_one_thing_that_differs(
    name: str, pixels: int, tokens_of_image_at_most: int
) -> None:
    whole, held = get_model(THE_EXPERIMENT), get_model(f"{THE_EXPERIMENT}-{name}")

    assert held.mm_processor_kwargs == {"min_pixels": 65536, "max_pixels": pixels}
    assert pixels // 1024 == tokens_of_image_at_most
    assert (
        dataclasses.replace(
            held,
            model_type=whole.model_type,
            mm_processor_kwargs=whole.mm_processor_kwargs,
            description=whole.description,
        )
        == whole
    )


def test_the_text_first_and_the_sixteen_at_once_each_differ_from_the_one_megapixel_model_in_that_alone() -> (
    None
):
    held = get_model(f"{THE_EXPERIMENT}-1m")
    text_first = get_model(f"{THE_EXPERIMENT}-1m-text-first")
    sixteen = get_model(f"{THE_EXPERIMENT}-1m-16")

    assert (held.default_image_first, text_first.default_image_first) == (True, False)
    assert (held.max_num_seqs, sixteen.max_num_seqs) == (8, 16)
    for other, changed in ((text_first, {"default_image_first": True}), (sixteen, {"max_num_seqs": 8})):
        assert (
            dataclasses.replace(other, model_type=held.model_type, description=held.description, **changed)
            == held
        )


def test_every_form_of_the_experiment_reads_the_same_weights_and_the_same_prompts() -> None:
    forms = [name for name in list_models() if name.startswith(THE_EXPERIMENT)]

    assert len(forms) == 7
    assert {get_model(name).weights_folder for name in forms} == {"Qwen3.8-27B-FP8"}
    assert {get_model(name).prompt_file for name in forms} == {"extraction_prompts.yaml"}
    assert all(get_model(name).chat_template_kwargs == {"enable_thinking": False} for name in forms)
