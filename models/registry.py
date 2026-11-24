"""The models the pipeline can run, by the name a task asks for.

A model is one ``register_vllm_model(VllmSpec(...))`` call below. Its specification holds what is true of
the model itself. How its engine is tuned (memory, context length, how many requests at once, the image
token budget) is in ``config/run_config.yml`` under ``inference.vllm.models``.

Nothing here imports the engine, so reading the registry costs nothing.
"""

from dataclasses import dataclass, field
from typing import Any

from common.config_schema import ConfigError


@dataclass(frozen=True)
class VllmSpec:
    """What is true of one model, whatever a run's settings.

    Attributes:
        model_type: The name a task asks for, e.g. "gemma4-31b-w4a16-vllm".
        prompt_file: The file of its extraction prompts, in ``prompts/``.
        description: A line for a reader.
        chat_template_kwargs: Sent with every request. For a chat template that can think,
            ``{"enable_thinking": False}``. A model whose template ignores that needs
            ``bootstrap.model.chat_template`` in the settings file instead.
        supports_data_parallel: Whether one whole engine fits on each card. False for a model too large
            for one card: starting an engine on each would run out of memory.
        default_image_first: Whether a request puts the image before the text. The model's chat template
            renders the two in the order given, so this is part of what the model is sent.
    """

    model_type: str
    prompt_file: str = "extraction_prompts.yaml"
    description: str = ""
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    supports_data_parallel: bool = True
    default_image_first: bool = False


_REGISTRY: dict[str, VllmSpec] = {}


def register_vllm_model(spec: VllmSpec) -> None:
    """Add a model. A second specification under the same name takes the first's place."""
    _REGISTRY[spec.model_type] = spec


def get_model(model_type: str) -> VllmSpec:
    """A registered model's specification.

    Raises:
        ConfigError: If no model is registered under the name, with the names that are.
    """
    if model_type not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ConfigError(
            f"Unknown model type: {model_type!r}.\n"
            f"  What: model type {model_type!r} is not registered in models/registry.py.\n"
            "  Where: the variable model in the job's environment, or config/run_config.yml -> "
            "bootstrap.model.type; a typo or a retired model name is the usual cause.\n"
            f"  Expected: one of the registered types: {available}\n"
            "  How to fix: name a registered model, e.g. model=gemma4-31b-w4a16-vllm, or "
            "bootstrap:\\n  model:\\n    type: gemma4-31b-w4a16-vllm in config/run_config.yml."
        )
    return _REGISTRY[model_type]


def list_models() -> list[str]:
    """The registered models' names, sorted."""
    return sorted(_REGISTRY)


def supports_data_parallel(model_type: str) -> bool:
    """Whether a model can run as one whole engine on each card.

    A name that is not registered answers True: ``get_model`` is the place that refuses it.
    """
    spec = _REGISTRY.get(model_type)
    return spec.supports_data_parallel if spec else True


# -- Gemma 4 31B -----------------------------------------------------------------
# Google's own QAT W4A16 checkpoint (compressed-tensors), which vLLM loads natively:
# 23,265,352,448 B = 21.67 GiB on disk. The vision embedder is excluded from
# quantisation, which is why it is not the 18 GB a plain 4-bit estimate gives.
#
# 21.67 GiB does not fit one 24 GiB L4 (about 22.5 usable), so this runs as ONE
# engine split across both cards (tp=2), over PCIe with no NVLink. Throughput
# measured that way is less than the model would give on one adequate card.
#
# Engine tuning (soft-token budget, max_model_len, gpu_memory_utilization) lives in
# run_config.yml under inference.vllm.models. Only capabilities are stated here.
register_vllm_model(
    VllmSpec(
        model_type="gemma4-31b-w4a16-vllm",
        prompt_file="extraction_prompts.yaml",
        description="Gemma 4 31B-it QAT W4A16 via vLLM (21.67 GiB, 2xL4 tp=2)",
        # Thinking off. This model's chat template takes a request without the argument as
        # thinking off too (rendered both ways on 21 November 2026), so this states the default.
        chat_template_kwargs={"enable_thinking": False},
        # The weights exceed a single card, so one engine on each card is
        # impossible and the data-parallel path must not be taken.
        supports_data_parallel=False,
        # The text before the image, for every task. The model card's examples put the image first,
        # but that order was measured on receipts and invoices on 24 November 2026 and cost 16% of
        # extract time for no gain: with the text first the engine reuses its work on the prompt
        # from one page to the next.
        default_image_first=False,
    )
)

# -- Gemma 4 12B, quantised by its maker ---------------------------------------------
# Gemma 4 12B "Unified", Google's QAT W4A16 (compressed-tensors). A different
# architecture from the 31B, not only a smaller size: Gemma4UnifiedForConditionalGeneration
# is encoder-free and projects raw image patches straight into the embedding space.
# Verified 2026-08-11 against the checkpoint's own config.json. Needs vLLM >= 0.23.0.
#
# 10,264,229,896 bytes (9.56 GiB), not the 6 GB a plain 4-bit estimate gives: the
# quantisation ignore list excludes the vision embedder, the patch dense layers and
# lm_head.
#
# At 9.56 GiB of about 22.5 GiB a card, one WHOLE engine fits on each GPU, so
# supports_data_parallel is True and run_dp starts one engine on each L4 (tp=1 each).
# That path uses no NCCL and no /dev/shm (see common/vllm_dp.py).
#
# Thinking is opt-in on this model, enabled by a `<|think|>` token at the start of a
# system prompt. This pipeline sends a bare user message with no system prompt, so
# thinking is already off; enable_thinking=False is belt and braces.
register_vllm_model(
    VllmSpec(
        model_type="gemma4-12b-unified-w4a16-vllm",
        prompt_file="extraction_prompts.yaml",
        description="Gemma 4 12B-it Unified QAT W4A16 via vLLM (9.56 GiB, 2xL4 DP=2/tp=1)",
        chat_template_kwargs={"enable_thinking": False},
        supports_data_parallel=True,
        # The text before the image, as for the 31B and for the reason given there.
        default_image_first=False,
    )
)
