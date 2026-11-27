"""The models the pipeline can run, by the name a task asks for.

A model is one ``register_vllm_model(VllmSpec(...))`` call below. Its specification holds what is true of
the model itself, and what its engine is started with on a card of 24 GB: the context length, how many
requests it runs at once, its share of the card's memory and its number of image tokens. Until
26 November 2026 those were a block for each model in the settings file.

Nothing here imports the engine, so reading the registry costs nothing.
"""

from dataclasses import dataclass, field
from typing import Any

from common.errors import ConfigError


@dataclass(frozen=True)
class VllmSpec:
    """What is true of one model, whatever a run's settings.

    Attributes:
        model_type: The name a task asks for, e.g. "gemma4-31b-w4a16-vllm".
        weights_folder: The name of the folder of its weights, under the folder a job names in
            ``DOC_EXTRACT_MODELS_DIR``. A model's weights cannot be loaded under another's name.
        prompt_file: The file of its extraction prompts, in ``prompts/``.
        description: A line for a reader.
        chat_template_kwargs: Sent with every request. For a chat template that can think,
            ``{"enable_thinking": False}``. A model whose template ignores that needs a
            ``chat_template`` of its own instead.
        supports_data_parallel: Whether one whole engine fits on each card. False for a model too large
            for one card: starting an engine on each would run out of memory.
        default_image_first: Whether a request puts the image before the text. The model's chat template
            renders the two in the order given, so this is part of what the model is sent.
        max_model_len: The longest request and reply together, in tokens.
        max_num_seqs: How many requests the engine runs at once. An engine call may hold more: the
            engine queues the rest, and stays full while it does.
        gpu_memory_utilization: The share of each card's memory the engine may take. A job may set
            another in ``DOC_EXTRACT_GPU_MEMORY_UTILIZATION``.
        mm_processor_kwargs: The number of image tokens (``IMAGE_TOKENS``), as the model's processor is
            asked for it.
        hf_overrides: The same number as the model's own configuration names it. The two models name
            it differently.
        chat_template: The path of a ``*.jinja`` file to use in place of the model's own chat template,
            or None for the model's own: the last resort for a model whose template thinks by default.
        reply_tokens_added: Tokens added to the budget of every reply. 0 for a model that answers at
            once. For one that deliberates before it answers, even with thinking off, enough that the
            deliberation cannot use up the budget before the answer is written.
    """

    model_type: str
    weights_folder: str
    max_model_len: int
    max_num_seqs: int
    gpu_memory_utilization: float
    mm_processor_kwargs: dict[str, Any]
    hf_overrides: dict[str, Any]
    prompt_file: str = "extraction_prompts.yaml"
    description: str = ""
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    supports_data_parallel: bool = True
    default_image_first: bool = False
    chat_template: str | None = None
    reply_tokens_added: int = 0


# The image tokens a page is given. Gemma 4 takes 70, 140, 280, 560 or 1120: 560 is the measured best
# for both models (280 loses small print, 1120 is far worse), so it is not to be raised.
IMAGE_TOKENS = 560

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
            "  Where: the variable model in the job's environment; a typo or a retired model name is "
            "the usual cause.\n"
            f"  Expected: one of the registered types: {available}\n"
            "  How to fix: name a registered model, e.g. model=gemma4-31b-w4a16-vllm."
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
register_vllm_model(
    VllmSpec(
        model_type="gemma4-31b-w4a16-vllm",
        weights_folder="gemma-4-31B-it-qat-w4a16-ct",
        max_model_len=8192,
        # 16 is where the gain stops: the engine's cache holds about 16 bank statements at once.
        max_num_seqs=16,
        gpu_memory_utilization=0.90,
        mm_processor_kwargs={"max_soft_tokens": IMAGE_TOKENS},
        hf_overrides={
            "vision_soft_tokens_per_image": IMAGE_TOKENS,
            "vision_config": {"default_output_length": IMAGE_TOKENS},
        },
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
        weights_folder="gemma-4-12B-it-qat-w4a16-ct",
        # What every run has used.
        max_model_len=8192,
        max_num_seqs=16,
        gpu_memory_utilization=0.90,
        mm_processor_kwargs={"max_soft_tokens": IMAGE_TOKENS},
        # This model names the number vision_config.num_soft_tokens. The 31B's names set nothing here.
        hf_overrides={"vision_config": {"num_soft_tokens": IMAGE_TOKENS}},
        prompt_file="extraction_prompts.yaml",
        description="Gemma 4 12B-it Unified QAT W4A16 via vLLM (9.56 GiB, 2xL4 DP=2/tp=1)",
        chat_template_kwargs={"enable_thinking": False},
        supports_data_parallel=True,
        # The text before the image, as for the 31B and for the reason given there.
        default_image_first=False,
    )
)

# -- Qwen 3.8 27B, FP8: AN EXPERIMENT, not a model the pipeline runs ------------------
# Registered on this branch alone, to measure it again beside the Gemma 31B. It was measured once, on
# 22 September 2026 (mean F1 0.937 against Gemma's 0.987 on 165 clean pages), removed on 20 November
# with the other models no longer run, and is not to be merged as it stands.
#
# Qwen/Qwen3.8-27B-FP8. The weights are 27,781,427,952 B = 25.87 GiB in 66 shards, more than one
# 24 GiB L4 holds, so this is one engine across both cards (tp=2), as the Gemma 31B is. Measured at
# load on 21 September: 14.86 GiB a card, and a cache for 86,167 tokens.
#
# The engine's values are those of 22 September, so that what differs from that run is the pipeline
# and the one thing changed on purpose (reply_tokens_added):
#   max_model_len  16384 holds a bank statement at its own size (about 6,166 image tokens) with the
#                  prompt and the reply.
#   max_num_seqs   8, as it was run.
#   mm_processor_kwargs  the checkpoint's own limits, written out. This model sizes an image by its
#                  area in pixels (image tokens = pixels / 1024), not by a number of tokens asked for,
#                  so it takes no hf_overrides.
#
# Thinking is off (chat_template_kwargs). Its chat template was read on 27 November 2026: given
# enable_thinking false it writes an empty, closed block of thinking after the assistant's header
# and adds no instruction to reason. On 22 September the model deliberated in plain text all the
# same, on 107 of 165 pages, and closed a block it had not opened: about three times the output of
# the Gemma, and on four receipts the deliberation used up the budget before the answer was written.
# reply_tokens_added is what is changed for that: room for the deliberation AND the answer. It does
# not stop the deliberation, which is the model's own and is paid for in time.
register_vllm_model(
    VllmSpec(
        model_type="qwen38-27b-fp8-vllm",
        weights_folder="Qwen3.8-27B-FP8",
        max_model_len=16384,
        max_num_seqs=8,
        gpu_memory_utilization=0.90,
        mm_processor_kwargs={"min_pixels": 65536, "max_pixels": 16777216},
        hf_overrides={},
        prompt_file="extraction_prompts.yaml",
        description="Qwen 3.8 27B FP8 via vLLM (25.87 GiB, 2xL4 tp=2): an experiment",
        chat_template_kwargs={"enable_thinking": False},
        supports_data_parallel=False,
        # The image before the text, as its chat template's own examples have it and as it was run on
        # 22 September. The Gemma models are sent the text first, a choice measured for them.
        default_image_first=True,
        # The longest replies of 22 September were about 5,000 characters, cut short at 1,408 tokens.
        reply_tokens_added=2048,
    )
)
