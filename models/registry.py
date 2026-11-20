"""Model registry with lazy loading for document extraction processors.

Central registry mapping model type strings to loader/creator callables.
All heavy imports (torch, transformers) are deferred to function bodies
so that importing this module has zero GPU/ML overhead.
"""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable

type ModelLoader = Callable[..., AbstractContextManager[tuple[Any, Any]]]
type ProcessorCreator = Callable[..., Any]


@dataclass
class ModelRegistration:
    """Registration entry for a vision-language model.

    Attributes:
        model_type: Short identifier, e.g. "internvl3".
        loader: Callable(PipelineConfig) -> ContextManager[(model, tokenizer)].
        processor_creator: Callable(model, tokenizer, config, prompt_config,
                           universal_fields, field_definitions) -> DocumentProcessor.
        prompt_file: Extraction prompt YAML filename, e.g. "internvl3_prompts.yaml".
        description: Human-readable description for help text.
    """

    model_type: str
    loader: ModelLoader
    processor_creator: ProcessorCreator
    prompt_file: str
    description: str = ""
    requires_sharding: bool = False  # True = model must shard across GPUs (keep device_map="auto")
    is_vllm: bool = False  # True = vLLM backend (eligible for data-parallel)
    # False = one whole engine per GPU won't fit; the DP fast path must not fire.
    supports_data_parallel: bool = True


_REGISTRY: dict[str, ModelRegistration] = {}


def register_model(registration: ModelRegistration) -> None:
    """Register a model type. Overwrites any existing registration."""
    _REGISTRY[registration.model_type] = registration


def get_model(model_type: str) -> ModelRegistration:
    """Look up a registered model by type string.

    Raises:
        ValueError: If the model_type is not registered, with a list of available types.
    """
    if model_type not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(
            f"Unknown model type: {model_type!r}.\n"
            f"  What: model type {model_type!r} is not registered in models/registry.py.\n"
            "  Where: config/run_config.yml -> bootstrap.model.type (the single source that "
            "selects the model); a typo or a retired model name here is the usual cause.\n"
            f"  Expected: one of the registered types: {available}\n"
            "  How to fix: set bootstrap.model.type in config/run_config.yml to a registered "
            "value (e.g. bootstrap:\\n  model:\\n    type: gemma4-31b-w4a16-vllm)."
        )
    return _REGISTRY[model_type]


def list_models() -> list[str]:
    """Return sorted list of registered model type strings."""
    return sorted(_REGISTRY)


def is_vllm_model(model_type: str) -> bool:
    """Check if a registered model uses the vLLM backend."""
    reg = _REGISTRY.get(model_type)
    return reg is not None and reg.is_vllm


def supports_data_parallel(model_type: str) -> bool:
    """Check whether *model_type* can run one independent engine per GPU.

    False for models too large to replicate per GPU (a quantised 31B still wants
    the whole card once KV cache and vision activations are counted). Unknown
    types answer True — ``get_model()`` is the place that rejects them.
    """
    reg = _REGISTRY.get(model_type)
    return reg.supports_data_parallel if reg else True


def _get_requires_sharding(model_type: str) -> bool:
    """Check if a registered model requires cross-GPU sharding."""
    reg = _REGISTRY.get(model_type)
    return reg.requires_sharding if reg else False


# ============================================================================
# Declarative registrations via VllmSpec
#
# Each VllmSpec replaces ~120 lines of hand-written vLLM loader + creator.
# ============================================================================

from models.model_loader import VllmSpec, register_vllm_model  # noqa: E402

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
        prompt_file="internvl3_prompts.yaml",
        description="Gemma 4 31B-it QAT W4A16 via vLLM (21.67 GiB, 2xL4 tp=2)",
        # Gemma's chat template reasons by default and honours enable_thinking,
        # so it is suppressed here rather than through a template file.
        chat_template_kwargs={"enable_thinking": False},
        # The weights exceed a single card, so one engine on each card is
        # impossible and the data-parallel path must not be taken.
        supports_data_parallel=False,
        # The model card recommends image content before the text.
        default_image_first=True,
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
        prompt_file="internvl3_prompts.yaml",
        description="Gemma 4 12B-it Unified QAT W4A16 via vLLM (9.56 GiB, 2xL4 DP=2/tp=1)",
        chat_template_kwargs={"enable_thinking": False},
        supports_data_parallel=True,
        default_image_first=True,
    )
)
