"""Every setting the pipeline knows, so that one it does not know is refused.

A settings file that carries a key no code reads cannot be trusted to say what a run does: the key looks
like an instruction and is ignored. So the keys are declared here, once, and a file is checked against
them before any work begins. A key that is not declared stops the run.

This is the one place the keys are named. It holds no values and no rules about values: those checks stay
with the code that reads each setting.

Two marks stand for levels whose names are data and not settings:

- ``*`` is one level of any name, such as a model's name under ``inference.vllm.models``.
- ``**`` at the end is everything beneath, for the blocks handed to the engine as they are
  (``mm_processor_kwargs`` and ``hf_overrides``), whose keys belong to vLLM and the model.
"""

from pathlib import Path
from typing import Any

KNOWN_KEYS: frozenset[str] = frozenset(
    {
        "bootstrap.gpus.data_parallel_size",
        "bootstrap.gpus.num_gpus",
        "bootstrap.logging.log_dir",
        "bootstrap.model.chat_template",
        "bootstrap.model.default_paths.*",
        "bootstrap.model.device_map",
        "bootstrap.model.dtype",
        "bootstrap.model.enforce_eager",
        "bootstrap.model.flash_attn",
        "bootstrap.model.path",
        "bootstrap.model.type",
        "inference.generation.defaults.do_sample",
        "inference.generation.defaults.max_new_tokens_base",
        "inference.generation.defaults.max_new_tokens_per_field",
        "inference.generation.defaults.temperature",
        "inference.generation.defaults.top_p",
        "inference.generation.defaults.use_cache",
        "inference.generation.models.*.max_new_tokens_base",
        "inference.generation.models.*.max_new_tokens_per_field",
        "inference.generation.token_limits.2b",
        "inference.generation.token_limits.8b",
        "inference.max_new_tokens",
        "inference.tracing.path",
        "inference.tracing.raw_prompts",
        "inference.vllm.defaults.enable_prefix_caching",
        "inference.vllm.defaults.gpu_memory_utilization",
        "inference.vllm.defaults.limit_mm_per_prompt",
        "inference.vllm.defaults.max_model_len",
        "inference.vllm.defaults.max_num_seqs",
        "inference.vllm.models.*.enable_prefix_caching",
        "inference.vllm.models.*.gpu_memory_utilization",
        "inference.vllm.models.*.hf_overrides.**",
        "inference.vllm.models.*.limit_mm_per_prompt",
        "inference.vllm.models.*.max_model_len",
        "inference.vllm.models.*.max_num_seqs",
        "inference.vllm.models.*.mm_processor_kwargs.**",
        "inference.vllm.models.*.stage_overrides.classify.hf_overrides.**",
        "inference.vllm.models.*.stage_overrides.classify.mm_processor_kwargs.**",
        "pipeline.batch.auto_detect",
        "pipeline.batch.conservative_sizes.*",
        "pipeline.batch.default_sizes.*",
        "pipeline.batch.max_sizes.*",
        "pipeline.batch.min_size",
        "pipeline.batch.strategy",
        "pipeline.batching.bank_statement",
        "pipeline.batching.classify",
        "pipeline.batching.default",
        "pipeline.batching.invoice",
        "pipeline.batching.receipt",
        "pipeline.classification.fallback_type",
        "pipeline.extraction.order",
        "pipeline.extraction.secondary_sort",
        "pipeline.extraction.skip_labels",
        "pipeline.information_extraction.input.dir",
        "pipeline.information_extraction.input.document_types",
        "pipeline.information_extraction.input.ground_truth",
        "pipeline.information_extraction.input.max_images",
        "pipeline.information_extraction.output.dir",
        "pipeline.information_extraction.output.skip_reports",
        "pipeline.information_extraction.output.skip_visualizations",
        "pipeline.photo_split.batch_size",
        "pipeline.photo_split.enabled",
        "pipeline.photo_split.finder.blur_sigma_px",
        "pipeline.photo_split.finder.desk_ring_frac",
        "pipeline.photo_split.finder.drift_row_percentile",
        "pipeline.photo_split.finder.edge_gap_close_px",
        "pipeline.photo_split.finder.edge_seal_px",
        "pipeline.photo_split.finder.edge_tolerance_px",
        "pipeline.photo_split.finder.mark_block_px",
        "pipeline.photo_split.finder.mark_darker_by",
        "pipeline.photo_split.finder.mark_max_heights",
        "pipeline.photo_split.finder.mask_min_distance",
        "pipeline.photo_split.finder.mask_noise_multiple",
        "pipeline.photo_split.finder.min_edge_support",
        "pipeline.photo_split.finder.min_outline_edge_frac",
        "pipeline.photo_split.finder.min_region_frac",
        "pipeline.photo_split.finder.outline_tolerance_px",
        "pipeline.photo_split.finder.paper_block_in_box_min",
        "pipeline.photo_split.finder.paper_block_min_share",
        "pipeline.photo_split.finder.paper_border_frac",
        "pipeline.photo_split.finder.paper_extension_min_fit",
        "pipeline.photo_split.finder.paper_join_heights",
        "pipeline.photo_split.finder.paper_min_border_difference",
        "pipeline.photo_split.finder.paper_refine_inside_frac",
        "pipeline.photo_split.finder.paper_refine_outside_frac",
        "pipeline.photo_split.finder.paper_region_grow_frac",
        "pipeline.photo_split.finder.paper_rounds",
        "pipeline.photo_split.finder.paper_work_long_edge",
        "pipeline.photo_split.finder.ransac_seed",
        "pipeline.photo_split.finder.refine_inside_final_frac",
        "pipeline.photo_split.finder.refine_inside_first_frac",
        "pipeline.photo_split.finder.refine_outside_final_frac",
        "pipeline.photo_split.finder.refine_outside_first_frac",
        "pipeline.photo_split.finder.shadow_min_factor",
        "pipeline.photo_split.finder.shadow_reach_frac",
        "pipeline.photo_split.finder.shadow_reach_heights",
        "pipeline.photo_split.finder.step_min",
        "pipeline.photo_split.finder.step_min_area_px",
        "pipeline.photo_split.finder.step_span_px",
        "pipeline.photo_split.finder.text_agree_deg",
        "pipeline.photo_split.finder.text_join_heights",
        "pipeline.photo_split.finder.text_max_row_thickness_heights",
        "pipeline.photo_split.finder.text_min_row_heights",
        "pipeline.photo_split.finder.text_min_rows",
        "pipeline.photo_split.finder.work_long_edge",
        "pipeline.photo_split.margin_frac",
        "pipeline.photo_split.max_overlap_frac",
        "pipeline.photo_split.max_receipts",
        "pipeline.photo_split.max_tokens",
        "pipeline.photo_split.min_box_frac",
        "pipeline.photo_split.prompt",
        "pipeline.photo_split.rectify_min_area_frac",
        "pipeline.photo_split.rectify_min_edges",
        "pipeline.photo_split.rectify_min_print_frac",
        "pipeline.photo_split.rectify_pages",
        "pipeline.photo_split.rectify_paper_min_fit",
        "pipeline.photo_split.rectify_paper_model",
        "pipeline.photo_split.rectify_pieces",
        "pipeline.photo_split.straighten",
        "pipeline.photo_split.straighten_fill_rgb",
        "pipeline.photo_split.straighten_max_deg",
        "pipeline.photo_split.straighten_min_deg",
        "pipeline.photo_split.straighten_min_drift_heights",
        "pipeline.processing.batch_size",
        "pipeline.processing.debug",
        "pipeline.processing.verbose",
        "pipeline.sroie.batch_size",
        "pipeline.sroie.data_dir",
        "pipeline.sroie.max_new_tokens",
        "pipeline.sroie.output_dir",
        "pipeline.token_budgets.bank_statement_floor",
        "pipeline.token_budgets.classify",
        "pipeline.token_budgets.extract_bank",
        "pipeline.token_budgets.fallback_base",
        "pipeline.token_budgets.fallback_per_field",
        "resources.gpu_memory.high",
        "resources.gpu_memory.low",
        "resources.gpu_memory.medium",
        "resources.gpu_memory.very_high",
        "resources.infrastructure.dp_join_timeout",
        "resources.infrastructure.gpu_memory_fallback_gb",
        "resources.infrastructure.gpu_memory_threshold_gb",
    }
)


def leaf_paths(raw: dict[str, Any], prefix: str = "") -> list[str]:
    """The dotted path of every leaf in a settings mapping. An empty mapping is itself a leaf."""
    paths: list[str] = []
    for key, value in raw.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            paths.extend(leaf_paths(value, f"{path}."))
        else:
            paths.append(path)
    return paths


def is_known(path: str) -> bool:
    """Whether a leaf's dotted path matches a declared key."""
    parts = path.split(".")
    return any(_matches(parts, known.split(".")) for known in KNOWN_KEYS)


def _matches(parts: list[str], known: list[str]) -> bool:
    if known[-1] == "**":
        head = known[:-1]
        return len(parts) > len(head) and all(k in ("*", p) for k, p in zip(head, parts, strict=False))
    return len(parts) == len(known) and all(k in ("*", p) for k, p in zip(known, parts, strict=True))


def check_known_keys(raw: dict[str, Any], *, config_path: Path) -> None:
    """Stop the run if the settings hold a key the pipeline does not know.

    Args:
        raw: The settings file as loaded.
        config_path: Where it was loaded from, for the message.

    Raises:
        ValueError: Naming every unknown key, where it is, what is expected and how to mend it.
    """
    unknown = [path for path in leaf_paths(raw) if not is_known(path)]
    if not unknown:
        return
    listed = "\n".join(f"    {path}" for path in unknown)
    section = unknown[0].rsplit(".", 1)[0]
    examples = sorted(key for key in KNOWN_KEYS if key.startswith(f"{section}."))[:6]
    expected = "\n".join(f"    {key}" for key in examples) or "    (no setting is known under that section)"
    raise ValueError(
        f"What: the settings file holds {len(unknown)} key(s) the pipeline does not know:\n{listed}\n"
        f"Where: {config_path} -> {unknown[0]}\n"
        f"Expected: only keys declared in common/config_schema.py. Under '{section}' these are known:\n"
        f"{expected}\n"
        f"How to fix: remove the key(s) from {config_path}. A key from an older settings file names "
        f"something that has been removed from the pipeline, so it has no effect to keep. If the key is "
        f"misspelt, correct it to one of the known keys."
    )
