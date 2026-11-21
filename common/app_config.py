"""Unified application configuration.

One way in (AppConfig.load) for every stage, and no module-level state that can be changed
after loading.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from common.vllm_stage_overrides import STAGES, apply_stage_overrides, validate_stage_overrides

if TYPE_CHECKING:
    from common.pipeline_config import PipelineConfig

_VALID_SECONDARY_SORTS = ("none", "image_area_asc", "image_area_desc")


class ConfigError(Exception):
    """Raised by AppConfig.load() when validation fails."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


def _validate_model_override_keys(
    *,
    vllm_models: dict,
    generation_models: dict,
) -> None:
    """Reject per-model override keys that no registered model resolves to.

    Both ``inference.vllm.models`` and ``inference.generation.models`` fall back
    to their ``defaults`` block when a key is absent. That fallback is
    deliberate -- it is how a model inherits shared tuning -- but it makes a
    TYPO indistinguishable from inheritance: the block reads as configured
    while doing nothing, and the run completes on default tuning.

    The two sections use DIFFERENT key conventions, which is the trap this
    guards. ``inference.vllm.models`` is keyed by the FULL registered type
    (``gemma4-12b-unified-w4a16-vllm``); ``inference.generation.models`` is
    keyed by the NORMALISED form (``gemma4-12b-unified``), and the
    normalisation is not uniform across suffixes.

    Raises:
        ValueError: If any key matches no registered model, with a
            four-element diagnostic naming the offending keys.
    """
    from models.registry import list_models

    registered = list_models()
    normalised = {AppConfig._normalize_model_type(name) for name in registered}

    unknown_vllm = sorted(set(vllm_models) - set(registered))
    unknown_generation = sorted(set(generation_models) - normalised)
    if not unknown_vllm and not unknown_generation:
        return

    problems = []
    if unknown_vllm:
        problems.append(f"inference.vllm.models: {unknown_vllm}")
    if unknown_generation:
        problems.append(f"inference.generation.models: {unknown_generation}")

    msg = (
        f"What:  per-model override key(s) match no registered model, so the "
        f"block is dead config -- the lookup silently falls through to the "
        f"`defaults` block and the tuning is never applied. "
        f"Offending: {'; '.join(problems)}\n"
        f"  Where: config/run_config.yml -> {' and '.join(p.split(':')[0] for p in problems)}\n"
        f"  Expected: inference.vllm.models is keyed by the FULL registered "
        f"type, one of: {', '.join(sorted(registered))}. "
        f"inference.generation.models is keyed by the NORMALISED type, one of: "
        f"{', '.join(sorted(normalised))}. The two conventions differ and the "
        f"normalisation is not uniform -- check it rather than inferring it.\n"
        f"  How to fix: correct the key to a listed value, or delete the block "
        f"if the model should inherit the `defaults` above it."
    )
    raise ValueError(msg)


def _build_generation_registry(raw_config: dict, config_file: str) -> dict[str, dict]:
    """The generation settings of each model: ``generation.defaults`` with the model's own laid over them.

    Raises:
        ValueError: If ``inference.generation.defaults`` is missing. Nothing is assumed in its place.
    """
    gen = raw_config.get("inference", {}).get("generation", {})
    if not isinstance(gen.get("defaults"), dict):
        raise ValueError(
            "What: the settings file has no inference.generation.defaults block.\n"
            f"Where: {config_file}, key inference.generation.defaults.\n"
            "Expected: a block such as\n"
            "  inference:\n"
            "    generation:\n"
            "      defaults:\n"
            "        max_new_tokens_base: 512\n"
            "        max_new_tokens_per_field: 64\n"
            "        temperature: 0.0\n"
            "        do_sample: false\n"
            "        top_p: 0.95\n"
            "        use_cache: true\n"
            f"How to fix: add that block to {config_file}."
        )
    defaults = dict(gen["defaults"])
    registry: dict[str, dict] = {
        model_name: {**defaults, **overrides} for model_name, overrides in (gen.get("models") or {}).items()
    }
    # What a model with no block of its own gets.
    registry["__defaults__"] = defaults
    return registry


class AppConfig:
    """Unified, immutable configuration surface.

    Constructed once at startup, threaded to all consumers.
    Nothing in it is changed after loading.
    """

    __slots__ = (
        "pipeline",
        "_generation_registry",
        "_token_budgets",
        "_vllm_config",
        "_infrastructure",
        "_classification",
        "_extraction_order",
        "_secondary_sort",
        "_extraction_skip_labels",
        "_stage",
        "config_path",
    )

    def __init__(
        self,
        pipeline: "PipelineConfig",
        generation_registry: dict[str, dict],
        token_budgets: dict[str, int] | None = None,
        vllm_config: dict[str, dict] | None = None,
        infrastructure: dict[str, int | float] | None = None,
        classification: dict[str, str] | None = None,
        extraction_order: list[str] | None = None,
        secondary_sort: str = "none",
        extraction_skip_labels: list[str] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self._generation_registry = generation_registry
        self._token_budgets = token_budgets or {}
        self._vllm_config = vllm_config or {}
        self._infrastructure = dict(infrastructure or {})
        self._classification = classification or {}
        self._extraction_order = extraction_order or []
        self._secondary_sort = secondary_sort
        self._extraction_skip_labels = extraction_skip_labels or []
        self._stage: str | None = None
        # The settings file this was loaded from; None when it was built without one.
        self.config_path: Path | None = None

    @classmethod
    def load(
        cls,
        cli_args: dict[str, Any],
        *,
        config_path: Path | None = None,
    ) -> AppConfig:
        """Single entry point. No mutable globals touched.

        Handles: YAML loading, merge (CLI > YAML > defaults),
        validation, batch settings, generation registry, field schema,
        token budgets, vLLM config, infrastructure settings.

        Raises:
            ConfigError: On validation failure.
            FileNotFoundError: If an explicit config_path does not exist.
        """
        from common.pipeline_config import (
            load_yaml_config,
            merge_configs,
            validate_config,
        )

        # 1. Resolve config path
        default = Path(__file__).parent.parent / "config" / "run_config.yml"
        resolved = config_path or (default if default.exists() else None)
        config_file = str(resolved) if resolved else "config/run_config.yml"

        # 2. Load YAML
        yaml_config: dict[str, Any] = {}
        raw_config: dict[str, Any] = {}
        if resolved:
            yaml_config, raw_config = load_yaml_config(resolved)

        # 3. Pre-validate required fields (before PipelineConfig construction)
        merged_preview = {**yaml_config, **cli_args}
        errors: list[str] = []
        if not merged_preview.get("data_dir"):
            errors.append(
                "--data-dir is required (via CLI, or pipeline.information_extraction.input.dir "
                "in config/run_config.yml)"
            )
        if not merged_preview.get("output_dir"):
            errors.append(
                "--output-dir is required (via CLI, or pipeline.information_extraction.output.dir "
                "in config/run_config.yml)"
            )
        if errors:
            raise ConfigError(errors)

        # 3b. Extract vLLM-specific CLI overrides before PipelineConfig merge
        #     (PipelineConfig doesn't know about max_num_seqs)
        cli_max_num_seqs = cli_args.pop("max_num_seqs", None)

        # 4. Merge with precedence: CLI > YAML > defaults
        pipeline = merge_configs(cli_args, yaml_config, raw_config)

        # 5. Validate
        val_errors = validate_config(pipeline)
        if val_errors:
            raise ConfigError(val_errors)

        # 7. Build generation registry (immutable copy with YAML overrides)
        generation_registry = _build_generation_registry(raw_config, config_file)

        # 11. Token budgets — YAML is the single source of truth
        yaml_budgets = raw_config.get("pipeline", {}).get("token_budgets", {})

        # 12. Build vLLM config (defaults + per-model overrides)
        vllm_section = raw_config.get("inference", {}).get("vllm", {})
        vllm_defaults = vllm_section.get("defaults")
        if not isinstance(vllm_defaults, dict):
            raise ValueError(
                "What: the settings file has no inference.vllm.defaults block.\n"
                f"Where: {config_file}, key inference.vllm.defaults.\n"
                "Expected: a block such as\n"
                "  inference:\n"
                "    vllm:\n"
                "      defaults:\n"
                "        gpu_memory_utilization: 0.90\n"
                "        max_model_len: 8192\n"
                "        max_num_seqs: 1\n"
                "        limit_mm_per_prompt: 1\n"
                "        enable_prefix_caching: true\n"
                f"How to fix: add that block to {config_file}."
            )
        vllm_models = vllm_section.get("models", {})

        # A per-model key that matches no registered model is dead config: the
        # lookup falls through to `defaults` and the block reads as configured
        # while doing nothing. Catch it at startup rather than at the end of a
        # GPU run whose tuning was silently ignored.
        _validate_model_override_keys(
            vllm_models=vllm_models,
            generation_models=raw_config.get("inference", {}).get("generation", {}).get("models", {}),
        )
        validate_stage_overrides(
            vllm_models, config_path=str(resolved) if resolved else "config/run_config.yml"
        )
        vllm_config: dict[str, dict] = {"__defaults__": dict(vllm_defaults)}
        for model_name, overrides in vllm_models.items():
            vllm_config[model_name] = {**vllm_defaults, **overrides}

        # 12b. Apply CLI max_num_seqs override (beats YAML per-model and defaults)
        if cli_max_num_seqs is not None:
            for key in vllm_config:
                vllm_config[key]["max_num_seqs"] = cli_max_num_seqs

        # 13. Build infrastructure settings
        infra_section = raw_config.get("resources", {}).get("infrastructure", {})

        # 14. Build classification settings
        classification_section = raw_config.get("pipeline", {}).get("classification", {})

        # 15. Validate and build extraction_order
        extraction_order = cls._validate_extraction_order(raw_config, config_file)

        # 16. Validate and build secondary_sort
        secondary_sort = cls._validate_secondary_sort(raw_config, config_file)

        # 17. Validate and build extraction_skip_labels
        skip_labels = cls._validate_extraction_skip_labels(raw_config, config_file)

        loaded = cls(
            pipeline=pipeline,
            generation_registry=generation_registry,
            token_budgets=yaml_budgets,
            vllm_config=vllm_config,
            infrastructure=infra_section,
            classification=classification_section,
            extraction_order=extraction_order,
            secondary_sort=secondary_sort,
            extraction_skip_labels=skip_labels,
        )
        loaded.config_path = resolved
        return loaded

    # -- Token budgets (Phase 1) -----------------------------------------------

    def get_token_budget(self, name: str) -> int:
        """Resolve a named token budget.

        Args:
            name: Budget name (e.g. "classify", "extract_bank").

        Returns:
            Token count for the named budget.

        Raises:
            KeyError: If *name* is not a known budget, with available names listed.
        """
        try:
            return self._token_budgets[name]
        except KeyError:
            available = ", ".join(sorted(self._token_budgets))
            msg = f"Unknown token budget {name!r}. Available budgets: {available}"
            raise KeyError(msg) from None

    # -- vLLM config (Phase 3) -------------------------------------------------

    def for_stage(self, stage: str) -> AppConfig:
        """Return a copy whose vLLM settings are the ones *stage* loads its engine with.

        A model may state settings one stage loads differently, in
        ``inference.vllm.models.<model>.stage_overrides``. Everything else is shared
        with this config.

        Args:
            stage: The stage about to load an engine.

        Raises:
            ValueError: If *stage* is not a stage that takes overrides.
        """
        if stage not in STAGES:
            msg = f"Unknown stage {stage!r} for vLLM stage overrides. Known stages: {', '.join(STAGES)}"
            raise ValueError(msg)
        staged = copy.copy(self)
        staged._stage = stage
        return staged

    def get_vllm_config(self, model_type: str) -> dict[str, Any]:
        """Return vLLM engine parameters for *model_type*.

        A model without a block of its own gets ``inference.vllm.defaults``. On a config made by
        ``for_stage``, the model's overrides for that stage are applied.
        """
        defaults = self._vllm_config["__defaults__"]
        per_model = self._vllm_config.get(model_type, {})
        if per_model:
            return apply_stage_overrides({**defaults, **per_model}, self._stage)
        return dict(defaults)

    # -- Infrastructure (Phase 5) -----------------------------------------------

    def get_infra(self, name: str) -> int | float:
        """Resolve an infrastructure setting by name.

        Args:
            name: Setting name (e.g. "dp_join_timeout", "gpu_memory_threshold_gb").

        Returns:
            The setting value.

        Raises:
            KeyError: If *name* is not a known infrastructure setting.
        """
        try:
            return self._infrastructure[name]
        except KeyError:
            available = ", ".join(sorted(self._infrastructure))
            msg = f"Unknown infrastructure setting {name!r}. Available settings: {available}"
            raise KeyError(msg) from None

    # -- Classification config --------------------------------------------------

    @property
    def classification_fallback_type(self) -> str:
        """Default document type when classification response can't be parsed."""
        return self._classification["fallback_type"]

    # -- Generation settings by model -------------------------------------------

    @staticmethod
    def _normalize_model_type(model_type: str) -> str:
        """Strip deployment suffixes to find base model config.

        ``"gemma4-31b-w4a16-vllm"`` -> ``"gemma4-31b"``,
        ``"gemma4-12b-unified-w4a16-vllm"`` -> ``"gemma4-12b-unified"``.
        """
        key = model_type.lower()
        # Strip -vllm suffix
        if key.endswith("-vllm"):
            key = key[: -len("-vllm")]
        # Strip size suffixes (-8b, -14b, -38b, etc.)
        parts = key.rsplit("-", 1)
        if len(parts) == 2 and parts[1].endswith("b") and parts[1][:-1].isdigit():
            key = parts[0]
        # Strip -w4a16 quantization suffix
        if key.endswith("-w4a16"):
            key = key[: -len("-w4a16")]
        return key

    def get_generation_config(self, model_type: str) -> dict[str, Any]:
        """The generation settings of a model: its own block, else ``generation.defaults``.

        Returns a copy so callers can mutate freely. The name is looked up without its deployment
        suffixes (``-vllm``, the size, ``-w4a16``).
        """
        key = self._normalize_model_type(model_type)
        return dict(self._generation_registry.get(key) or self._generation_registry["__defaults__"])

    # -- Extraction ordering (Phase 2) -----------------------------------------

    @property
    def extraction_order(self) -> list[str]:
        """Ordered list of doc_type values for batch submission sorting."""
        return list(self._extraction_order)

    @property
    def secondary_sort(self) -> str:
        """Secondary sort key within each doc_type group."""
        return self._secondary_sort

    # -- Extraction skip labels (Phase 4) --------------------------------------

    @property
    def extraction_skip_labels(self) -> list[str]:
        """Classification labels that bypass extraction entirely."""
        return list(self._extraction_skip_labels)

    # -- Validation classmethods -----------------------------------------------

    @classmethod
    def _validate_extraction_order(cls, raw_config: dict, config_file: str) -> list[str]:
        """Validate ``pipeline.extraction.order`` key in YAML."""
        extraction_section = raw_config.get("pipeline", {}).get("extraction", {})
        if "order" not in extraction_section:
            raise ConfigError(
                [
                    f"Missing required key 'pipeline.extraction.order' in {config_file}. "
                    f"What: the key 'pipeline.extraction.order' is absent. "
                    f"Where: {config_file} → pipeline.extraction.order. "
                    f"Expected: a YAML list of doc_type strings, e.g.:\n"
                    f"  pipeline:\n"
                    f"    extraction:\n"
                    f"      order:\n"
                    f"        - bank_statement\n"
                    f"        - invoice\n"
                    f"        - receipt\n"
                    f"How to fix: add a 'pipeline.extraction.order:' list to {config_file}."
                ]
            )
        value = extraction_section["order"]
        if not isinstance(value, list):
            raise ConfigError(
                [
                    f"Invalid type for 'pipeline.extraction.order' in {config_file}: "
                    f"expected a list, got {type(value).__name__}. "
                    f"Where: {config_file} → pipeline.extraction.order. "
                    f"Expected: a YAML list of doc_type strings, e.g.:\n"
                    f"  pipeline:\n"
                    f"    extraction:\n"
                    f"      order:\n"
                    f"        - bank_statement\n"
                    f"        - invoice\n"
                    f"        - receipt\n"
                    f"How to fix: change 'pipeline.extraction.order' to a YAML list in {config_file}."
                ]
            )
        return list(value)

    @classmethod
    def _validate_secondary_sort(cls, raw_config: dict, config_file: str) -> str:
        """Validate ``pipeline.extraction.secondary_sort`` key in YAML."""
        extraction_section = raw_config.get("pipeline", {}).get("extraction", {})
        if "secondary_sort" not in extraction_section:
            raise ConfigError(
                [
                    f"Missing required key 'pipeline.extraction.secondary_sort' in {config_file}. "
                    f"What: the key 'pipeline.extraction.secondary_sort' is absent. "
                    f"Where: {config_file} → pipeline.extraction.secondary_sort. "
                    f"Expected: one of {list(_VALID_SECONDARY_SORTS)}, e.g.:\n"
                    f"  pipeline:\n"
                    f"    extraction:\n"
                    f"      secondary_sort: none\n"
                    f"How to fix: add 'pipeline.extraction.secondary_sort: none' to {config_file}."
                ]
            )
        value = extraction_section["secondary_sort"]
        if value not in _VALID_SECONDARY_SORTS:
            raise ConfigError(
                [
                    f"Invalid value for 'pipeline.extraction.secondary_sort' in {config_file}: "
                    f"got {value!r}, expected one of {list(_VALID_SECONDARY_SORTS)}. "
                    f"Where: {config_file} → pipeline.extraction.secondary_sort. "
                    f"How to fix: set 'pipeline.extraction.secondary_sort' to one of "
                    f"{list(_VALID_SECONDARY_SORTS)} in {config_file}."
                ]
            )
        return str(value)

    @classmethod
    def _validate_extraction_skip_labels(cls, raw_config: dict, config_file: str) -> list[str]:
        """Validate ``pipeline.extraction.skip_labels`` key in YAML."""
        extraction_section = raw_config.get("pipeline", {}).get("extraction", {})
        if "skip_labels" not in extraction_section:
            raise ConfigError(
                [
                    f"Missing required key 'pipeline.extraction.skip_labels' in {config_file}. "
                    f"What: the key 'pipeline.extraction.skip_labels' is absent. "
                    f"Where: {config_file} → pipeline.extraction.skip_labels. "
                    f"Expected: a YAML list of label strings (may be empty), e.g.:\n"
                    f"  pipeline:\n"
                    f"    extraction:\n"
                    f"      skip_labels: []\n"
                    f"How to fix: add 'pipeline.extraction.skip_labels: []' to {config_file}."
                ]
            )
        value = extraction_section["skip_labels"]
        # YAML parses `skip_labels:` (no value) as None
        if value is None:
            value = []
        if not isinstance(value, list):
            raise ConfigError(
                [
                    f"Invalid type for 'pipeline.extraction.skip_labels' in {config_file}: "
                    f"expected a list, got {type(value).__name__}. "
                    f"Where: {config_file} → pipeline.extraction.skip_labels. "
                    f"Expected: a YAML list of label strings, e.g.:\n"
                    f"  pipeline:\n"
                    f"    extraction:\n"
                    f"      skip_labels:\n"
                    f"        - junk\n"
                    f"        - blank\n"
                    f"How to fix: change 'pipeline.extraction.skip_labels' to a YAML list "
                    f"in {config_file}."
                ]
            )
        for i, entry in enumerate(value):
            if not isinstance(entry, str):
                raise ConfigError(
                    [
                        f"Invalid entry at index {i} in 'pipeline.extraction.skip_labels' "
                        f"in {config_file}: expected a string, got "
                        f"{type(entry).__name__} ({entry!r}). "
                        f"Where: {config_file} → pipeline.extraction.skip_labels[{i}]. "
                        f"Expected: a string label, e.g. 'junk'. "
                        f"How to fix: ensure all entries in 'pipeline.extraction.skip_labels' "
                        f"are strings in {config_file}."
                    ]
                )
        return list(value)
