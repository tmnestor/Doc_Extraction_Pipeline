"""vLLM model loader factory.

Provides the ``VllmSpec`` dataclass and factory functions that turn an ~8-line
declarative registration into a vLLM engine loader + processor creator wired
into a ``DocumentOrchestrator``.

Usage:
    register_vllm_model(VllmSpec(
        model_type="internvl3-vllm",
        prompt_file="extraction_prompts.yaml",
    ))
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from common.prompt_trace import effective_trace_path
from models.backends.vllm_backend import VllmBackend
from models.orchestrator import DocumentOrchestrator


@dataclass(frozen=True)
class VllmSpec:
    """Declarative specification for a vLLM model.

    Holds only *structural* fields. Engine tuning (gpu_memory_utilization,
    max_model_len, max_num_seqs, limit_mm_per_prompt, enable_prefix_caching)
    lives in ``config/run_config.yml`` under ``inference.vllm:`` — the YAML is the
    single source of truth and is resolved at load time via
    ``app_config.get_vllm_config(model_type)``.
    """

    model_type: str
    prompt_file: str = "extraction_prompts.yaml"
    description: str = ""
    attention_backend: str | None = None  # None = vLLM auto-selects
    mm_processor_kwargs: dict[str, Any] = field(default_factory=dict)
    hf_overrides: dict[str, Any] = field(default_factory=dict)
    # Forwarded to every engine.chat() call. Suppresses a thinking-by-default
    # chat template (e.g. Gemma 4: {"enable_thinking": False}). NOTE: this does
    # nothing on InternVL3.5, where thinking is driven by the system prompt, not
    # an enable_thinking flag — use bootstrap.model.chat_template there instead.
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)

    # -- Capabilities: properties of the MODEL, so they live in code, not YAML --
    # (Operator *intent* stays in run_config.yml; what a checkpoint can physically
    # do is structural, same rationale as attention_backend above.)

    # False for models too large to hold one whole engine per GPU — the DP fast
    # path would start N independent engines and OOM.
    supports_data_parallel: bool = True
    # Default content order for generate(). Gemma 4's card recommends image
    # before text; InternVL keeps text-first. An explicit extra["image_first"]
    # always wins over this default.
    default_image_first: bool = False


def _resolve_vllm_overrides(app_config: Any | None, model_type: str) -> dict[str, Any]:
    """Resolve per-model vLLM engine tuning from the runtime AppConfig.

    The YAML is the single source of truth — there is no spec-level or Python
    fallback, so an AppConfig is mandatory for vLLM loads.

    Raises:
        ValueError: If *app_config* is None, with a four-element diagnostic.
    """
    if app_config is None:
        raise ValueError(
            f"vLLM model load requires an AppConfig, but none was provided for "
            f"model_type={model_type!r}.\n"
            "  What:  load_model() was called without app_config= for a vLLM backend; "
            "per-model engine tuning has no Python fallback.\n"
            "  Where: the caller of common.pipeline_ops.load_model (a stage, worker, or "
            "benchmark script), plus the `vllm:` section of config/run_config.yml.\n"
            "  Example: load_model(config, app_config=app_cfg)  # app_cfg = AppConfig.load(...)\n"
            "  Fix:   thread the AppConfig already in scope into load_model; scripts that "
            "call registration.loader() directly must pass loader(cfg, app_config=app_cfg)."
        ) from None
    return app_config.get_vllm_config(model_type)


def build_vllm_loader(spec: VllmSpec):
    """Build a context-manager vLLM loader from a VllmSpec.

    The returned loader accepts ``app_config`` at call time (mirroring
    ``create_processor``); per-model engine tuning is sourced from
    ``app_config.get_vllm_config(spec.model_type)``.
    """

    def _loader_fn(config, *, app_config: Any | None = None):
        from rich.console import Console

        console = Console()

        @contextmanager
        def _loader(cfg):
            import os

            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

            from vllm import LLM

            llm = None

            try:
                # Detect GPU count: config.num_gpus > CUDA_VISIBLE_DEVICES > nvidia-smi
                if cfg.num_gpus and cfg.num_gpus > 0:
                    tp_size = cfg.num_gpus
                else:
                    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                    if cuda_visible is not None:
                        tp_size = len(cuda_visible.split(","))
                    else:
                        import subprocess

                        result = subprocess.run(
                            ["nvidia-smi", "-L"],
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        tp_size = len(result.stdout.strip().splitlines()) if result.returncode == 0 else 1
                tp_size = max(1, tp_size)

                # All engine tuning comes from app_config (YAML single source of
                # truth). No spec-level fallback — a vLLM model MUST be loaded
                # with an AppConfig. Keys are required, not optional, per the
                # "every config key is required" rule.
                vllm_overrides = _resolve_vllm_overrides(app_config, spec.model_type)

                # gpu_memory_utilization precedence: env var > YAML.
                gpu_mem_util = float(
                    os.environ.get(
                        "DOC_EXTRACT_GPU_MEMORY_UTILIZATION",
                        vllm_overrides["gpu_memory_utilization"],
                    )
                )
                effective_max_model_len = vllm_overrides["max_model_len"]
                effective_max_num_seqs = vllm_overrides["max_num_seqs"]
                effective_enable_prefix_caching = vllm_overrides["enable_prefix_caching"]
                effective_limit_mm = vllm_overrides["limit_mm_per_prompt"]

                # max_num_seqs is printed because it is the ceiling on
                # concurrency and there is otherwise NO way to tell from a run
                # whether the configured value reached the engine — vLLM's own
                # startup lines are INFO, and entrypoint.sh suppresses those.
                console.print(
                    f"\n[bold]Loading {spec.model_type} via vLLM "
                    f"(tp={tp_size}, max_model_len={effective_max_model_len}, "
                    f"max_num_seqs={effective_max_num_seqs})[/bold]"
                )
                console.print(f"[dim]Model path: {cfg.model_path}[/dim]")
                console.print(
                    f"[dim]enforce_eager: {cfg.enforce_eager} "
                    f"({'skip CUDA graphs' if cfg.enforce_eager else 'CUDA graphs enabled'})[/dim]"
                )
                if os.environ.get("DOC_EXTRACT_GPU_MEMORY_UTILIZATION"):
                    console.print(
                        f"[dim]gpu_memory_utilization: {gpu_mem_util} "
                        f"(from DOC_EXTRACT_GPU_MEMORY_UTILIZATION env var)[/dim]"
                    )
                elif vllm_overrides.get("gpu_memory_utilization") is not None:
                    console.print(f"[dim]gpu_memory_utilization: {gpu_mem_util} (from app_config)[/dim]")

                llm_kwargs: dict[str, Any] = {
                    "model": str(cfg.model_path),
                    "tensor_parallel_size": tp_size,
                    "max_model_len": effective_max_model_len,
                    "gpu_memory_utilization": gpu_mem_util,
                    "max_num_seqs": effective_max_num_seqs,
                    "limit_mm_per_prompt": {"image": effective_limit_mm},
                    "trust_remote_code": True,
                    "disable_log_stats": True,
                    "enforce_eager": cfg.enforce_eager,
                    "enable_prefix_caching": effective_enable_prefix_caching,
                }

                if spec.attention_backend is not None:
                    llm_kwargs["attention_backend"] = spec.attention_backend

                # mm_processor_kwargs: spec defaults, then app_config overlay
                effective_mm_proc = dict(spec.mm_processor_kwargs) if spec.mm_processor_kwargs else {}
                if vllm_overrides.get("mm_processor_kwargs"):
                    effective_mm_proc.update(vllm_overrides["mm_processor_kwargs"])
                if effective_mm_proc:
                    llm_kwargs["mm_processor_kwargs"] = effective_mm_proc

                # hf_overrides: spec defaults, then app_config (run_config) overlay.
                # InternVL tile count (max_dynamic_patch) must be set HERE, by
                # overriding the HF config field — NOT via mm_processor_kwargs,
                # which vLLM forwards to the video processor constructor (which
                # rejects it: InternVLVideoProcessor has no max_dynamic_patch).
                effective_hf = dict(spec.hf_overrides) if spec.hf_overrides else {}
                if vllm_overrides.get("hf_overrides"):
                    effective_hf.update(vllm_overrides["hf_overrides"])
                if effective_hf:
                    llm_kwargs["hf_overrides"] = effective_hf
                if effective_mm_proc or effective_hf:
                    console.print(
                        f"[dim]mm_processor_kwargs: {effective_mm_proc}; hf_overrides: {effective_hf}[/dim]"
                    )

                llm = LLM(**llm_kwargs)

                console.print("[bold green]vLLM engine ready![/bold green]")

                yield llm, None

            finally:
                del llm

        return _loader(config)

    return _loader_fn


def build_vllm_processor_creator(spec: VllmSpec):
    """Build a processor_creator for a vLLM model."""

    def _creator(
        model,
        tokenizer_or_processor,
        config,
        prompt_config,
        universal_fields,
        field_definitions,
        *,
        app_config=None,
    ):
        # debug=config.debug → Tier C; verbose=config.verbose → Tier B. See
        # `plans/quiet-pipeline-output.md` for the split.
        backend = VllmBackend(
            engine=model,
            model_type_key=spec.model_type,
            chat_template=config.chat_template,
            chat_template_kwargs=spec.chat_template_kwargs,
            default_image_first=spec.default_image_first,
            trace_path=effective_trace_path(config),
            debug=config.debug,
        )

        return DocumentOrchestrator(
            backend=backend,
            field_list=universal_fields,
            prompt_config=prompt_config,
            field_definitions=field_definitions,
            debug=config.debug,
            verbose=config.verbose,
            device=str(config.device_map),
            batch_size=config.batch_size,
            model_type_key=spec.model_type,
            app_config=app_config,
            has_oom_recovery=False,
        )

    return _creator


# Registered specs, keyed by model_type. ``register_model`` stores loaders and
# processor creators built from a spec via closures, so without this map a spec's
# declared capabilities — chat_template_kwargs,
# default_image_first — are unreachable once registration has happened. That
# matters because those capabilities fail QUIETLY: a model that reasons when it
# should not still returns a parseable response.
_VLLM_SPECS: dict[str, VllmSpec] = {}


def register_vllm_model(spec: VllmSpec) -> None:
    """Register a vLLM model with the registry.

    Per-model engine tuning is resolved from the runtime AppConfig at load
    time (see ``build_vllm_loader``), so no config is captured at registration.

    Args:
        spec: Declarative vLLM model specification.
    """
    from models.registry import ModelRegistration, register_model

    _VLLM_SPECS[spec.model_type] = spec

    register_model(
        ModelRegistration(
            model_type=spec.model_type,
            loader=build_vllm_loader(spec),
            processor_creator=build_vllm_processor_creator(spec),
            prompt_file=spec.prompt_file,
            description=spec.description,
            requires_sharding=True,
            is_vllm=True,
            supports_data_parallel=spec.supports_data_parallel,
        )
    )
