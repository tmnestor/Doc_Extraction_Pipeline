"""Starting a model's vLLM engine, and building the document processor on it.

Two functions, both given the model's specification (``models.registry``) and the run's settings:

- ``load_engine`` starts the engine and stops it on the way out;
- ``build_processor`` wraps a started engine in the backend and the document processor.

Every engine setting comes from ``config/run_config.yml`` (``settings.get_vllm_config``). Nothing is
defaulted here.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from common.cards import cards_to_use
from common.prompt_trace import effective_trace_path
from common.settings import Settings
from models.backends.vllm_backend import VllmBackend
from models.orchestrator import DocumentOrchestrator
from models.registry import VllmSpec


@contextmanager
def load_engine(spec: VllmSpec, settings: Settings) -> Iterator[Any]:
    """Start the model's vLLM engine, and let go of it on the way out.

    Args:
        spec: The model's specification.
        settings: The run's settings; for a stage with its own engine settings, the copy made by
            ``for_stage``.

    Yields:
        The engine (``vllm.LLM``).
    """
    from rich.console import Console

    console = Console()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM

    engine = None
    try:
        # The engine is split across every card the run has (common.cards). A worker that has one card
        # to itself is handed settings that say so.
        cards = cards_to_use(settings)
        tuning = settings.get_vllm_config(spec.model_type)

        # The share of each card's memory: the job's, else the settings file's. The job's is read
        # and checked once, with the job's other variables (common.stage_settings), and comes here
        # with the settings: until 26 November 2026 it was read here from the environment.
        from_environment = settings.gpu_memory_utilization
        memory_share = float(from_environment or tuning["gpu_memory_utilization"])

        # max_num_seqs is printed because it is the ceiling on concurrency and there is otherwise no
        # way to tell from a run whether the configured value reached the engine: vLLM's own startup
        # lines are INFO, and entrypoint.sh suppresses those.
        console.print(
            f"\n[bold]Loading {spec.model_type} via vLLM "
            f"(tp={cards}, max_model_len={tuning['max_model_len']}, "
            f"max_num_seqs={tuning['max_num_seqs']})[/bold]"
        )
        console.print(f"[dim]Model path: {settings.model_path}[/dim]")
        console.print(
            f"[dim]enforce_eager: {settings.enforce_eager} "
            f"({'skip CUDA graphs' if settings.enforce_eager else 'CUDA graphs enabled'})[/dim]"
        )
        console.print(
            f"[dim]gpu_memory_utilization: {memory_share} "
            f"(from {'DOC_EXTRACT_GPU_MEMORY_UTILIZATION' if from_environment else 'the settings file'})"
            f"[/dim]"
        )

        arguments: dict[str, Any] = {
            "model": str(settings.model_path),
            "tensor_parallel_size": cards,
            "max_model_len": tuning["max_model_len"],
            "gpu_memory_utilization": memory_share,
            "max_num_seqs": tuning["max_num_seqs"],
            "limit_mm_per_prompt": {"image": tuning["limit_mm_per_prompt"]},
            "trust_remote_code": True,
            "disable_log_stats": True,
            "enforce_eager": settings.enforce_eager,
            "enable_prefix_caching": tuning["enable_prefix_caching"],
        }
        # The two blocks handed to the engine as the settings file gives them, when it gives them.
        for block in ("mm_processor_kwargs", "hf_overrides"):
            if tuning.get(block):
                arguments[block] = dict(tuning[block])
        if tuning.get("mm_processor_kwargs") or tuning.get("hf_overrides"):
            console.print(
                f"[dim]mm_processor_kwargs: {arguments.get('mm_processor_kwargs', {})}; "
                f"hf_overrides: {arguments.get('hf_overrides', {})}[/dim]"
            )

        engine = LLM(**arguments)
        console.print("[bold green]vLLM engine ready![/bold green]")
        yield engine
    finally:
        del engine


def build_processor(spec: VllmSpec, engine: Any, settings: Settings) -> DocumentOrchestrator:
    """The document processor on a started engine.

    Args:
        spec: The model's specification: how its requests are formed.
        engine: The engine ``load_engine`` gave.
        settings: The run's settings.
    """
    backend = VllmBackend(
        engine=engine,
        model_type_key=spec.model_type,
        chat_template=settings.chat_template,
        chat_template_kwargs=spec.chat_template_kwargs,
        default_image_first=spec.default_image_first,
        trace_path=effective_trace_path(settings),
    )
    return DocumentOrchestrator(backend, model_type_key=spec.model_type, app_config=settings)
