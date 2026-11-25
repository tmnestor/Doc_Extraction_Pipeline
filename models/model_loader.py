"""Starting a model's vLLM engine, and building the document processor on it.

Two functions, both given the model's specification (``models.registry``) and the run's settings:

- ``load_engine`` starts the engine and stops it on the way out;
- ``build_processor`` wraps a started engine in the backend and the document processor.

What the engine is started with is the model's own, in its registration, and three values that are the
same for every model, below.
"""

import copy
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from common.cards import cards_to_use
from common.stage_settings import StageSettings
from models.backends.vllm_backend import VllmBackend
from models.orchestrator import DocumentOrchestrator
from models.registry import VllmSpec


# The same for every model. False: the engine compiles CUDA graphs when it starts, a slower start and
# faster reading, which is the right trade for a run of more than a few pages.
ENFORCE_EAGER = False
# A page is one image.
IMAGES_PER_REQUEST = 1
# The engine reuses its work on the part of a prompt shared with the last.
PREFIX_CACHING = True


@contextmanager
def load_engine(spec: VllmSpec, settings: StageSettings) -> Iterator[Any]:
    """Start the model's vLLM engine, and let go of it on the way out.

    Args:
        spec: The model's specification.
        settings: The run's settings.

    Yields:
        The engine (``vllm.LLM``).
    """
    from rich.console import Console

    console = Console()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM

    engine = None
    try:
        # The engine is split across every card the process sees (common.cards). A worker's process
        # sees the one card it was given.
        cards = cards_to_use()

        # The share of each card's memory: the job's, else the model's own. The job's is read
        # and checked once, with the job's other variables (common.stage_settings), and comes here
        # with the settings: until 26 November 2026 it was read here from the environment.
        from_environment = settings.gpu_memory_utilization
        memory_share = float(from_environment or spec.gpu_memory_utilization)

        # max_num_seqs is printed because it is the ceiling on concurrency and there is otherwise no
        # way to tell from a run whether the configured value reached the engine: vLLM's own startup
        # lines are INFO, and entrypoint.sh suppresses those.
        console.print(
            f"\n[bold]Loading {spec.model_type} via vLLM "
            f"(tp={cards}, max_model_len={spec.max_model_len}, "
            f"max_num_seqs={spec.max_num_seqs})[/bold]"
        )
        console.print(f"[dim]Model path: {settings.model_path}[/dim]")
        console.print(
            f"[dim]enforce_eager: {ENFORCE_EAGER} "
            f"({'skip CUDA graphs' if ENFORCE_EAGER else 'CUDA graphs enabled'})[/dim]"
        )
        console.print(
            f"[dim]gpu_memory_utilization: {memory_share} "
            f"(from {'DOC_EXTRACT_GPU_MEMORY_UTILIZATION' if from_environment else 'the registration'})"
            f"[/dim]"
        )

        arguments: dict[str, Any] = {
            "model": str(settings.model_path),
            "tensor_parallel_size": cards,
            "max_model_len": spec.max_model_len,
            "gpu_memory_utilization": memory_share,
            "max_num_seqs": spec.max_num_seqs,
            "limit_mm_per_prompt": {"image": IMAGES_PER_REQUEST},
            "trust_remote_code": True,
            "disable_log_stats": True,
            "enforce_eager": ENFORCE_EAGER,
            "enable_prefix_caching": PREFIX_CACHING,
        }
        # The two blocks handed to the engine as the registration gives them, when it gives them.
        for block in ("mm_processor_kwargs", "hf_overrides"):
            if getattr(spec, block):
                arguments[block] = copy.deepcopy(getattr(spec, block))
        if spec.mm_processor_kwargs or spec.hf_overrides:
            console.print(
                f"[dim]mm_processor_kwargs: {arguments.get('mm_processor_kwargs', {})}; "
                f"hf_overrides: {arguments.get('hf_overrides', {})}[/dim]"
            )

        engine = LLM(**arguments)
        console.print("[bold green]vLLM engine ready![/bold green]")
        yield engine
    finally:
        del engine


def build_processor(spec: VllmSpec, engine: Any, settings: StageSettings) -> DocumentOrchestrator:
    """The document processor on a started engine.

    Args:
        spec: The model's specification: how its requests are formed.
        engine: The engine ``load_engine`` gave.
        settings: The run's settings.
    """
    backend = VllmBackend(
        engine=engine,
        model_type_key=spec.model_type,
        chat_template=spec.chat_template,
        chat_template_kwargs=spec.chat_template_kwargs,
        default_image_first=spec.default_image_first,
        trace_path=settings.trace_file,
    )
    return DocumentOrchestrator(backend, model_type_key=spec.model_type)
