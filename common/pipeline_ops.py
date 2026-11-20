"""Pipeline operations: model loading, processor creation, batch execution.

Extracted from cli.py so the stages can share the load/create/run seams
without importing the CLI.
"""

from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from common.field_schema import get_field_schema
from common.pipeline_config import PipelineConfig
from common.prompt_catalog import PromptCatalog
from models.registry import get_model

type BatchOutput = tuple[list[dict], list[float], dict[str, int], dict[str, float]]

console = Console()


def load_prompt_config(model_type: str = "internvl3-vllm") -> dict[str, Any]:
    """Build prompt routing config from PromptCatalog (single source of truth).

    Derives supported document types from the extraction prompt YAML keys
    rather than maintaining a hardcoded list.
    """
    catalog = PromptCatalog()

    try:
        routing = catalog.build_extraction_routing(model_type)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]FATAL: {e}[/red]")
        raise typer.Exit(1) from None

    # Build extraction_files in the legacy shape that orchestrator expects.
    # All extraction types map to the same YAML file.
    registration = get_model(model_type)
    extraction_path = str(Path(__file__).parent.parent / "prompts" / registration.prompt_file)
    detection_path = str(Path(__file__).parent.parent / "prompts" / "document_type_detection.yaml")

    return {
        "detection_file": detection_path,
        "detection_key": "detection",
        "extraction_files": {doc_type: extraction_path for doc_type in routing},
    }


def load_pipeline_configs(
    model_type: str = "internvl3-vllm",
) -> tuple[dict[str, Any], list[str], dict[str, list[str]]]:
    """Load prompt configuration and build universal field list.

    Returns:
        Tuple of (prompt_config, sorted universal_fields, field_definitions).
    """
    prompt_config = load_prompt_config(model_type)
    field_definitions = get_field_schema().get_all_doc_type_fields()

    all_fields: set[str] = set()
    for fields in field_definitions.values():
        all_fields.update(fields)
    universal_fields = sorted(all_fields)

    if not universal_fields:
        console.print("[red]FATAL: No field definitions found in config/field_definitions.yaml[/red]")
        console.print("[yellow]Expected: document_fields section with field lists[/yellow]")
        raise typer.Exit(1) from None

    return prompt_config, universal_fields, field_definitions


def load_model(config: PipelineConfig, *, app_config: Any | None = None):
    """Context manager for loading and cleaning up model resources.

    Delegates to the registered loader for config.model_type, forwarding
    ``app_config`` so vLLM loaders can resolve per-model engine tuning from
    ``run_config.yml`` (mirrors ``create_processor``). HF loaders ignore it.
    """
    from models.registry import get_model

    registration = get_model(config.model_type)
    return registration.loader(config, app_config=app_config)


def create_processor(
    model,
    tokenizer,
    config: PipelineConfig,
    prompt_config: dict[str, Any],
    universal_fields: list[str],
    field_definitions: dict[str, list[str]],
    *,
    app_config: Any | None = None,
) -> Any:
    """Create document extraction processor from loaded components.

    Delegates to the registered processor_creator for config.model_type.
    """
    from models.registry import get_model

    registration = get_model(config.model_type)
    return registration.processor_creator(
        model,
        tokenizer,
        config,
        prompt_config,
        universal_fields,
        field_definitions,
        app_config=app_config,
    )


def run_batch(
    config: PipelineConfig,
    processor: Any,
    images: list[Path],
    field_definitions: dict[str, list[str]],
) -> BatchOutput:
    """Run batch document processing with optional bank statement adapter."""
    from common.document_pipeline import create_document_pipeline
    from common.unified_bank_extractor import UnifiedBankExtractor

    bank_adapter = None
    if config.bank_v2 and getattr(processor, "supports_multi_turn", True):
        console.print("[bold cyan]Setting up sophisticated bank statement extraction...[/bold cyan]")

        bank_adapter = UnifiedBankExtractor(
            generate_fn=processor.generate,
            verbose=config.verbose,
            use_balance_correction=config.balance_correction,
            bank_strategy=config.bank_strategy,
        )

        console.print("[green]V2: Sophisticated bank statement extraction enabled[/green]")
        console.print(
            f"[dim]  Balance correction: {'Enabled' if config.balance_correction else 'Disabled'}[/dim]"
        )

    pipeline = create_document_pipeline(
        processor,
        ground_truth_csv=str(config.ground_truth) if config.ground_truth else None,
        bank_adapter=bank_adapter,
        field_definitions=field_definitions,
        batch_size=config.batch_size,
        enable_math_enhancement=False,
        console=console,
    )

    console.print(f"\n[bold]Processing {len(images)} images...[/bold]")

    batch_results, processing_times, document_types_found = pipeline.process_batch(
        [str(img) for img in images],
        verbose=config.verbose,
    )

    return (
        batch_results,
        processing_times,
        document_types_found,
        pipeline.batch_stats,
    )
