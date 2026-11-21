"""What every stage that reads pages needs first: the prompts and fields, the model, and the
document processor built on it.
"""

import contextlib
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from rich.console import Console

from common.field_schema import get_field_schema
from common.pipeline_config import PipelineConfig
from common.prompt_catalog import PromptCatalog
from models.registry import get_model

console = Console()
logger = logging.getLogger(__name__)


def load_prompt_config(model_type: str) -> dict[str, Any]:
    """Where the classification prompt is. The model's extraction prompts are checked on the way."""
    try:
        PromptCatalog().check_extraction_prompts(model_type)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]FATAL: {e}[/red]")
        raise SystemExit(1) from None
    return {
        "detection_file": str(Path(__file__).parent.parent / "prompts" / "document_type_detection.yaml"),
        "detection_key": "detection",
    }


def load_pipeline_configs(
    model_type: str,
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
        raise SystemExit(1) from None

    return prompt_config, universal_fields, field_definitions


def load_model(config: PipelineConfig, *, app_config: Any | None = None):
    """Context manager for loading and cleaning up model resources.

    Delegates to the registered loader for config.model_type, forwarding
    ``app_config`` so vLLM loaders can resolve per-model engine tuning from
    ``run_config.yml`` (mirrors ``create_processor``). HF loaders ignore it.
    """

    registration = get_model(config.model_type)
    return registration.loader(config, app_config=app_config)


@contextlib.contextmanager
def open_processor(app_config: Any) -> Iterator[Any]:
    """Load the run's model and give the document processor built on it; unload the model on the way out.

    Every stage that reads pages, and every worker that reads a share of them, gets its processor here, so
    the model is loaded and the processor built in one way only.

    Args:
        app_config: The run's settings; for a stage with its own engine settings, the copy made by
            ``for_stage``.

    Yields:
        The document processor.
    """
    config = app_config.pipeline
    prompt_config, universal_fields, field_definitions = load_pipeline_configs(config.model_type)
    logger.info("Loading model: %s", config.model_type)
    with load_model(config, app_config=app_config) as (model, tokenizer):
        yield create_processor(
            model,
            tokenizer,
            config,
            prompt_config,
            universal_fields,
            field_definitions,
            app_config=app_config,
        )


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
