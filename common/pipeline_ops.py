"""What every stage that reads pages needs first: the model started, and the document processor on it."""

import contextlib
import logging
from collections.abc import Iterator

from rich.console import Console

from common.prompt_catalog import PromptCatalog
from common.settings import Settings
from models.model_loader import build_processor, load_engine
from models.orchestrator import DocumentOrchestrator
from models.registry import get_model

console = Console()
logger = logging.getLogger(__name__)


def check_prompts(model_type: str) -> None:
    """Stop the run unless the model has an extraction prompt for every document type."""
    try:
        PromptCatalog().check_extraction_prompts(model_type)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]FATAL: {e}[/red]")
        raise SystemExit(1) from None


@contextlib.contextmanager
def open_processor(settings: Settings) -> Iterator[DocumentOrchestrator]:
    """Start the run's model and give the document processor built on it; stop the model on the way out.

    Every stage that reads pages, and every worker that reads a share of them, gets its processor here, so
    the model is started and the processor built in one way only.

    Args:
        settings: The run's settings; for a stage with its own engine settings, the copy made by
            ``for_stage``.

    Yields:
        The document processor.
    """
    check_prompts(settings.model_type)
    spec = get_model(settings.model_type)
    logger.info("Loading model: %s", settings.model_type)
    with load_engine(spec, settings) as engine:
        yield build_processor(spec, engine, settings)
