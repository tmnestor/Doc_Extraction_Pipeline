"""What every stage that reads pages needs first: the model started, and the document processor on it."""

import contextlib
import logging
from collections.abc import Iterator

import yaml

from common.config_schema import ConfigError
from common.prompt_catalog import PromptCatalog
from common.settings import Settings
from models.model_loader import build_processor, load_engine
from models.orchestrator import DocumentOrchestrator
from models.registry import get_model

logger = logging.getLogger(__name__)


def check_processor(settings: Settings) -> None:
    """Everything a document processor needs that can be checked without an engine.

    Called before any engine is started (``common.vllm_dp.read_on_the_cards``), and again by
    ``open_processor`` for a caller that starts an engine itself.

    Raises:
        ConfigError: With what is wrong, where, what is expected and how to mend it.
    """
    check_prompts(settings.model_type)


def check_prompts(model_type: str) -> None:
    """Refuse a model that is not registered, or has not one extraction prompt for each document type.

    Raises:
        ConfigError: With what is wrong, where, what is expected and how to mend it.
    """
    try:
        PromptCatalog().check_extraction_prompts(model_type)
    except (OSError, yaml.YAMLError) as err:
        raise ConfigError(
            f"What: the prompt file of the model {model_type!r} cannot be read: {err}\n"
            f"Where: the folder prompts/ of the repository, the file the model's registration names "
            f"(models/registry.py, prompt_file).\n"
            f"Expected: a YAML file with one prompt under prompts: for each document type.\n"
            f"How to fix: restore the file from the repository, or correct what the message names."
        ) from err


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
    check_processor(settings)
    spec = get_model(settings.model_type)
    logger.info("Loading model: %s", settings.model_type)
    with load_engine(spec, settings) as engine:
        yield build_processor(spec, engine, settings)
