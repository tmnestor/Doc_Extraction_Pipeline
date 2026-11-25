"""What every stage that reads pages needs first: the model started, and the document processor on it."""

import contextlib
import logging
from collections.abc import Iterator

import yaml

from common.errors import ConfigError
from common.prompt_catalog import PromptCatalog
from common.stage_settings import StageSettings
from models.model_loader import build_processor, load_engine
from models.orchestrator import DocumentOrchestrator
from models.registry import get_model

logger = logging.getLogger(__name__)


def check_processor(settings: StageSettings) -> None:
    """Everything a document processor needs that can be checked without an engine.

    Called before any engine is started (``common.vllm_dp.read_on_the_cards``), and again by
    ``open_processor`` for a caller that starts an engine itself.

    Raises:
        ConfigError: With what is wrong, where, what is expected and how to mend it.
    """
    check_prompts(settings.model)
    if not (settings.model_path / "config.json").is_file():
        raise ConfigError(
            f"What: the weights of the model {settings.model!r} were not found: "
            f"{settings.model_path} does not hold a config.json.\n"
            f"Where: the variable DOC_EXTRACT_MODELS_DIR in the environment of this job, which names "
            f"the folder that holds each model's folder. The model's own folder is weights_folder in "
            f"its registration (models/registry.py).\n"
            f"Expected: {settings.model_path}/config.json, with the model's weights beside it.\n"
            f"How to fix: point DOC_EXTRACT_MODELS_DIR at the folder that holds "
            f"{settings.model_path.name}."
        )


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
def open_processor(settings: StageSettings) -> Iterator[DocumentOrchestrator]:
    """Start the run's model and give the document processor built on it; stop the model on the way out.

    Every stage that reads pages, and every worker that reads a share of them, gets its processor here, so
    the model is started and the processor built in one way only.

    Args:
        settings: The run's settings.

    Yields:
        The document processor.
    """
    check_processor(settings)
    spec = get_model(settings.model)
    logger.info("Loading model: %s", settings.model)
    with load_engine(spec, settings) as engine:
        yield build_processor(spec, engine, settings)
