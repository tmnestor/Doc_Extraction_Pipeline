"""What a task is run with: which model, which folders, and the choices of the run.

A task is started by ``entrypoint.sh`` with ``KFP_TASK`` naming it, and its settings are the job's
variables: the script's header lists them. The script has checked each before anything here runs:
that the task has the variables it needs, and that each value can be used. A variable that was unset,
empty or the word ``None`` is absent here, and the two that are true or false are written out.

Here they are read into one record, which the stages are handed. Until 26 November 2026 each was
looked for in the environment and then in a settings file, and checked here.

What is left in the settings file is where each model's weights are (``common.settings``).
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.settings import Settings, read_settings_file

TASKS = ("split", "classify", "extract", "clean", "evaluate")

_MEMORY_SHARE = "DOC_EXTRACT_GPU_MEMORY_UTILIZATION"


@dataclass(frozen=True)
class StageSettings:
    """Everything a task is run with."""

    task: str
    # None for the tasks that load no model and are given none: clean and evaluate.
    model: str | None
    image_dir: Path | None
    output: Path
    ground_truth: Path | None
    clear_previous_output: bool
    # The settings file: where each model's weights are.
    config_path: Path
    # Only pages whose file name holds one of these are read. None: every page.
    document_types: tuple[str, ...] | None = None
    # At most this many pages are read. None: every page.
    max_images: int | None = None
    # Whether the split task asks the model where each receipt is and cuts a photograph of several.
    photo_split: bool = False
    # The document type the job gives every page, in which case classify asks no model.
    given_type: str | None = None
    # The file that what is sent to the model is kept in. None: it is not kept.
    trace_file: Path | None = None
    # The share of each card's memory the job sets, or None for the model's own.
    gpu_memory_utilization: float | None = None
    # What the settings file holds, as it was read once when the task was worked out.
    raw: dict[str, Any] | None = field(default=None, repr=False, compare=False)

    # Every file a stage writes is under the output folder, so that one stage finds what another wrote.

    @property
    def split_dir(self) -> Path:
        return self.output / "split"

    @property
    def classifications(self) -> Path:
        return self.output / "classifications.jsonl"

    @property
    def raw_extractions(self) -> Path:
        return self.output / "raw_extractions.jsonl"

    @property
    def cleaned_extractions(self) -> Path:
        return self.output / "cleaned_extractions.jsonl"

    @property
    def evaluation_dir(self) -> Path:
        return self.output / "evaluation"


def run_settings(task: StageSettings) -> Settings:
    """The run's settings for a task that uses the cards, from what the task was worked out from.

    The settings file is not read again: the task holds what it held. The model's folder is looked
    for here, so this is asked only for a task that loads a model.

    Raises:
        ConfigError: With what is wrong, where, what is expected and how to mend it.
    """
    return Settings.load(
        {
            "model_type": task.model,
            "gpu_memory_utilization": task.gpu_memory_utilization,
            "document_types": task.document_types,
            "max_images": task.max_images,
            "trace_file": task.trace_file,
        },
        config_path=task.config_path,
        raw=task.raw,
    )


def resolve(task: str, environ: Mapping[str, str], config_path: Path) -> StageSettings:
    """What a task is run with, read from the job's variables as ``entrypoint.sh`` left them.

    Args:
        task: The task's name, as ``KFP_TASK`` gives it.
        environ: The job's environment.
        config_path: The settings file.

    Raises:
        ConfigError: If the settings file cannot be read.
    """

    def path(name: str) -> Path | None:
        return Path(environ[name]) if name in environ else None

    types = tuple(name.strip() for name in environ.get("document_types", "").split(",") if name.strip())
    return StageSettings(
        task=task,
        model=environ.get("model"),
        image_dir=path("image_dir"),
        output=Path(environ["output"]),
        ground_truth=path("ground_truth"),
        clear_previous_output=environ.get("CLEAR_PREV_OUTPUT") == "true",
        config_path=config_path,
        document_types=types or None,
        max_images=int(environ["max_images"]) if "max_images" in environ else None,
        photo_split=environ.get("photo_split") == "true",
        given_type=environ.get("document_type"),
        trace_file=path("DOC_EXTRACT_TRACE_FILE"),
        gpu_memory_utilization=float(environ[_MEMORY_SHARE]) if _MEMORY_SHARE in environ else None,
        raw=read_settings_file(config_path),
    )
