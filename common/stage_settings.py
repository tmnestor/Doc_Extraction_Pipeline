"""What a task is run with: which model, which folders, and the choices of the run.

A task is started by ``entrypoint.sh`` with ``KFP_TASK`` naming it, and its settings are the job's
variables: the script's header lists them. The script has checked each before anything here runs:
that the task has the variables it needs, and that each value can be used. A variable that was unset,
empty or the word ``None`` is absent here, and the two that are true or false are written out.

Here they are read into one record, which is the whole of a run's settings: the stages are handed it,
and so is the engine's loader. It is frozen, and crosses to a worker's process as it is. Until
26 November 2026 each setting was looked for in the environment and then in a settings file, and a
second object was built from that file for the engine.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from models.registry import get_model

TASKS = ("split", "classify", "extract", "clean", "evaluate")

_MEMORY_SHARE = "DOC_EXTRACT_GPU_MEMORY_UTILIZATION"


@dataclass(frozen=True)
class StageSettings:
    """Everything a task is run with."""

    task: str
    # Empty for the tasks that load no model and are given none: clean and evaluate.
    model: str
    image_dir: Path | None
    output: Path
    ground_truth: Path | None
    clear_previous_output: bool
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
    # The folder that holds each model's folder of weights (DOC_EXTRACT_MODELS_DIR).
    models_dir: Path | None = None

    @property
    def model_path(self) -> Path:
        """The folder of the model's weights: the one its registration names, under ``models_dir``."""
        assert self.models_dir is not None
        return self.models_dir / get_model(self.model).weights_folder

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


def resolve(task: str, environ: Mapping[str, str]) -> StageSettings:
    """What a task is run with, read from the job's variables as ``entrypoint.sh`` left them.

    Args:
        task: The task's name, as ``KFP_TASK`` gives it.
        environ: The job's environment.
    """

    def path(name: str) -> Path | None:
        return Path(environ[name]) if name in environ else None

    types = tuple(name.strip() for name in environ.get("document_types", "").split(",") if name.strip())
    return StageSettings(
        task=task,
        model=environ.get("model", ""),
        image_dir=path("image_dir"),
        output=Path(environ["output"]),
        ground_truth=path("ground_truth"),
        clear_previous_output=environ.get("CLEAR_PREV_OUTPUT") == "true",
        document_types=types or None,
        max_images=int(environ["max_images"]) if "max_images" in environ else None,
        photo_split=environ.get("photo_split") == "true",
        given_type=environ.get("document_type"),
        trace_file=path("DOC_EXTRACT_TRACE_FILE"),
        gpu_memory_utilization=float(environ[_MEMORY_SHARE]) if _MEMORY_SHARE in environ else None,
        models_dir=path("DOC_EXTRACT_MODELS_DIR"),
    )
