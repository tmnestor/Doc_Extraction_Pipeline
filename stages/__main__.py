"""Run the task named by ``KFP_TASK``, with its settings worked out once.

This is what ``entrypoint.sh`` starts, for every task:

    KFP_TASK=classify model=<a registered model> image_dir=<folder> output=<folder> python -m stages

The settings come from ``common.stage_settings.resolve``: the environment first, the settings file second.
Here they are handed to the stage's own ``run``. Three things are done around a stage, the same for every
stage. A run carries on from where an earlier one stopped unless ``CLEAR_PREV_OUTPUT=true`` asks for the
stage's own earlier output to be deleted first: pages arrive over time, and a second run should read only
the new ones. A stage that carries on checks first that the earlier output is its own run's: the same
model, folder of pages and commit of the code, by the note the earlier run left in the output folder
(``common.run_manifest``). What a stage spent on the cards it notes itself (``common.stage_timing``).
"""

import importlib
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from common import run_manifest
from common.stage_settings import ConfigError, StageSettings, resolve
from stages.io import Journal

logger = logging.getLogger("stages")

REPOSITORY = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPOSITORY / "config" / "run_config.yml"

_BANNERS = {
    "split": "Stage 0: split, one image for each document in a photograph (GPU)",
    "classify": "Stage 1: classify, the document type of each page (GPU)",
    "extract": "Stage 2: extract, the fields of each page (GPU)",
    "clean": "Stage 3: clean, the model's replies parsed and tidied (CPU)",
    "evaluate": "Stage 4: evaluate, the fields scored against the answer key (CPU)",
    "sroie": "SROIE benchmark: 347 public receipts, four fields each (GPU)",
}


def _stage_run(task: str) -> Callable[..., Any]:
    """The stage's own ``run``. Imported only now, so that a task loads nothing another task needs."""
    return importlib.import_module(f"stages.{task}").run


def _start_afresh(settings: StageSettings, *files: Path) -> None:
    """Delete the stage's own earlier output, if that was asked for. Never a folder, never a log."""
    if not settings.clear_previous_output:
        return
    for output in files:
        # With an output goes its journal: what an interrupted run had read and not yet written.
        for file in (output, Journal(output).path):
            if file.is_file():
                file.unlink()
                logger.info("CLEAR_PREV_OUTPUT: removed %s", file)


def _check_what_made_the_folder(settings: StageSettings) -> None:
    """A stage that carries on from its earlier output checks that it is the same run, and notes itself.

    The other stages say what the notes in the folder hold, so that a score carries the model and the
    commit that produced it.

    Raises:
        ConfigError: If the stage's earlier output is from a run with another model, folder of pages or
            commit of the code (``common.run_manifest``).
    """
    if settings.task not in run_manifest.STAGES:
        for line in run_manifest.describe(settings.output):
            logger.info("  made by %s", line)
        return
    assert settings.image_dir is not None
    run_manifest.check_and_note(
        settings.output,
        settings.task,
        model=settings.model,
        image_dir=settings.image_dir,
        repository=REPOSITORY,
        # The split stage has no earlier output deleted for it, so it always checks.
        starting_afresh=settings.clear_previous_output and settings.task != "split",
    )
    if settings.task == "extract":
        run_manifest.warn_if_made_another_way(settings.output, "extract", after="classify")


def run_task(settings: StageSettings) -> None:
    """Run one task with its settings.

    Args:
        settings: What the task is run with.
    """
    run = _stage_run(settings.task)
    logger.info("=================================================================")
    logger.info("    %s", _BANNERS[settings.task])
    logger.info("=================================================================")
    for name in ("model", "image_dir", "output", "ground_truth", "clear_previous_output"):
        logger.info("  %-22s %s", name + ":", getattr(settings, name))

    if settings.task == "sroie":
        # The benchmark reads its folders from pipeline.sroie in the settings file, writes over its two
        # files every time, and does not resume.
        if settings.sroie_max_images is not None:
            logger.info(
                "  sroie_max_images: %d (a part run: not a score to publish)", settings.sroie_max_images
            )
        run(model_type=settings.model, max_images=settings.sroie_max_images)
        return

    settings.output.mkdir(parents=True, exist_ok=True)
    _check_what_made_the_folder(settings)
    started = time.time()
    if settings.task == "split":
        settings.split_dir.mkdir(parents=True, exist_ok=True)
        run(settings.image_dir, settings.split_dir, model_type=settings.model)
        logger.info(
            "Split complete. Later stages read their images from: %s", settings.split_dir / "images"
        )
    elif settings.task == "classify":
        _start_afresh(settings, settings.classifications)
        run(settings.image_dir, settings.classifications, model_type=settings.model)
        logger.info("Classification complete (%ds).", int(time.time() - started))
    elif settings.task == "extract":
        _start_afresh(settings, settings.raw_extractions)
        run(
            settings.classifications,
            settings.image_dir,
            settings.raw_extractions,
            model_type=settings.model,
        )
        logger.info("Extraction complete (%ds).", int(time.time() - started))
    elif settings.task == "clean":
        _start_afresh(settings, settings.cleaned_extractions)
        run(settings.raw_extractions, settings.cleaned_extractions)
        logger.info("Cleaning complete.")
    elif settings.task == "evaluate":
        settings.evaluation_dir.mkdir(parents=True, exist_ok=True)
        _start_afresh(settings, settings.evaluation_dir / "evaluation_results.jsonl")
        run(settings.cleaned_extractions, settings.ground_truth, settings.evaluation_dir)
        logger.info("Evaluation complete.")


def main() -> int:
    """Work out the settings of the task ``KFP_TASK`` names and run it."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        run_task(resolve(os.environ.get("KFP_TASK", ""), os.environ, CONFIG_PATH))
    except ConfigError as err:
        print(f"FATAL: the task cannot be started.\n{err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
