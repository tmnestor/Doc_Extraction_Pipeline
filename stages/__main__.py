"""Run the task named by ``KFP_TASK``, with its settings worked out once.

This is what ``entrypoint.sh`` starts, for every task:

    KFP_TASK=classify model=<a registered model> image_dir=<folder> output=<folder> python -m stages

The settings come from ``common.stage_settings.resolve``: the environment first, the settings file second.
Here they are handed to the stage's own ``run``. Two things are done around a stage, the same for every
stage. A run carries on from where an earlier one stopped unless ``CLEAR_PREV_OUTPUT=true`` asks for the
stage's own earlier output to be deleted first: pages arrive over time, and a second run should read only
the new ones. And each stage that uses the cards notes the whole seconds it took, which the evaluate stage
adds up.
"""

import importlib
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from common.stage_settings import SettingsError, StageSettings, resolve

logger = logging.getLogger("stages")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "run_config.yml"

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
    for file in files:
        if file.is_file():
            file.unlink()
            logger.info("CLEAR_PREV_OUTPUT: removed %s", file)


def _seconds_noted(settings: StageSettings) -> int | None:
    """The whole seconds the stages that use the cards took, added up, or None if none was noted."""
    if not settings.inference_elapsed.is_file():
        logger.warning(
            "%s not found: the rate will be worked out from the pages' own times.",
            settings.inference_elapsed,
        )
        return None
    total = sum(int(line) for line in settings.inference_elapsed.read_text().split())
    logger.info("Time on the cards: %ds (from %s)", total, settings.inference_elapsed)
    return total


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
    started = time.time()
    if settings.task == "split":
        settings.split_dir.mkdir(parents=True, exist_ok=True)
        run(settings.image_dir, settings.split_dir, model_type=settings.model)
        logger.info(
            "Split complete. Later stages read their images from: %s", settings.split_dir / "images"
        )
    elif settings.task == "classify":
        _start_afresh(settings, settings.classifications, settings.inference_elapsed)
        run(settings.image_dir, settings.classifications, model_type=settings.model)
        elapsed = int(time.time() - started)
        settings.inference_elapsed.write_text(f"{elapsed}\n")
        logger.info("Classification complete (%ds).", elapsed)
    elif settings.task == "extract":
        _start_afresh(settings, settings.raw_extractions)
        run(
            settings.classifications,
            settings.image_dir,
            settings.raw_extractions,
            model_type=settings.model,
        )
        with settings.inference_elapsed.open("a") as noted:
            noted.write(f"{int(time.time() - started)}\n")
        logger.info("Extraction complete.")
    elif settings.task == "clean":
        _start_afresh(settings, settings.cleaned_extractions)
        run(settings.raw_extractions, settings.cleaned_extractions)
        logger.info("Cleaning complete.")
    elif settings.task == "evaluate":
        settings.evaluation_dir.mkdir(parents=True, exist_ok=True)
        _start_afresh(settings, settings.evaluation_dir / "evaluation_results.jsonl")
        run(
            settings.cleaned_extractions,
            settings.ground_truth,
            settings.evaluation_dir,
            inference_seconds=_seconds_noted(settings),
        )
        logger.info("Evaluation complete.")


def main() -> int:
    """Work out the settings of the task ``KFP_TASK`` names and run it."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        settings = resolve(os.environ.get("KFP_TASK", ""), os.environ, CONFIG_PATH)
    except SettingsError as err:
        print(f"FATAL: the task cannot be started.\n{err}")
        return 1
    run_task(settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
