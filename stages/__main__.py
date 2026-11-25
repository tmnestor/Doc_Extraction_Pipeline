"""Run the task named by ``KFP_TASK``, with its settings worked out once.

This is what ``entrypoint.sh`` starts, for every task:

    KFP_TASK=classify model=<a registered model> image_dir=<folder> output=<folder> python -m stages

The settings are the job's variables, which ``entrypoint.sh`` has checked, read into one record
(``common.stage_settings.resolve``). Here they are handed to the stage's own ``run``. Three things are done around a stage, the same for every
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
from pathlib import Path
from typing import Any

from common import run_manifest
from common.engine_calls import StageStopped
from common.errors import ConfigError
from common.stage_settings import StageSettings, resolve


logger = logging.getLogger("stages")

REPOSITORY = Path(__file__).resolve().parent.parent


def _stage(task: str) -> Any:
    """The stage's own module. Imported only now, so that a task loads nothing another task needs.

    A stage is a module of ``stages`` with a ``BANNER``, the line that names it in the log, a
    ``check`` that is given the task's settings, reads everything the stage will use, writes nothing
    and gives what it worked out, and a ``start`` that is given the task's settings and what
    ``check`` gave, runs the stage and returns the line that says it is done. Nothing a stage reads
    to be checked is read or worked out again to be used. What a stage reads and writes, and which of its files a run started afresh deletes, is the
    stage's own to know. Until 25 November 2026 this module knew it for every stage, in a branch each.
    """
    return importlib.import_module(f"stages.{task}")


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
    settings.output.mkdir(parents=True, exist_ok=True)
    run_manifest.check_and_note(
        settings.output,
        settings.task,
        model=settings.model,
        image_dir=settings.image_dir,
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
    stage = _stage(settings.task)
    logger.info("=================================================================")
    logger.info("    %s", stage.BANNER)
    logger.info("=================================================================")

    # In this order. The task is checked whole before anything is written: the note of what made the
    # folder is one thing written, and a stage started afresh deletes its earlier output. Until
    # 26 November 2026 the note came first, and the stage read its own settings only once started.
    # A task that could not start then left a note in a folder that held none of its output, and the
    # next start with another model was refused as "output from another run"; started afresh, it had
    # deleted the earlier output before it found that it could not replace it.
    checked = stage.check(settings)
    _check_what_made_the_folder(settings)
    logger.info(stage.start(settings, checked))


def main() -> int:
    """Work out the settings of the task ``KFP_TASK`` names and run it."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        run_task(resolve(os.environ["KFP_TASK"], os.environ))
    except ConfigError as err:
        print(f"FATAL: the task cannot be started.\n{err}")
        return 1
    except StageStopped as err:
        print(f"FATAL: the task was stopped part way.\n{err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
