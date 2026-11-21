"""Workers for an engine on each card.

``vllm_dp.run_dp`` starts one process for each card and calls one of these functions in it, by name, with
that card's share of the pages. A worker does three things: it loads the run's settings and the model on
its card, it calls the stage's own reading function on its share, and it hands the records back.

What is read, and how pages are grouped into engine calls, is the stage's business and is not repeated
here. One engine across the cards calls the same reading functions, so the two arrangements cannot come to
differ.

Everything passed in is plain data (text, numbers, lists, dicts), since it crosses a process boundary.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from common.settings import Settings
from stages.classify import classify_pages
from stages.extract import extract_pages
from stages.sroie import read_sroie

logger = logging.getLogger(__name__)


def _on_card(
    gpu_id: int,
    config_path: str | None,
    cli_overrides: dict[str, Any],
    read: Callable[[Any, Settings], list[dict[str, Any]]],
    *,
    stage: str | None = None,
) -> list[dict[str, Any]]:
    """Load the model on this card, read with it, and hand the records back marked with the card.

    Args:
        gpu_id: The card's number, for the log and for the records.
        config_path: Path to the settings file, or None for the shipped one.
        cli_overrides: The settings the stage was started with, over the file's.
        read: Given the document processor and the settings, returns this card's records.
        stage: A stage whose engine settings differ from the model's own, if this is one.

    Returns:
        The records ``read`` returned. Each carries the card's number, so that the parent can report the
        slowest card's time: the cards work at once, and adding their times would give the work done, not
        the time taken.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(levelname)s [GPU {gpu_id}] %(name)s: %(message)s",
    )
    from common.pipeline_ops import open_processor

    # This process sees one card only.
    app_config = Settings.load(
        {**cli_overrides, "num_gpus": 1}, config_path=Path(config_path) if config_path else None
    )
    if stage is not None:
        app_config = app_config.for_stage(stage)

    with open_processor(app_config) as processor:
        records = read(processor, app_config)
    for record in records:
        record["gpu_id"] = gpu_id
    return records


def classify_worker(
    gpu_id: int,
    image_paths: list[str],
    *,
    config_path: str | None,
    cli_overrides: dict[str, Any],
) -> list[dict[str, Any]]:
    """Classify this card's share of the pages."""
    pages = [Path(path) for path in image_paths]
    return _on_card(
        gpu_id,
        config_path,
        cli_overrides,
        lambda processor, app_config: classify_pages(processor, pages, app_config),
        stage="classify",
    )


def extract_worker(
    gpu_id: int,
    image_paths: list[str],
    *,
    config_path: str | None,
    cli_overrides: dict[str, Any],
    classifications: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Read the fields of this card's share of the pages.

    Every worker is given every classification record and keeps those of its own pages, in the order the
    stage put them in.
    """
    mine = set(image_paths)
    share = [record for record in classifications if record["image_path"] in mine]
    return _on_card(
        gpu_id,
        config_path,
        cli_overrides,
        lambda processor, app_config: extract_pages(processor, share, app_config),
    )


def sroie_worker(
    gpu_id: int,
    image_paths: list[str],
    *,
    config_path: str | None,
    cli_overrides: dict[str, Any],
) -> list[dict[str, Any]]:
    """Read this card's share of the SROIE benchmark."""
    return _on_card(
        gpu_id,
        config_path,
        cli_overrides,
        lambda processor, app_config: read_sroie(processor, image_paths, app_config),
    )
