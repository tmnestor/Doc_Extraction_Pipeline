"""Stage 1: Document type classification (GPU).

Detects document types for all images and writes classifications.jsonl.

Started by ``python -m stages`` with ``KFP_TASK`` naming it:

    KFP_TASK=classify model=<a registered model> image_dir=<folder> output=<folder> bash entrypoint.sh
"""

import logging
import time
from pathlib import Path
from typing import Any


from common.engine_calls import groups_of, in_engine_calls
from common.extraction_batching import read_classify_batch_size

from .io import append_jsonl, read_completed_images, write_jsonl

logger = logging.getLogger(__name__)


def _filter_already_classified(images: list[Path], output_path: Path) -> tuple[list[Path], int]:
    """Drop images already present in *output_path* (resume support).

    Mirrors the extract stage: a prior (possibly crashed) run's partial
    classifications.jsonl is read for its ``image_name`` set, and those images
    are skipped. Returns ``(remaining_images, completed_count)``; when the output
    is absent or empty, returns the input unchanged with a count of 0.
    """
    completed = read_completed_images(output_path)
    if not completed:
        return images, 0
    remaining = [img for img in images if img.name not in completed]
    return remaining, len(completed)


def run(
    image_dir: Path,
    output_path: Path,
    *,
    model_type: str | None = None,
    config_path: Path | None = None,
) -> Path:
    """Detect document types for all images, write classifications.jsonl.

    Args:
        image_dir: Directory containing images.
        output_path: Path to write classifications.jsonl.
        model_type: A registered model.
        config_path: Optional path to run_config.yml.

    Returns:
        Path to the written classifications.jsonl.
    """
    from common.settings import Settings
    from stages.io import discover_images

    cli_args: dict[str, Any] = {
        "data_dir": str(image_dir),
        "output_dir": str(output_path.parent),
    }
    if model_type is not None:
        cli_args["model_type"] = model_type

    app_cfg = Settings.load(cli_args, config_path=config_path).for_stage("classify")
    config = app_cfg
    # Discover images
    # Both filters are config-driven and MUST be passed: calling with the
    # directory alone silently ignored max_images and document_types, so a
    # 12-image request processed all 165 (fixed 2026-09-21).
    images = list(
        discover_images(
            config.data_dir,
            document_types=config.document_types,
            max_images=config.max_images,
        )
    )
    if not images:
        msg = f"No images found in {config.data_dir}"
        raise FileNotFoundError(msg)

    logger.info("Found %d images in %s", len(images), config.data_dir)

    # Resume: skip images already classified in a prior (possibly crashed) run,
    # and APPEND below instead of truncating. A clean-slate re-run is the opt-in
    # (clear the output first: the entrypoint does this for full re-runs). This
    # mirrors the extract stage and runs BEFORE the model load, so a fully
    # resumed run skips the GPU entirely.
    original_total = len(images)
    images, completed_count = _filter_already_classified(images, output_path)
    resuming = completed_count > 0
    if resuming:
        logger.info(
            "Resuming: %d already done, %d remaining (of %d total)",
            completed_count,
            len(images),
            original_total,
        )
    if not images:
        logger.info("All images already classified -- nothing to do")
        return output_path

    # One engine on each card where the model allows it, else one engine across the cards. Either way the
    # pages are read by classify_pages; only where the engine is started differs.
    from common.stage_timing import inference_seconds_from_records
    from common.vllm_dp import read_on_the_cards

    started = time.time()
    records = read_on_the_cards(app_cfg, images, classify_pages).records

    elapsed = inference_seconds_from_records(records, fallback=time.time() - started)
    logger.info(
        "Classification complete: %d images in %.1fs (%.2fs/image)",
        len(records),
        elapsed,
        elapsed / len(records) if records else 0,
    )

    # Write output (append when resuming so prior records are preserved)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_fn = append_jsonl if resuming else write_jsonl
    count = write_fn(output_path, records)
    logger.info("Wrote %d classifications to %s", count, output_path)

    return output_path


def classify_pages(processor: Any, images: list[Path], app_config: Any) -> list[dict[str, Any]]:
    """Give each page its document type. The one place pages are classified, however engines are arranged.

    One engine across the cards calls this once with every page. With an engine on each card, each worker
    calls it with its share. It loads no model, opens no file and decides nothing about which pages are
    already done: the stage's ``run`` does those.

    Each record carries a processing time. Inside a shared engine call the requests overlap, so a page's
    own time is not separable: each page is charged an equal share of the whole.

    Args:
        processor: The document processor.
        images: The pages, in order.
        app_config: The run's settings.

    Returns:
        One classification record per page, in the order given.
    """
    started = time.time()
    records = _classify_images(
        processor,
        [str(image) for image in images],
        read_classify_batch_size(app_config.raw, config_path=app_config.config_path),
        app_config.verbose,
    )
    share = (time.time() - started) / len(records) if records else 0.0
    for record in records:
        record["processing_time"] = share
    return records


def _record(image_path: str, result: dict[str, Any]) -> dict[str, Any]:
    record = {
        "image_path": image_path,
        "image_name": Path(image_path).name,
        "document_type": result["document_type"],
        "confidence": result.get("confidence", 1.0),
        "raw_response": result.get("raw_response", ""),
        "prompt_used": result.get("prompt_used", "detection"),
    }
    if result.get("error"):
        record["error"] = result["error"]
    return record


def _classify_images(
    processor: Any, image_paths: list[str], batch_size: int, verbose: bool
) -> list[dict[str, Any]]:
    """Classify pages, ``batch_size`` to an engine call.

    A page is one independent request, so the engine runs the pages of a call at once. A call that fails
    is tried again one page at a time, and a page that still fails is given the fallback type and
    carries its error, so that one page never ends the run (``common.engine_calls``).

    Args:
        processor: The document processor (orchestrator).
        image_paths: Pages to classify, in order.
        batch_size: Pages per engine call. 1 means one call a page.
        verbose: Passed through to the processor.

    Returns:
        One classification record per page, in input order.
    """
    total = len(image_paths)
    if batch_size > 1:
        logger.info(
            "Classification batching: %d pages in %d calls (%d a call)",
            total,
            -(-total // batch_size),
            batch_size,
        )

    records: list[dict[str, Any]] = []
    calls = in_engine_calls(
        groups_of(image_paths, batch_size),
        lambda pages: processor.detect_many(pages, verbose=verbose),
        what="pages",
    )
    for outcomes in calls:
        for outcome in outcomes:
            result = outcome.reply if outcome.reply is not None else processor.unclassified(outcome.error)
            records.append(_record(outcome.item, result))
            # Tier A: always-on per-image progress.
            logger.info(
                "[%d/%d] %s -> %s", len(records), total, Path(outcome.item).name, result["document_type"]
            )
    return records
