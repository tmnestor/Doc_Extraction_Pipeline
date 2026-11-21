"""Stage 1: Document type classification (GPU).

Detects document types for all images and writes classifications.jsonl.

Started by ``python -m stages`` with ``KFP_TASK`` naming it:

    KFP_TASK=classify model=<a registered model> image_dir=<folder> output=<folder> bash entrypoint.sh
"""

import logging
import time
from pathlib import Path
from typing import Any


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
    from common.pipeline_ops import open_processor

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
    # (clear the output first — the entrypoint does this for full re-runs). This
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
    from common.vllm_dp import resolve_dp_gpus, run_dp

    started = time.time()
    engines = resolve_dp_gpus(config, config.model_type)
    if engines:
        records = run_dp(
            num_gpus=engines,
            images=images,
            worker_fn="common.vllm_dp_workers.classify_worker",
            worker_kwargs={
                "config_path": str(config_path) if config_path else None,
                "cli_overrides": cli_args,
            },
            app_config=app_cfg,
        )
    else:
        with open_processor(app_cfg) as processor:
            records = classify_pages(processor, images, app_cfg)

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
    return {
        "image_path": image_path,
        "image_name": Path(image_path).name,
        "document_type": result["document_type"],
        "confidence": result.get("confidence", 1.0),
        "raw_response": result.get("raw_response", ""),
        "prompt_used": result.get("prompt_used", "detection"),
    }


def _classify_images(
    processor: Any, image_paths: list[str], batch_size: int, verbose: bool
) -> list[dict[str, Any]]:
    """Classify pages, several to an engine call when the backend allows it.

    A page is one independent model call. With a batch size above 1 and a
    backend that can share an engine call, each batch is submitted together so
    the engine runs the pages concurrently; each request is built as the
    one-at-a-time call builds it. A batch that fails is retried one page at a
    time, so a single bad page costs its batch only the concurrency.

    Args:
        processor: The document processor (orchestrator).
        image_paths: Pages to classify, in order.
        batch_size: Pages per engine call. 1 means one call a page.
        verbose: Passed through to the processor.

    Returns:
        One classification record per page, in input order.
    """
    total = len(image_paths)
    share = batch_size > 1 and getattr(processor, "supports_generate_many", False)
    step = batch_size if share else 1
    if share:
        logger.info(
            "Classification batching: %d pages in %d calls (%d a call)",
            total,
            -(-total // step),
            step,
        )

    records: list[dict[str, Any]] = []
    for start in range(0, total, step):
        batch = image_paths[start : start + step]
        results = None
        if share:
            try:
                results = processor.detect_many(batch, verbose=verbose)
            except Exception as err:  # noqa: BLE001 - any batch failure falls back to single calls
                logger.warning(
                    "Classify batch of %d failed (%s: %s); retrying one at a time",
                    len(batch),
                    type(err).__name__,
                    err,
                )
        if results is None:
            results = []
            for offset, image_path in enumerate(batch):
                logger.info(
                    "Sending image %d/%d to engine: %s", start + offset + 1, total, Path(image_path).name
                )
                results.append(processor.detect_and_classify_document(image_path, verbose=verbose))
        for offset, (image_path, result) in enumerate(zip(batch, results, strict=True)):
            records.append(_record(image_path, result))
            # Tier A: always-on per-image progress.
            logger.info(
                "[%d/%d] %s -> %s",
                start + offset + 1,
                total,
                Path(image_path).name,
                result["document_type"],
            )
    return records
