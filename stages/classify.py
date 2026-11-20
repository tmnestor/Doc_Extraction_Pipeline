# ruff: noqa: B008 - typer.Option in defaults is the standard Typer pattern
"""Stage 1: Document type classification (GPU).

Detects document types for all images and writes classifications.jsonl.

Usage:
    python -m stages.classify --image-dir /data --output /artifacts/classifications.jsonl
    python -m stages.classify --image-dir /data --output /artifacts/classifications.jsonl --model llama
"""

import logging
import time
from pathlib import Path
from typing import Any

import typer

from .io import append_jsonl, read_completed_images, write_jsonl

logger = logging.getLogger(__name__)
app = typer.Typer()


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
    batch_size: int | None = None,
    verbose: bool | None = None,
    debug: bool | None = None,
    config_path: Path | None = None,
) -> Path:
    """Detect document types for all images, write classifications.jsonl.

    Args:
        image_dir: Directory containing images.
        output_path: Path to write classifications.jsonl.
        model_type: Model type (e.g. "internvl3-vllm").
        batch_size: Images per batch (None = auto-detect, 1 = sequential).
        verbose: Tier B output (init/config details). None = read from YAML.
        debug: Tier C output (dev-noise: PARSING DEBUG, prompt dumps). None = YAML.
        config_path: Optional path to run_config.yml.

    Returns:
        Path to the written classifications.jsonl.
    """
    from common.pipeline_ops import load_pipeline_configs
    from common.app_config import AppConfig
    from common.pipeline_config import discover_images
    from common.pipeline_ops import create_processor, load_model

    # Build config through the standard cascade. Only inject verbose/debug
    # when the caller passed an explicit bool — None means "let YAML win".
    cli_args: dict[str, Any] = {
        "data_dir": str(image_dir),
        "output_dir": str(output_path.parent),
    }
    if model_type is not None:
        cli_args["model_type"] = model_type
    if verbose is not None:
        cli_args["verbose"] = verbose
    if debug is not None:
        cli_args["debug"] = debug
    if batch_size is not None:
        cli_args["batch_size"] = batch_size

    app_cfg = AppConfig.load(cli_args, config_path=config_path).for_stage("classify")
    config = app_cfg.pipeline
    # Tier B gate — resolved from CLI > YAML > defaults.
    effective_verbose = config.verbose

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

    # -- vLLM data-parallel fast path ------------------------------------------
    from models.registry import is_vllm_model

    if is_vllm_model(config.model_type):
        from common.vllm_dp import resolve_dp_gpus, run_dp

        # None -> single GPU, or a model that can't replicate per GPU.
        resolved_gpus = resolve_dp_gpus(config, config.model_type)
        if resolved_gpus:
            dp_records = run_dp(
                num_gpus=resolved_gpus,
                images=images,
                worker_fn="common.vllm_dp_workers.classify_worker",
                worker_kwargs={
                    "config_path": str(config_path) if config_path else None,
                    "cli_overrides": cli_args,
                },
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_fn = append_jsonl if resuming else write_jsonl
            count = write_fn(output_path, dp_records)
            logger.info("Wrote %d classifications to %s", count, output_path)
            return output_path

    # -- Single-GPU / HF path -------------------------------------------------

    # Load pipeline configs (prompts, fields)
    prompt_config, universal_fields, field_definitions = load_pipeline_configs(config.model_type)

    # Load model and run detection
    logger.info("Loading model: %s", config.model_type)
    model_cm = load_model(config, app_config=app_cfg)
    model, tokenizer = model_cm.__enter__()

    try:
        processor = create_processor(
            model,
            tokenizer,
            config,
            prompt_config,
            universal_fields,
            field_definitions,
            app_config=app_cfg,
        )

        image_paths = [str(img) for img in images]
        records: list[dict[str, Any]] = []

        effective_batch = config.batch_size or 1
        shares_calls = getattr(processor, "supports_generate_many", False)
        can_batch = not shares_calls and getattr(processor, "supports_batch", False) and effective_batch > 1

        start = time.time()

        if shares_calls:
            records = _classify_images(
                processor, image_paths, _classify_batch_size(config_path), effective_verbose
            )
        elif can_batch:
            for batch_start in range(0, len(image_paths), effective_batch):
                batch_end = min(batch_start + effective_batch, len(image_paths))
                batch = image_paths[batch_start:batch_end]

                logger.info(
                    "Detecting batch [%d-%d] / %d",
                    batch_start + 1,
                    batch_end,
                    len(image_paths),
                )

                results = processor.detect_batch(batch, verbose=effective_verbose)
                for i, result in enumerate(results):
                    image_name = Path(batch[i]).name
                    records.append(
                        {
                            "image_path": batch[i],
                            "image_name": image_name,
                            "document_type": result["document_type"],
                            "confidence": result.get("confidence", 1.0),
                            "raw_response": result.get("raw_response", ""),
                            "prompt_used": result.get("prompt_used", "detection"),
                        }
                    )
                    # Tier A: always-on per-image progress.
                    logger.info(
                        "[%d/%d] %s -> %s",
                        batch_start + i + 1,
                        len(image_paths),
                        image_name,
                        result["document_type"],
                    )
        else:
            records = _classify_images(processor, image_paths, 1, effective_verbose)

        elapsed = time.time() - start
        logger.info(
            "Classification complete: %d images in %.1fs (%.2fs/image)",
            len(records),
            elapsed,
            elapsed / len(records) if records else 0,
        )
    finally:
        model_cm.__exit__(None, None, None)

    # Write output (append when resuming so prior records are preserved)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_fn = append_jsonl if resuming else write_jsonl
    count = write_fn(output_path, records)
    logger.info("Wrote %d classifications to %s", count, output_path)

    return output_path


def _classify_batch_size(config_path: Path | None) -> int:
    """Pages per classify engine call, from pipeline.batching.classify."""
    from common.extraction_batching import read_classify_batch_size
    from common.pipeline_config import load_yaml_config

    resolved = config_path or Path(__file__).resolve().parent.parent / "config" / "run_config.yml"
    _, raw_config = load_yaml_config(resolved)
    return read_classify_batch_size(raw_config, config_path=resolved)


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


@app.command()
def main(
    image_dir: Path = typer.Option(..., "--data-dir", "-d", help="Directory containing images"),
    output: Path = typer.Option(..., "--output-dir", "-o", help="Path to write classifications.jsonl"),
    model: str | None = typer.Option(None, "--model", help="Model type"),
    batch_size: int | None = typer.Option(None, "--batch-size", help="Images per detection batch"),
    config: Path | None = typer.Option(None, "--config", help="YAML configuration file"),
    verbose: bool | None = typer.Option(
        None,
        "--verbose/--no-verbose",
        help="Tier B output (init details). Default: read YAML processing.verbose.",
    ),
    debug: bool | None = typer.Option(
        None,
        "--debug/--no-debug",
        help="Tier C output (PARSING DEBUG, prompt dumps). Default: YAML processing.debug.",
    ),
) -> None:
    """Stage 1: Classify document types for all images."""
    # Tier A (per-image progress, phase headings) is always at INFO — it's the
    # only indication the user has that inference is progressing. Verbose/debug
    # are independent switches for Tiers B/C inside the orchestrator.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    run(
        image_dir,
        output,
        model_type=model,
        batch_size=batch_size,
        verbose=verbose,
        debug=debug,
        config_path=config,
    )


if __name__ == "__main__":
    app()
