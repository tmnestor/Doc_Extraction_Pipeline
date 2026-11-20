# ruff: noqa: B008 - typer.Option in defaults is the standard Typer pattern
"""Stage 2: Field extraction (GPU).

Reads classifications.jsonl, runs type-specific extraction on each image,
and writes RAW model responses to raw_extractions.jsonl.

The raw response is the most valuable debugging artifact -- you can re-parse,
re-clean, and re-evaluate without touching the GPU.

Supports resumption: if the output file already has partial results from a
crashed run, skips already-processed images and appends.

Usage:
    python -m stages.extract \
        --classifications /artifacts/classifications.jsonl \
        --image-dir /data \
        --output /artifacts/raw_extractions.jsonl
"""

import logging
import time
from pathlib import Path
from typing import Any

import typer

from .io import (
    RecordSink,
    StreamingJsonlWriter,
    append_jsonl,
    read_completed_images,
    read_jsonl,
    write_jsonl,
)

logger = logging.getLogger(__name__)
app = typer.Typer()


def _record_stage_timing(
    output_dir: Path,
    *,
    stage: str,
    records: list[dict[str, Any]],
    wall_clock: float,
    execution_mode: str,
) -> None:
    """Write this stage's timing sidecar for the evaluate summary.

    Observability only — a failure here must never fail a run that has
    already done its expensive work.
    """
    try:
        from common.stage_timing import (
            StageTiming,
            inference_seconds_from_records,
            write_stage_timing,
        )

        write_stage_timing(
            output_dir,
            StageTiming(
                stage=stage,
                wall_clock=wall_clock,
                inference_seconds=inference_seconds_from_records(records, fallback=wall_clock),
                execution_mode=execution_mode,
                images=len(records),
            ),
        )
    except Exception as err:  # noqa: BLE001 - timing must not break the stage
        logger.warning("Could not record %s stage timing: %s", stage, err)


def _build_cli_args(
    image_dir: Path,
    output_path: Path,
    *,
    model_type: str | None = None,
    batch_size: int | None = None,
    bank_v2: bool | None = None,
    max_num_seqs: int | None = None,
    verbose: bool | None = None,
    debug: bool | None = None,
) -> dict[str, Any]:
    """Build the CLI-overrides dict for the config cascade (CLI > YAML).

    Every optional key is injected only when the caller passed an explicit
    value — None means "let YAML win" (run_config.yml pipeline.processing.*).
    """
    cli_args: dict[str, Any] = {
        "data_dir": str(image_dir),
        "output_dir": str(output_path.parent),
    }
    if model_type is not None:
        cli_args["model_type"] = model_type
    if batch_size is not None:
        cli_args["batch_size"] = batch_size
    if bank_v2 is not None:
        cli_args["bank_v2"] = bank_v2
    if max_num_seqs is not None:
        cli_args["max_num_seqs"] = max_num_seqs
    if verbose is not None:
        cli_args["verbose"] = verbose
    if debug is not None:
        cli_args["debug"] = debug
    return cli_args


def run(
    classifications_path: Path | None,
    image_dir: Path,
    output_path: Path,
    *,
    model_type: str | None = None,
    batch_size: int | None = None,
    bank_v2: bool | None = None,
    max_num_seqs: int | None = None,
    verbose: bool | None = None,
    debug: bool | None = None,
    config_path: Path | None = None,
) -> Path:
    """Extract fields from classified images, write raw_extractions.jsonl.

    Writes the **raw model response string** per image, not parsed/cleaned
    data.  The clean stage handles parsing.

    Supports resumption: if output_path exists with partial results, skips
    already-processed images and appends new results.

    Args:
        classifications_path: Path to classifications.jsonl from Stage 1.
        image_dir: Directory containing the original images.
        output_path: Path to write raw_extractions.jsonl.
        model_type: Model type (e.g. "internvl3-vllm").
        batch_size: Images per batch (None = auto-detect, 1 = sequential).
        bank_v2: Use UnifiedBankExtractor for bank statements. None = YAML
            processing.bank_v2.
        verbose: Tier B output (init/config details). None = read from YAML.
        debug: Tier C output (dev-noise: PARSING DEBUG, prompt dumps). None = YAML.
        config_path: Optional path to run_config.yml.

    Returns:
        Path to the written raw_extractions.jsonl.
    """
    if classifications_path is None:
        msg = "--classifications is required"
        raise typer.BadParameter(msg)

    from common.pipeline_ops import load_pipeline_configs
    from common.app_config import AppConfig
    from common.extraction_sort import filter_skip_labels, sort_for_extraction
    from common.pipeline_ops import create_processor, load_model
    from common.unified_bank_extractor import UnifiedBankExtractor

    # Read classifications
    classifications = read_jsonl(classifications_path)
    if not classifications:
        msg = f"No classifications found in {classifications_path}"
        raise FileNotFoundError(msg)

    logger.info("Read %d classifications from %s", len(classifications), classifications_path)

    # Check for resumption
    completed = read_completed_images(output_path)
    if completed:
        original_count = len(classifications)
        classifications = [c for c in classifications if c["image_name"] not in completed]
        logger.info(
            "Resuming: %d already done, %d remaining (of %d total)",
            len(completed),
            len(classifications),
            original_count,
        )

    if not classifications:
        logger.info("All images already processed -- nothing to do")
        return output_path

    # Build config through the standard cascade. Optional keys are injected
    # only when the caller passed an explicit value — None means "let YAML win".
    cli_args = _build_cli_args(
        image_dir,
        output_path,
        model_type=model_type,
        batch_size=batch_size,
        bank_v2=bank_v2,
        max_num_seqs=max_num_seqs,
        verbose=verbose,
        debug=debug,
    )

    app_cfg = AppConfig.load(cli_args, config_path=config_path)
    config = app_cfg.pipeline
    # Tier B gate — resolved from CLI > YAML > defaults.
    effective_verbose = config.verbose

    # -- Phase 4: filter out skip-labels before extraction -----------------------
    classifications, skipped = filter_skip_labels(classifications, app_cfg.extraction_skip_labels)
    if skipped:
        logger.info("Skipping %d records with skip labels", len(skipped))
        # Write skipped records to output so downstream sees one record per input
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with StreamingJsonlWriter(output_path) as writer:
            for rec in skipped:
                writer.write(
                    {
                        "image_name": rec.get("image_name", ""),
                        "image_path": rec.get("image_path", ""),
                        "document_type": rec.get("document_type", ""),
                        "raw_response": "",
                        "processing_time": 0.0,
                        "prompt_used": "skipped",
                        "error": None,
                        "skipped": True,
                        "skip_reason": rec.get("document_type", ""),
                        "fields": {},
                    }
                )

    if not classifications:
        logger.info("All remaining images skipped -- nothing to extract")
        return output_path

    # -- Phase 2: sort by doc_type for prefix-cache reuse ------------------------
    classifications = sort_for_extraction(
        classifications,
        app_cfg.extraction_order,
        secondary_sort=app_cfg.secondary_sort,
    )
    logger.info(
        "Sorted %d records by extraction_order (secondary=%s)",
        len(classifications),
        app_cfg.secondary_sort,
    )

    # Stage wall clock, used by BOTH paths below so "Total Wall Clock" and
    # "Engine Startup" mean the same thing however the stage ran.
    stage_started = time.time()

    # -- vLLM data-parallel fast path (classified) --------------------------------
    from models.registry import is_vllm_model

    if is_vllm_model(config.model_type):
        from common.vllm_dp import resolve_dp_gpus, run_dp

        # None -> single GPU, or a model that can't replicate per GPU.
        resolved_gpus = resolve_dp_gpus(config, config.model_type)
        if resolved_gpus:
            records = run_dp(
                num_gpus=resolved_gpus,
                images=[Path(c["image_path"]) for c in classifications],
                worker_fn="common.vllm_dp_workers.classified_extract_worker",
                worker_kwargs={
                    "config_path": str(config_path) if config_path else None,
                    "cli_overrides": cli_args,
                    "classifications": classifications,
                },
            )
            # Wall clock here spans each worker spinning up its own engine;
            # inference comes from the slowest worker's per-image times.
            _record_stage_timing(
                output_path.parent,
                stage="extract",
                records=records,
                wall_clock=time.time() - stage_started,
                execution_mode=f"data-parallel ({resolved_gpus} GPUs)",
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Resume-safe: APPEND when resuming (preserve already-extracted
            # records) OR when skip-label records were written above (a fresh DP
            # run with skips). write_jsonl truncates, which would drop either.
            write_fn = append_jsonl if (completed or skipped) else write_jsonl
            count = write_fn(output_path, records)
            logger.info("Wrote %d extractions to %s", count, output_path)
            return output_path

    # -- Single-GPU / HF path ----------------------------------------------------

    prompt_config, universal_fields, field_definitions = load_pipeline_configs(config.model_type)

    # Load model
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

        # Set up bank processing
        bank_adapter = None
        has_bank = any(c["document_type"].upper() == "BANK_STATEMENT" for c in classifications)

        if has_bank and config.bank_v2 and getattr(processor, "supports_multi_turn", True):
            bank_adapter = UnifiedBankExtractor(
                generate_fn=processor.generate,
                generate_many_fn=(
                    processor.generate_many if getattr(processor, "supports_generate_many", False) else None
                ),
                verbose=effective_verbose,
            )
            logger.info("Bank adapter enabled")

        # Process images with streaming writer (flush per record for crash recovery)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        start = time.time()
        count = 0
        total = len(classifications)

        batches = _plan_batches(classifications, config_path, _unbatchable_types(bank_adapter))

        with StreamingJsonlWriter(output_path) as writer:
            for batch in batches:
                # A multi-document batch is one engine call. Singletons (any type
                # whose batch size is 1) fall through to the per-image path below.
                if len(batch) > 1:
                    if batch[0]["document_type"].upper() == "BANK_STATEMENT":
                        # Only planned when the adapter can batch.
                        _extract_bank_single_pass_batch(writer, bank_adapter, batch)
                    else:
                        _extract_standard_batch(writer, processor, batch, effective_verbose)
                    count += len(batch)
                    logger.info(
                        "[%d/%d] batch of %d %s",
                        count,
                        total,
                        len(batch),
                        batch[0]["document_type"],
                    )
                    continue

                classification = batch[0]
                image_path = classification["image_path"]
                image_name = classification["image_name"]
                doc_type = classification["document_type"]

                img_start = time.time()

                try:
                    if doc_type.upper() == "BANK_STATEMENT" and bank_adapter is not None:
                        _extract_bank_with_adapter(
                            writer,
                            bank_adapter,
                            image_path,
                            image_name,
                            doc_type,
                        )
                    else:
                        _extract_standard(
                            writer,
                            processor,
                            image_path,
                            image_name,
                            doc_type,
                            classification,
                            effective_verbose,
                        )
                except Exception as e:
                    logger.error("Error extracting %s: %s", image_name, e)
                    img_time = time.time() - img_start
                    writer.write(
                        {
                            "image_name": image_name,
                            "image_path": image_path,
                            "document_type": doc_type,
                            "raw_response": "",
                            "processing_time": img_time,
                            "prompt_used": "error",
                            "error": str(e),
                        }
                    )

                count += 1
                img_time = time.time() - img_start
                logger.info(
                    "[%d/%d] %s: %s (%.1fs)",
                    count,
                    total,
                    image_name,
                    doc_type,
                    img_time,
                )

        elapsed = time.time() - start
        logger.info("Extraction complete: %d images in %.1fs", count, elapsed)
        # Sequential, so the loop's elapsed IS the inference time; the gap
        # to stage wall clock is engine startup.
        _record_stage_timing(
            output_path.parent,
            stage="extract",
            records=[{"processing_time": elapsed}],
            wall_clock=time.time() - stage_started,
            execution_mode="single-engine",
        )
    finally:
        model_cm.__exit__(None, None, None)

    return output_path


def _extract_bank_with_adapter(
    writer: RecordSink,
    bank_adapter,
    image_path: str,
    image_name: str,
    doc_type: str,
) -> None:
    """Extract bank statement via UnifiedBankExtractor, write raw responses."""
    start = time.time()

    schema_fields, metadata = bank_adapter.extract_bank_statement(image_path)

    img_time = time.time() - start

    # Serialize structured fields into flat FIELD: value text that the
    # standard hybrid parser already handles.  This keeps the stage contract
    # clean: extract produces a parseable raw_response, clean parses it.
    from common.unified_bank_extractor import serialise_bank_response

    raw_response_str = serialise_bank_response(schema_fields, metadata)

    strategy = metadata.get("strategy_used", "unknown")

    writer.write(
        {
            "image_name": image_name,
            "image_path": image_path,
            "document_type": doc_type,
            "raw_response": raw_response_str,
            "processing_time": img_time,
            "prompt_used": f"unified_bank_{strategy}",
            "error": None,
        }
    )


# Types that cannot share an engine call unless the run can batch them.
UNBATCHABLE_TYPES = {"BANK_STATEMENT"}


def _unbatchable_types(bank_adapter: Any) -> set[str]:
    """The document types that must each be their own engine call in this run.

    A bank statement is one independent call, so several can be submitted together and run
    concurrently. That needs the adapter and a backend with a batched call; otherwise they stay
    unbatched.
    """
    if bank_adapter is not None and bank_adapter.can_batch:
        return UNBATCHABLE_TYPES - {"BANK_STATEMENT"}
    return set(UNBATCHABLE_TYPES)


def _extract_bank_single_pass_batch(
    writer: RecordSink,
    bank_adapter: Any,
    batch: list[dict[str, Any]],
) -> None:
    """Read a batch of bank statements single pass in one engine call.

    Writes the record _extract_bank_with_adapter writes for each statement,
    so batched and unbatched runs are indistinguishable in the output file. A
    batch that raises is retried one statement at a time, and a statement that
    still fails is recorded as an error, as on the per-image path.
    """
    from common.unified_bank_extractor import serialise_bank_response

    started = time.time()
    try:
        results = bank_adapter.extract_bank_statements_single_pass([c["image_path"] for c in batch])
    except Exception as err:  # noqa: BLE001 - one batch must not end the run
        logger.warning(
            "Batch of %d BANK_STATEMENT failed (%s); retrying one at a time",
            len(batch),
            err,
        )
        for classification in batch:
            retry_started = time.time()
            try:
                _extract_bank_with_adapter(
                    writer,
                    bank_adapter,
                    classification["image_path"],
                    classification["image_name"],
                    classification["document_type"],
                )
            except Exception as retry_err:  # noqa: BLE001 - recorded, not raised
                logger.error("Error extracting %s: %s", classification["image_name"], retry_err)
                writer.write(
                    {
                        "image_name": classification["image_name"],
                        "image_path": classification["image_path"],
                        "document_type": classification["document_type"],
                        "raw_response": "",
                        "processing_time": time.time() - retry_started,
                        "prompt_used": "error",
                        "error": str(retry_err),
                    }
                )
        return

    # Inside a batch the requests overlap, so per-image timing is not
    # separable; charge each an equal share of the call's wall clock.
    share = (time.time() - started) / len(batch)
    for classification, (schema_fields, metadata) in zip(batch, results, strict=True):
        writer.write(
            {
                "image_name": classification["image_name"],
                "image_path": classification["image_path"],
                "document_type": classification["document_type"],
                "raw_response": serialise_bank_response(schema_fields, metadata),
                "processing_time": share,
                "prompt_used": f"unified_bank_{metadata.get('strategy_used', 'unknown')}",
                "error": None,
            }
        )


def _plan_batches(
    classifications: list[dict[str, Any]],
    config_path: Path | None,
    unbatchable: set[str],
) -> list[list[dict[str, Any]]]:
    """Group classifications into engine-call batches.

    Falls back to one document per batch if the batching config cannot be
    read — the pre-batching behaviour, which is always safe.
    """
    from common.extraction_batching import plan_extraction_batches, read_batch_sizes
    from common.pipeline_config import load_yaml_config

    resolved = config_path or Path(__file__).resolve().parent.parent / "config" / "run_config.yml"
    _, raw_config = load_yaml_config(resolved)
    batch_sizes = read_batch_sizes(raw_config, config_path=resolved)

    batches = plan_extraction_batches(
        classifications,
        batch_sizes=batch_sizes,
        unbatchable=unbatchable,
    )
    logger.info(
        "Extraction batching: %d documents in %d calls (sizes: %s)",
        len(classifications),
        len(batches),
        batch_sizes,
    )
    return batches


def _extract_standard_batch(
    writer: StreamingJsonlWriter,
    processor,
    batch: list[dict[str, Any]],
    verbose: bool,
) -> None:
    """Extract a batch of standard documents in one engine call.

    Writes the same record shape as ``_extract_standard`` — RAW responses,
    which the clean stage parses. The two paths must stay
    indistinguishable in the output file, or resume and re-cleaning break.

    A batch that raises is retried one document at a time, so an
    out-of-memory on a large batch costs only the document that genuinely
    fails rather than every document that shared it.
    """
    started = time.time()
    image_paths = [c["image_path"] for c in batch]

    try:
        results = processor.extract_batch(image_paths, batch, verbose=verbose)
        if len(results) != len(batch):
            raise ValueError(
                f"backend returned {len(results)} results for {len(batch)} images; "
                f"matching by position is unsafe"
            )
    except Exception as err:  # noqa: BLE001 - one batch must not end the run
        logger.warning(
            "Batch of %d %s failed (%s) — retrying one at a time",
            len(batch),
            batch[0]["document_type"],
            err,
        )
        for classification in batch:
            _extract_standard(
                writer,
                processor,
                classification["image_path"],
                classification["image_name"],
                classification["document_type"],
                classification,
                verbose,
            )
        return

    # Inside a batch the requests overlap, so per-image timing is not
    # separable; charge each an equal share of the call's wall clock.
    share = (time.time() - started) / len(batch)
    for classification, result in zip(batch, results, strict=True):
        writer.write(
            {
                "image_name": classification["image_name"],
                "image_path": classification["image_path"],
                "document_type": classification["document_type"],
                "raw_response": result.get("raw_response", ""),
                "processing_time": share,
                "prompt_used": classification["document_type"].lower(),
                "error": None,
            }
        )


def _extract_standard(
    writer: StreamingJsonlWriter,
    processor,
    image_path: str,
    image_name: str,
    doc_type: str,
    classification: dict[str, Any],
    verbose: bool,
) -> None:
    """Extract standard document, write raw response."""
    start = time.time()
    result = processor.process_document_aware(image_path, classification, verbose=verbose)
    img_time = time.time() - start

    writer.write(
        {
            "image_name": image_name,
            "image_path": image_path,
            "document_type": doc_type,
            "raw_response": result.get("raw_response", ""),
            "processing_time": img_time,
            "prompt_used": doc_type.lower(),
            "error": None,
        }
    )


@app.command()
def main(
    classifications: Path | None = typer.Option(
        None, "--classifications", help="Path to classifications.jsonl from Stage 1"
    ),
    image_dir: Path = typer.Option(..., "--data-dir", "-d", help="Directory containing images"),
    output: Path = typer.Option(..., "--output-dir", "-o", help="Path to write raw_extractions.jsonl"),
    model: str | None = typer.Option(None, "--model", help="Model type"),
    batch_size: int | None = typer.Option(None, "--batch-size", help="Images per extraction batch"),
    bank_v2: bool | None = typer.Option(
        None,
        "--bank-v2/--no-bank-v2",
        help="Use UnifiedBankExtractor for bank statements. Default: YAML processing.bank_v2.",
    ),
    max_num_seqs: int | None = typer.Option(
        None,
        "--max-num-seqs",
        help="Override vLLM max_num_seqs (continuous batching). Sweep: 4 → 8 → 16.",
    ),
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
    """Stage 2: Extract fields from classified images (raw responses)."""
    # Tier A (per-image progress, phase headings) is always at INFO — it's the
    # only indication the user has that inference is progressing. Verbose/debug
    # are independent switches for Tiers B/C inside the orchestrator.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    run(
        classifications,
        image_dir,
        output,
        model_type=model,
        batch_size=batch_size,
        bank_v2=bank_v2,
        max_num_seqs=max_num_seqs,
        verbose=verbose,
        debug=debug,
        config_path=config,
    )


if __name__ == "__main__":
    app()
