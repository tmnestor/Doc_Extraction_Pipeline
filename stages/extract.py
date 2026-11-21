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
        model_type: A registered model.
        batch_size: Images per batch (None = auto-detect, 1 = sequential).
        verbose: Tier B output (init/config details). None = read from YAML.
        debug: Tier C output (dev-noise: PARSING DEBUG, prompt dumps). None = YAML.
        config_path: Optional path to run_config.yml.

    Returns:
        Path to the written raw_extractions.jsonl.
    """
    if classifications_path is None:
        msg = "--classifications is required"
        raise typer.BadParameter(msg)

    from common.app_config import AppConfig
    from common.extraction_sort import filter_skip_labels, sort_for_extraction
    from common.pipeline_ops import open_processor

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
        max_num_seqs=max_num_seqs,
        verbose=verbose,
        debug=debug,
    )

    app_cfg = AppConfig.load(cli_args, config_path=config_path)
    config = app_cfg.pipeline
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

    # One engine on each card where the model allows it, else one engine across the cards. Either way the
    # pages are read by extract_pages; only where the engine is started differs.
    from common.stage_timing import inference_seconds_from_records
    from common.vllm_dp import resolve_dp_gpus, run_dp

    stage_started = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    engines = resolve_dp_gpus(config, config.model_type)
    if engines:
        records = run_dp(
            num_gpus=engines,
            images=[Path(c["image_path"]) for c in classifications],
            worker_fn="common.vllm_dp_workers.extract_worker",
            worker_kwargs={
                "config_path": str(config_path) if config_path else None,
                "cli_overrides": cli_args,
                "classifications": classifications,
            },
        )
        # The workers hand their records back when they finish. Append when resuming, or when skipped
        # pages were written above: write_jsonl starts the file again and would drop either.
        write_fn = append_jsonl if (completed or skipped) else write_jsonl
        write_fn(output_path, records)
        execution_mode = f"data-parallel ({engines} GPUs)"
    else:
        # One engine writes each record as it is made, so a run that stops part way can be resumed.
        with open_processor(app_cfg) as processor, StreamingJsonlWriter(output_path) as writer:
            records = extract_pages(processor, classifications, app_cfg, sink=writer)
        execution_mode = "single-engine"

    wall_clock = time.time() - stage_started
    # The model's time, apart from starting the engine: the sum of the pages' times, and with an engine on
    # each card the slowest card's sum.
    logger.info(
        "Extraction complete: %d images in %.1fs",
        len(records),
        inference_seconds_from_records(records, fallback=wall_clock),
    )
    _record_stage_timing(
        output_path.parent,
        stage="extract",
        records=records,
        wall_clock=wall_clock,
        execution_mode=execution_mode,
    )
    logger.info("Wrote %d extractions to %s", len(records), output_path)
    return output_path


class _Collected:
    """Keeps every record a reading function writes, and passes each on to a sink if there is one."""

    def __init__(self, sink: RecordSink | None) -> None:
        self.records: list[dict[str, Any]] = []
        self._sink = sink

    def write(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        if self._sink is not None:
            self._sink.write(record)


def extract_pages(
    processor: Any,
    classified: list[dict[str, Any]],
    app_config: Any,
    *,
    sink: RecordSink | None = None,
) -> list[dict[str, Any]]:
    """Read the fields of each classified page. The one place pages are read, however engines are arranged.

    One engine across the cards calls this once with every page. With an engine on each card, each worker
    calls it with its share. It loads no model and decides nothing about which pages are already done: the
    stage's ``run`` does those.

    Pages are grouped into engine calls by document type, by the sizes in ``pipeline.batching``. A call that
    fails is tried again one page at a time, and a page that still fails is recorded with its error, so one
    page never ends the run.

    Args:
        processor: The document processor.
        classified: The pages' classification records, one document type after another.
        app_config: The run's settings.
        sink: Where to write each record as soon as it is made, if anywhere.

    Returns:
        One record per page, holding the model's raw reply, in the order given.
    """
    collected = _Collected(sink)
    done, total = 0, len(classified)
    for batch in _plan_batches(classified, app_config.config_path):
        # A dense statement can take a while, so say what is in flight before it starts.
        logger.info(
            "[%d/%d] reading %d %s...", done + len(batch), total, len(batch), batch[0]["document_type"]
        )
        _read_batch(collected, processor, batch)
        done += len(batch)
    return collected.records


def _record(classification: dict[str, Any], **rest: Any) -> dict[str, Any]:
    return {
        "image_name": classification["image_name"],
        "image_path": classification["image_path"],
        "document_type": classification["document_type"],
        **rest,
    }


def _prompt_name(document_type: str) -> str:
    """What a record says its page was read with. A statement keeps the name it had when it had its own way."""
    return (
        "unified_bank_single_pass" if document_type.upper() == "BANK_STATEMENT" else document_type.lower()
    )


def _read_batch(writer: RecordSink, processor: Any, batch: list[dict[str, Any]]) -> None:
    """Read a group of pages of one document type in one engine call, and write a record for each.

    A group that fails is read again one page at a time, so a single bad page costs its group only the
    shared call. A page that fails on its own is recorded with its error, so that it never ends the run.
    """
    from common import prompt_trace

    kind = batch[0]["document_type"]
    started = time.time()
    try:
        if len(batch) == 1:
            # Names the page in the trace of what was sent to the model, when that trace is switched on.
            with prompt_trace.trace_context(
                image_name=batch[0]["image_name"],
                pipeline="information_extraction",
                label=f"extract_{kind.lower()}",
            ):
                replies = processor.read_pages([batch[0]["image_path"]], kind)
        else:
            replies = processor.read_pages([c["image_path"] for c in batch], kind)
        if len(replies) != len(batch):
            raise ValueError(
                f"backend returned {len(replies)} replies for {len(batch)} pages; "
                f"matching by position is unsafe"
            )
    except Exception as err:  # noqa: BLE001 - one group, or one page, must not end the run
        if len(batch) > 1:
            logger.warning("Batch of %d %s failed (%s); retrying one at a time", len(batch), kind, err)
            for classification in batch:
                _read_batch(writer, processor, [classification])
            return
        logger.error("Error extracting %s: %s", batch[0]["image_name"], err)
        writer.write(
            _record(
                batch[0],
                raw_response="",
                processing_time=time.time() - started,
                prompt_used="error",
                error=str(err),
            )
        )
        return

    # Inside a shared call the requests overlap, so each page is charged an equal share of its time.
    share = (time.time() - started) / len(batch)
    for classification, reply in zip(batch, replies, strict=True):
        writer.write(
            _record(
                classification,
                raw_response=reply,
                processing_time=share,
                prompt_used=_prompt_name(kind),
                error=None,
            )
        )


def _plan_batches(
    classifications: list[dict[str, Any]],
    config_path: Path | None,
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
        unbatchable=set(),
    )
    logger.info(
        "Extraction batching: %d documents in %d calls (sizes: %s)",
        len(classifications),
        len(batches),
        batch_sizes,
    )
    return batches


@app.command()
def main(
    classifications: Path | None = typer.Option(
        None, "--classifications", help="Path to classifications.jsonl from Stage 1"
    ),
    image_dir: Path = typer.Option(..., "--data-dir", "-d", help="Directory containing images"),
    output: Path = typer.Option(..., "--output-dir", "-o", help="Path to write raw_extractions.jsonl"),
    model: str | None = typer.Option(None, "--model", help="Model type"),
    batch_size: int | None = typer.Option(None, "--batch-size", help="Images per extraction batch"),
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
        max_num_seqs=max_num_seqs,
        verbose=verbose,
        debug=debug,
        config_path=config,
    )


if __name__ == "__main__":
    app()
