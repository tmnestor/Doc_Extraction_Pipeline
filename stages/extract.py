"""Stage 2: Field extraction (GPU).

Reads classifications.jsonl, runs type-specific extraction on each image,
and writes RAW model responses to raw_extractions.jsonl.

The raw response is the most valuable debugging artifact -- you can re-parse,
re-clean, and re-evaluate without touching the GPU.

Supports resumption: if the output file already has partial results from a
crashed run, skips already-processed images and appends.

Started by ``python -m stages`` with ``KFP_TASK`` naming it:

    KFP_TASK=extract model=<a registered model> image_dir=<folder> output=<folder> bash entrypoint.sh
"""

import logging
import time
from pathlib import Path
from typing import Any

from common.config_schema import ConfigError
from common.engine_calls import EngineWatch, in_engine_calls
from common.vllm_dp import RecordSink

from .io import (
    Journal,
    StreamingJsonlWriter,
    append_jsonl,
    failed,
    read_completed_images,
    read_jsonl,
    take_out,
)

logger = logging.getLogger(__name__)


def run(
    classifications_path: Path,
    image_dir: Path,
    output_path: Path,
    *,
    model_type: str | None = None,
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
        config_path: Optional path to run_config.yml.

    Returns:
        Path to the written raw_extractions.jsonl.
    """
    from common.settings import Settings
    from common.extraction_sort import filter_skip_labels, sort_for_extraction

    # Read classifications
    classifications = read_jsonl(classifications_path)
    if not classifications:
        msg = f"No classifications found in {classifications_path}"
        raise FileNotFoundError(msg)

    logger.info("Read %d classifications from %s", len(classifications), classifications_path)

    cli_args: dict[str, Any] = {"data_dir": str(image_dir), "output_dir": str(output_path.parent)}
    if model_type is not None:
        cli_args["model_type"] = model_type

    app_cfg = Settings.load(cli_args, config_path=config_path)
    check_settings(app_cfg, classifications)

    # What an earlier run had read before it stopped is added to the output first, in the order the
    # stage reads pages in, and is then carried on from like any other earlier output.
    journal = Journal(output_path)
    in_order = sort_for_extraction(
        classifications, app_cfg.extraction_order, secondary_sort=app_cfg.secondary_sort
    )
    recovered = journal.recover([c["image_name"] for c in in_order])
    if recovered:
        logger.info("Recovered %d page(s) an earlier run had read before it stopped", len(recovered))
        from common.stage_timing import note_recovered

        note_recovered(output_path.parent, stage="extract", records=recovered)

    # Two kinds of page are read again, and their records taken out of the output so that they are not
    # among the pages already done: a page an earlier run could not read, and a page that has been
    # given another document type since it was read, whose record is of the wrong fields.
    now_typed = {c["image_name"]: c["document_type"] for c in classifications}

    def retyped(record: dict[str, Any]) -> bool:
        name = record.get("image_name")
        return name in now_typed and record.get("document_type") != now_typed[name]

    read_again = take_out(output_path, lambda record: failed(record) or retyped(record))
    if read_again:
        logger.info(
            "%d page(s) will be read again: %d an earlier run could not read, %d given another "
            "document type since",
            len(read_again),
            sum(1 for record in read_again if failed(record)),
            sum(1 for record in read_again if not failed(record)),
        )

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
    from common.stage_timing import inference_seconds_from_records, note_stage_timing
    from common.vllm_dp import read_on_the_cards

    stage_started = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Each record is kept in the journal as it is made, so that a run that stops part way keeps what it
    # had read, however the engines are arranged. The output is then added to, in the pages' order,
    # and never begun again: it may hold an earlier run's pages, or the skipped pages written above.
    with journal:
        read = read_on_the_cards(app_cfg, classifications, extract_pages, sink=journal)
    records, execution_mode = read.records, read.arrangement
    append_jsonl(output_path, records)
    journal.discard()

    wall_clock = time.time() - stage_started
    # The model's time, apart from starting the engine: the sum of the pages' times, and with an engine on
    # each card the slowest card's sum.
    logger.info(
        "Extraction complete: %d images in %.1fs",
        len(records),
        inference_seconds_from_records(records, fallback=wall_clock),
    )
    note_stage_timing(
        output_path.parent,
        stage="extract",
        records=records,
        wall_clock=wall_clock,
        execution_mode=execution_mode,
        carried_on=bool(completed),
        read_again=len(read_again),
    )
    logger.info("Wrote %d extractions to %s", len(records), output_path)
    could_not = sum(1 for record in records if failed(record))
    if could_not:
        logger.warning(
            "%d of %d page(s) could not be read. Each is recorded with its error, and is read again "
            "the next time the stage is run.",
            could_not,
            len(records),
        )
    return output_path


def check_settings(app_config: Any, classifications: list[dict[str, Any]]) -> None:
    """Everything reading the pages' fields depends on that can be wrong, checked before any engine starts.

    The pages to a call for each document type, and that every type the pages were given has a block
    in the schema file. Each is read again where it is used. Until 25 November 2026 that was the first
    time, after the engine had started, and a type with no block was recorded as a failed page for
    every page of that type.

    Raises:
        ConfigError: With what is wrong, where, what is expected and how to mend it.
    """
    from common.extraction_batching import read_batch_sizes
    from common.field_schema import get_field_schema

    read_batch_sizes(app_config.raw, config_path=app_config.config_path)
    schema = get_field_schema()
    not_read = {label.lower() for label in app_config.extraction_skip_labels}
    given = {
        str(c["document_type"]) for c in classifications if str(c["document_type"]).lower() not in not_read
    }
    unknown = sorted(kind for kind in given if schema.document_type(kind) is None)
    if unknown:
        raise ConfigError(
            f"What: the classifications give page(s) the document type(s) {', '.join(unknown)}, which "
            f"have no block in the schema file, so those pages cannot be read.\n"
            f"Where: config/field_definitions.yaml, under document_types.\n"
            f"Expected: one of {', '.join(schema.document_types)}.\n"
            f"How to fix: give each type a block with its fields and a prompt of the same name in the "
            f"model's prompt file, or name it under pipeline.extraction.skip_labels in "
            f"{app_config.config_path} if its pages are not to be read."
        )


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
    # One watch for the whole of this engine's reading: pages that fail one after another are counted
    # across its calls.
    watch = EngineWatch(app_config.stop_after_failed_requests)
    done, total = 0, len(classified)
    for batch in _plan_batches(classified, app_config):
        # A dense statement can take a while, so say what is in flight before it starts.
        logger.info(
            "[%d/%d] reading %d %s...", done + len(batch), total, len(batch), batch[0]["document_type"]
        )
        _read_batch(collected, processor, batch, watch)
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
    """What a record says its page was read with: the prompt of its type's name."""
    return document_type.lower()


def _read_batch(
    writer: RecordSink, processor: Any, batch: list[dict[str, Any]], watch: EngineWatch
) -> None:
    """Read a group of pages of one document type in one engine call, and write a record for each.

    A group that fails is read again one page at a time, so a single bad page costs its group only the
    shared call. A page that fails on its own is recorded with its error, so that it never ends the run
    (``common.engine_calls``).
    """
    from common import prompt_trace

    kind = batch[0]["document_type"]

    def ask(pages: list[dict[str, Any]]) -> list[str]:
        if len(pages) > 1:
            return processor.read_pages([page["image_path"] for page in pages], kind)
        # Names the page in the trace of what was sent to the model, when that trace is switched on.
        with prompt_trace.trace_context(
            image_name=pages[0]["image_name"],
            pipeline="information_extraction",
            label=f"extract_{kind.lower()}",
        ):
            return processor.read_pages([pages[0]["image_path"]], kind)

    for outcomes in in_engine_calls([batch], ask, watch=watch, what=kind):
        for outcome in outcomes:
            if outcome.error is not None:
                logger.error("Error extracting %s: %s", outcome.item["image_name"], outcome.error)
            writer.write(
                _record(
                    outcome.item,
                    raw_response=outcome.reply if outcome.error is None else "",
                    processing_time=outcome.seconds,
                    prompt_used=_prompt_name(kind) if outcome.error is None else "error",
                    error=outcome.error,
                )
            )


def _plan_batches(classifications: list[dict[str, Any]], app_config: Any) -> list[list[dict[str, Any]]]:
    """Group the pages into engine calls, by the sizes in ``pipeline.batching``."""
    from common.extraction_batching import plan_extraction_batches, read_batch_sizes

    batch_sizes = read_batch_sizes(app_config.raw, config_path=app_config.config_path)
    batches = plan_extraction_batches(classifications, batch_sizes=batch_sizes)
    logger.info(
        "Extraction batching: %d documents in %d calls (sizes: %s)",
        len(classifications),
        len(batches),
        batch_sizes,
    )
    return batches
