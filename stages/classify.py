"""Stage 1: Document type classification (GPU).

Detects document types for all images and writes classifications.jsonl.

Started by ``python -m stages`` with ``KFP_TASK`` naming it:

    KFP_TASK=classify model=<a registered model> image_dir=<folder> output=<folder> bash entrypoint.sh
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any

from common.config_schema import ConfigError
from common.engine_calls import EngineWatch, Outcome, groups_of, in_engine_calls
from common.vllm_dp import RecordSink
from common.extraction_batching import read_classify_batch_size

from .io import Journal, append_jsonl, failed, read_completed_images, take_out, write_jsonl

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

    cli_args: dict[str, Any] = {"output_dir": str(output_path.parent)}
    if model_type is not None:
        cli_args["model_type"] = model_type

    app_cfg = Settings.load(cli_args, config_path=config_path).for_stage("classify")
    check_settings(app_cfg)
    config = app_cfg
    # Discover images
    # Both filters are config-driven and MUST be passed: calling with the
    # directory alone silently ignored max_images and document_types, so a
    # 12-image request processed all 165 (fixed 2026-09-21).
    images = list(
        discover_images(
            image_dir,
            document_types=config.document_types,
            max_images=config.max_images,
        )
    )
    if not images:
        msg = f"No images found in {image_dir}"
        raise FileNotFoundError(msg)

    logger.info("Found %d images in %s", len(images), image_dir)

    # Resume: skip images already classified in a prior (possibly crashed) run,
    # and APPEND below instead of truncating. A clean-slate re-run is the opt-in
    # (clear the output first: the entrypoint does this for full re-runs). This
    # mirrors the extract stage and runs BEFORE the model load, so a fully
    # resumed run skips the GPU entirely.
    # What an earlier run had classified before it stopped is added to the output first, and is then
    # carried on from like any other earlier output.
    journal = Journal(output_path)
    recovered = journal.recover([image.name for image in images])
    if recovered:
        logger.info("Recovered %d page(s) an earlier run had classified before it stopped", len(recovered))
        from common.stage_timing import note_recovered

        note_recovered(output_path.parent, stage="classify", records=recovered)
    # A page an earlier run could not classify is asked about again: its record is taken out of the
    # output, so that it is not among the pages already done.
    read_again = take_out(output_path, failed)
    if read_again:
        logger.info(
            "%d page(s) an earlier run could not classify will be asked about again", len(read_again)
        )
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
    from common.stage_timing import inference_seconds_from_records, note_stage_timing
    from common.vllm_dp import read_on_the_cards

    started = time.time()
    # Each record is kept in the journal as it is made; the output is written below, in the pages' order.
    with journal:
        read = read_on_the_cards(app_cfg, images, classify_pages, sink=journal)
    records = read.records
    wall_clock = time.time() - started

    elapsed = inference_seconds_from_records(records, fallback=wall_clock)
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
    journal.discard()
    logger.info("Wrote %d classifications to %s", count, output_path)
    note_stage_timing(
        output_path.parent,
        stage="classify",
        records=records,
        wall_clock=wall_clock,
        execution_mode=read.arrangement,
        carried_on=resuming,
        read_again=len(read_again),
    )
    _say_how_the_types_were_reached(records)
    could_not = sum(1 for record in records if failed(record))
    if could_not:
        logger.warning(
            "%d of %d page(s) could not be classified. Each is recorded with its error and the "
            "fallback type, and is asked about again the next time the stage is run.",
            could_not,
            len(records),
        )

    return output_path


def _say_how_the_types_were_reached(records: list[dict[str, Any]]) -> None:
    """Count the pages by how each came by its type, and warn of those given the fallback type.

    A page given the fallback type has, in its record, a type like any other. Until 25 November 2026
    nothing else in the record said so, and nothing counted such pages: a reply the rules could make
    nothing of was silent.
    """
    reached: dict[str, int] = {}
    for record in records:
        how = record.get("type_from") or "not recorded"
        reached[how] = reached.get(how, 0) + 1
    logger.info(
        "How the types were reached: %s",
        ", ".join(f"{how} {count}" for how, count in sorted(reached.items())),
    )
    fallbacks = [
        record["image_name"]
        for record in records
        if str(record.get("type_from", "")).startswith("fallback")
    ]
    if fallbacks:
        logger.warning(
            "%d of %d page(s) were given the fallback type, since no type could be worked out for them: "
            "%s%s. Each record says why (type_from).",
            len(fallbacks),
            len(records),
            ", ".join(fallbacks[:10]),
            f" and {len(fallbacks) - 10} more" if len(fallbacks) > 10 else "",
        )


def check_settings(app_config: Any) -> None:
    """Everything classifying reads that can be wrong, checked before any engine is started.

    The pages to a call, the classification file (its questions, its column roles and its rules), and
    the fallback type. Each is read again where it is used. Until 25 November 2026 that was the first
    time: the pages to a call after the engine had started, and the classification file when the first
    reply was turned into a type, inside the engine call. A mistake in a rule was then taken for a
    failed call, and every page was given the fallback type.

    Raises:
        ConfigError: With what is wrong, where, what is expected and how to mend it.
    """
    from common import classification
    from common.field_schema import get_field_schema

    read_classify_batch_size(app_config.raw, config_path=app_config.config_path)
    classification.check()
    fallback = app_config.classification_fallback_type
    types = get_field_schema().document_types
    if not isinstance(fallback, str) or fallback.lower() not in types:
        raise ConfigError(
            f"What: the fallback type {fallback!r} is not a document type, so a page given it could not "
            f"be read by the extract stage.\n"
            f"Where: {app_config.config_path} -> pipeline.classification.fallback_type\n"
            f"Expected: a document type of the schema file, in either case: {', '.join(sorted(types))}.\n"
            f"How to fix: set fallback_type to one of those, e.g. fallback_type: UNIVERSAL"
        )


def classify_pages(
    processor: Any, images: list[Path], app_config: Any, *, sink: RecordSink | None = None
) -> list[dict[str, Any]]:
    """Give each page its document type. The one place pages are classified, however engines are arranged.

    One engine across the cards calls this once with every page. With an engine on each card, each worker
    calls it with its share. It loads no model, opens no file and decides nothing about which pages are
    already done: the stage's ``run`` does those.

    Each record carries a processing time. Inside a shared engine call the requests overlap, so a page's
    own time is not separable: each page is charged an equal share of its call.

    Args:
        processor: The document processor.
        images: The pages, in order.
        app_config: The run's settings.
        sink: Where to write each record as soon as it is made, if anywhere.

    Returns:
        One classification record per page, in the order given.
    """
    return _classify_images(
        processor,
        [str(image) for image in images],
        read_classify_batch_size(app_config.raw, config_path=app_config.config_path),
        app_config.verbose,
        sink,
        watch=EngineWatch(app_config.stop_after_failed_requests),
    )


def _record(image_path: str, result: dict[str, Any]) -> dict[str, Any]:
    record = {
        "image_path": image_path,
        "image_name": Path(image_path).name,
        "document_type": result["document_type"],
        # How the type was reached: by a rule, by the file's default, or as the fallback type and why.
        "type_from": result.get("type_from"),
        "confidence": result.get("confidence", 1.0),
        "raw_response": result.get("raw_response", ""),
        "prompt_used": result.get("prompt_used", "detection"),
    }
    if result.get("error"):
        record["error"] = result["error"]
    return record


def _classify_images(
    processor: Any,
    image_paths: list[str],
    batch_size: int,
    verbose: bool,
    sink: RecordSink | None = None,
    *,
    watch: EngineWatch,
) -> list[dict[str, Any]]:
    """Classify pages, ``batch_size`` to an engine call.

    A page is one independent request, so the engine runs the pages of a call at once. A call that fails
    is tried again one page at a time, and a page that still fails is given the fallback type and
    carries its error, so that one page never ends the run (``common.engine_calls``).

    Only asking the model is inside the call that may be tried again. A reply is turned into a type
    afterwards, so that a reply the model has given is never asked for twice.

    Args:
        processor: The document processor (orchestrator).
        image_paths: Pages to classify, in order.
        batch_size: Pages per engine call. 1 means one call a page.
        verbose: Whether each page's type is also printed as it is found.
        sink: Where to write each record as soon as it is made, if anywhere.
        watch: Counts the pages that fail one after another, and stops the stage when the engine has
            stopped answering.

    Returns:
        One classification record per page, in input order, each with its share of its call's time.
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
        groups_of(image_paths, batch_size), processor.ask_document_types, watch=watch, what="pages"
    )
    for outcomes in calls:
        for outcome in outcomes:
            result = _classified(processor, outcome)
            record = {**_record(outcome.item, result), "processing_time": outcome.seconds}
            records.append(record)
            if sink is not None:
                sink.write(record)
            name = Path(outcome.item).name
            logger.info("[%d/%d] %s -> %s", len(records), total, name, result["document_type"])
            if verbose:
                sys.stdout.write(f"Detected document type: {name}: {result['document_type']}\n")
                sys.stdout.flush()
    return records


def _classified(processor: Any, outcome: Outcome[str, str]) -> dict[str, Any]:
    """A page's classification: from the model's reply, or the fallback type with why there is none.

    A setting that cannot be used is raised (``ConfigError``): it is the run's, not the page's. Anything
    else that goes wrong in reading one reply is that page's, and is recorded on it with the reply.
    """
    if outcome.reply is None:
        return processor.unclassified(outcome.error)
    try:
        return processor.classified(outcome.reply)
    except ConfigError:
        raise
    except Exception as err:  # noqa: BLE001 - one reply that cannot be read must not end the run
        why = f"{type(err).__name__}: {err}"
        logger.error("The reply about %s could not be read (%s)", Path(outcome.item).name, why)
        return {**processor.unclassified(why), "raw_response": outcome.reply}
