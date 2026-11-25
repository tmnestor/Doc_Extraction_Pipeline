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

from common.stage_settings import StageSettings

from . import records as records_of
from .carry_on import Work, carry_on
from .io import start_afresh

logger = logging.getLogger(__name__)


def run(
    image_dir: Path,
    output_path: Path,
    *,
    model_type: str | None = None,
    config_path: Path | None = None,
) -> Path:
    """Detect document types for all images, write classifications.jsonl.

    A run carries on from what an earlier one left (``stages.carry_on``): pages already classified are
    not asked about again, a page an earlier run could not classify is, and the file is left in the
    pages' order however the run went. A run with nothing left starts no engine.

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
    # Both filters are config-driven and MUST be passed: calling with the
    # directory alone silently ignored max_images and document_types, so a
    # 12-image request processed all 165 (fixed 2026-09-21).
    images = list(
        discover_images(
            image_dir,
            document_types=app_cfg.document_types,
            max_images=app_cfg.max_images,
        )
    )
    if not images:
        msg = f"No images found in {image_dir}"
        raise FileNotFoundError(msg)

    logger.info("Found %d images in %s", len(images), image_dir)

    def taken_out(records: list[dict[str, Any]]) -> None:
        logger.info("%d page(s) an earlier run could not classify will be asked about again", len(records))

    # One engine on each card where the model allows it, else one engine across the cards. Either way the
    # pages are read by classify_pages; only where the engine is started differs.
    carried = carry_on(
        app_cfg,
        Work(
            stage="classify",
            output=output_path,
            items=images,
            name_of=lambda page: page.name,
            read=classify_pages,
            taken_out=taken_out,
            timing=output_path.parent,
        ),
    )
    if carried.arrangement is None:
        return output_path

    records = carried.records
    logger.info(
        "Classification complete: %d images in %.1fs (%.2fs/image)",
        len(records),
        carried.model_seconds,
        carried.model_seconds / len(records) if records else 0,
    )
    logger.info("Wrote %d classifications to %s", len(records), output_path)
    _say_how_the_types_were_reached(records)
    could_not = sum(1 for record in records if records_of.failed(record))
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
            record = records_of.classified(outcome.item, result, outcome.seconds)
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
        return processor.unclassified(why, reply=outcome.reply)


BANNER = "Stage 1: classify, the document type of each page (GPU)"


def start(task: StageSettings) -> str:
    """Run the stage for a task, and give the line that says it is done."""
    assert task.image_dir is not None
    start_afresh(task.clear_previous_output, task.classifications)
    started = time.time()
    run(task.image_dir, task.classifications, model_type=task.model, config_path=task.config_path)
    return f"Classification complete ({int(time.time() - started)}s)."
