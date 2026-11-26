"""Stage 1: Document type classification (GPU).

Detects document types for all images and writes classifications.jsonl.

Started by ``python -m stages`` with ``KFP_TASK`` naming it:

    KFP_TASK=classify model=<a registered model> image_dir=<folder> output=<folder> bash entrypoint.sh
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.classification import BY_THE_DATASET
from common.errors import ConfigError
from common.engine_calls import EngineWatch, Outcome, groups_of, in_engine_calls
from common import extraction_batching
from common.field_schema import get_field_schema
from common.pipeline_ops import check_processor
from common.stage_settings import StageSettings
from common.vllm_dp import RecordSink

from . import records as records_of
from .carry_on import Work, carry_on
from .io import discover_images, no_pages_in, start_afresh

logger = logging.getLogger(__name__)


def run(output_path: Path, checked: "Checked", *, given_type: str | None = None) -> Path:
    """Detect document types for all images, write classifications.jsonl.

    When the job says what type every page is (``document_type``), each is given that type and no
    model is asked: no engine is started.

    A run carries on from what an earlier one left (``stages.carry_on``): pages already classified are
    not asked about again, a page an earlier run could not classify is, and the file is left in the
    pages' order however the run went. A run with nothing left starts no engine.

    Args:
        output_path: Path to write classifications.jsonl.
        checked: The run's settings as the stage loads its engine with them, and the pages to
            classify: what ``checked`` gave.
        given_type: The document type the job gives every page, or None to ask the model.

    Returns:
        Path to the written classifications.jsonl.
    """
    app_cfg, images = checked.settings, checked.images

    def taken_out(records: list[dict[str, Any]]) -> None:
        logger.info(
            "%d page(s) an earlier run could not classify will be %s",
            len(records),
            "asked about again" if given_type is None else "given their type",
        )

    if given_type is not None:
        # The pages' dataset says what they are. No model is asked, and the stage carries on the one
        # way: a page the output holds keeps its record, a page an earlier run could not classify is
        # given its type, and the file is left in the pages' order. The record is the one a page the
        # model was asked about has, with nothing for a reply, so that every stage after this reads
        # it as it reads any other.
        given = carry_on(
            app_cfg,
            Work(
                stage="classify",
                output=output_path,
                items=images,
                name_of=lambda page: page.name,
                tell=lambda page: records_of.given_its_type(str(page), given_type, BY_THE_DATASET),
                taken_out=taken_out,
            ),
        )
        logger.info(
            "%d page(s) given the type %s by the dataset: the model is not asked",
            len(given.records),
            given_type,
        )
        return output_path

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


@dataclass(frozen=True)
class Checked:
    """What the stage is run with, read and checked."""

    settings: StageSettings
    # The pages to classify: those of the folder the settings select, in the order of their names.
    images: list[Path]


def checked(image_dir: Path, settings: StageSettings) -> Checked:
    """The run's settings and the pages to classify, each read and checked.

    The one place the stage works out what it is run with. What it gives is what ``run`` is handed:
    nothing is read or worked out a second time.

    Args:
        image_dir: The folder of pages.
        settings: The run's settings.

    Raises:
        ConfigError: With what is wrong, where, what is expected and how to mend it. A folder with no
            page to read is one such: until 26 November 2026 it ended the task with a traceback.
    """
    check_settings(settings)
    # Both filters are config-driven and MUST be passed: calling with the
    # directory alone silently ignored max_images and document_types, so a
    # 12-image request processed all 165 (fixed 2026-09-21).
    images = list(
        discover_images(
            image_dir,
            document_types=settings.document_types,
            max_images=settings.max_images,
        )
    )
    if not images:
        selected = (
            f" of the types the job selects ({', '.join(settings.document_types)}, in the variable "
            f"document_types)"
            if settings.document_types
            else ""
        )
        raise no_pages_in(image_dir, selected)
    logger.info("Found %d images in %s", len(images), image_dir)
    return Checked(settings, images)


def check(task: StageSettings) -> Checked:
    """Read and check everything the stage will use, and give it for ``start``. Nothing is written.

    Raises:
        ConfigError: With what is wrong, where, what is expected and how to mend it.
    """
    assert task.image_dir is not None
    found = checked(task.image_dir, task)
    # Given the type of every page, the stage asks no model: none is looked for.
    if task.given_type is None:
        check_processor(found.settings)
    return found


def check_settings(app_config: Any) -> None:
    """Everything classifying reads that can be wrong, checked before any engine is started.

    The classification file (its questions, its column roles and its rules), and the schema file, which
    holds the fallback type.
    Until 25 November 2026 the classification file was first read when the first reply was turned into
    a type, inside the engine call. A mistake in a rule was then taken for a failed call, and every
    page was given the fallback type.

    Raises:
        ConfigError: With what is wrong, where, what is expected and how to mend it.
    """
    from common import classification

    # The schema file is read with it: its fallback type is checked to be a document type there.
    get_field_schema()
    classification.check()


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
        extraction_batching.PAGES_PER_CALL,
        sink,
        watch=EngineWatch(),
    )


def _classify_images(
    processor: Any,
    image_paths: list[str],
    batch_size: int,
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


def start(task: StageSettings, checked: Checked) -> str:
    """Run the stage for a task with what ``check`` gave, and give the line that says it is done."""
    start_afresh(task.clear_previous_output, task.classifications)
    started = time.time()
    run(task.classifications, checked, given_type=task.given_type)
    return f"Classification complete ({int(time.time() - started)}s)."
