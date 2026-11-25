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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.config_schema import ConfigError
from common.engine_calls import EngineWatch, in_engine_calls
from common.pipeline_ops import check_processor
from common.settings import Settings
from common.stage_settings import StageSettings, run_settings
from common.vllm_dp import RecordSink

from . import records as records_of
from .carry_on import Work, carry_on
from .io import in_the_order, read_completed_images, read_input, start_afresh

logger = logging.getLogger(__name__)


def run(output_path: Path, checked: "Checked") -> Path:
    """Extract fields from classified images, write raw_extractions.jsonl.

    Writes the **raw model response string** per image, not parsed/cleaned
    data.  The clean stage handles parsing.

    A run carries on from what an earlier one left (``stages.carry_on``): pages already read are not
    read again, a page an earlier run could not read is, and so is a page given another document type
    since. The file is left in the order pages are read in, however the run went.

    Args:
        output_path: Path to write raw_extractions.jsonl.
        checked: The run's settings and the pages' classifications, checked: what ``checked`` gave.

    Returns:
        Path to the written raw_extractions.jsonl.
    """
    from common.extraction_sort import filter_skip_labels, sort_for_extraction

    app_cfg, classifications = checked.settings, checked.classifications

    # Pages of a type the settings say not to read are set apart. The rest are put one document type
    # after another, so that the engine reuses its work on a type's prompt.
    to_read, set_apart = filter_skip_labels(classifications, app_cfg.extraction_skip_labels)
    in_order = sort_for_extraction(to_read, app_cfg.extraction_order, secondary_sort=app_cfg.secondary_sort)
    logger.info(
        "Sorted %d records by extraction_order (secondary=%s)", len(in_order), app_cfg.secondary_sort
    )
    # The order of the output: the pages set apart, as they were classified, then the pages read.
    order = [page["image_name"] for page in set_apart] + [page["image_name"] for page in in_order]

    # Two kinds of page are read again: a page an earlier run could not read, and a page that has been
    # given another document type since it was read, whose record is of the wrong fields.
    now_typed = {c["image_name"]: c["document_type"] for c in classifications}

    def retyped(record: dict[str, Any]) -> bool:
        name = record.get("image_name")
        return name in now_typed and record.get("document_type") != now_typed[name]

    def taken_out(records: list[dict[str, Any]]) -> None:
        could_not = sum(1 for record in records if records_of.failed(record))
        logger.info(
            "%d page(s) will be read again: %d an earlier run could not read, %d given another "
            "document type since",
            len(records),
            could_not,
            len(records) - could_not,
        )

    # One engine on each card where the model allows it, else one engine across the cards. Either way the
    # pages are read by extract_pages; only where the engine is started differs.
    carried = carry_on(
        app_cfg,
        Work(
            stage="extract",
            output=output_path,
            items=in_order,
            name_of=lambda page: page["image_name"],
            read=extract_pages,
            order=order,
            read_again=lambda record: records_of.failed(record) or retyped(record),
            taken_out=taken_out,
            timing=output_path.parent,
        ),
    )

    # Each page set apart has a record too, so that the stages after this one see one record a page.
    # They are added after the reading and not before it: a first run with such pages is then not
    # taken for one that carried on. The order above still puts them first.
    held = read_completed_images(output_path)
    left_unread = [records_of.left_unread(page) for page in set_apart if page["image_name"] not in held]
    if left_unread:
        logger.info("Skipping %d records with skip labels", len(left_unread))
        in_the_order(output_path, order, adding=left_unread)

    if carried.arrangement is None:
        return output_path

    records = carried.records
    # The model's time, apart from starting the engine: the sum of the pages' times, and with an engine on
    # each card the slowest card's sum.
    logger.info("Extraction complete: %d images in %.1fs", len(records), carried.model_seconds)
    logger.info("Wrote %d extractions to %s", len(records), output_path)
    could_not = sum(1 for record in records if records_of.failed(record))
    if could_not:
        logger.warning(
            "%d of %d page(s) could not be read. Each is recorded with its error, and is read again "
            "the next time the stage is run.",
            could_not,
            len(records),
        )
    return output_path


@dataclass(frozen=True)
class Checked:
    """What the stage is run with, read and checked."""

    settings: Settings
    # The pages' classification records, as the classify stage left them.
    classifications: list[dict[str, Any]]


def checked(classifications_path: Path, settings: Settings) -> Checked:
    """The run's settings and the pages' classifications, with what reading the fields depends on checked.

    The one place the stage works out what it is run with. What it gives is what ``run`` is handed:
    nothing is read or worked out a second time.

    Args:
        classifications_path: Path to classifications.jsonl from Stage 1.
        settings: The run's settings.

    Raises:
        ConfigError: With what is wrong, where, what is expected and how to mend it. A
            classifications file that is not there, or cannot be read, is one such.
    """
    classifications = read_input(classifications_path, written_by="classify")
    logger.info("Read %d classifications from %s", len(classifications), classifications_path)
    check_settings(settings, classifications)
    return Checked(settings, classifications)


def check(task: StageSettings) -> Checked:
    """Read and check everything the stage will use, and give it for ``start``. Nothing is written.

    Raises:
        ConfigError: With what is wrong, where, what is expected and how to mend it.
    """
    found = checked(task.classifications, run_settings(task))
    check_processor(found.settings)
    return found


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
                records_of.read(
                    outcome.item,
                    reply=outcome.reply,
                    seconds=outcome.seconds,
                    prompt=_prompt_name(kind),
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


BANNER = "Stage 2: extract, the fields of each page (GPU)"


def start(task: StageSettings, checked: Checked) -> str:
    """Run the stage for a task with what ``check`` gave, and give the line that says it is done."""
    start_afresh(task.clear_previous_output, task.raw_extractions)
    started = time.time()
    run(task.raw_extractions, checked)
    return f"Extraction complete ({int(time.time() - started)}s)."
