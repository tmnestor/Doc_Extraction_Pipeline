"""Stage 4: Evaluation against ground truth (CPU).

Reads cleaned_extractions.jsonl, compares against the answer key (JSON lines),
computes F1 scores, and writes evaluation_results.jsonl.

No GPU needed -- this stage runs on CPU nodes.

Started by ``python -m stages`` with ``KFP_TASK`` naming it:

    KFP_TASK=evaluate ground_truth=<answer key> output=<folder> bash entrypoint.sh
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from common import scoring
from common.field_schema import get_field_schema
from common.stage_settings import StageSettings
from common.stage_timing import RunTime, read_stage_timings, run_time

from . import records as records_of
from .io import read_input, start_afresh, write_jsonl

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Headline:
    """A run's headline scores, with the pages they leave out counted beside them.

    A page whose extraction failed has no fields to score, and a page with no answer in the key cannot
    be scored at all. Neither is in ``mean_f1`` or ``median_f1``. Until 24 November 2026 nothing said how
    many such pages there were, so a run that lost twenty pages reported the mean of the rest as if it
    were the run's.
    """

    pages: int
    scored: int
    # Pages the extract stage recorded an error for.
    failed: int
    # Pages with no answer in the key.
    unanswered: int
    # Over the scored pages: the mean of their mean F1, and the mean of their median F1.
    mean_f1: float
    median_f1: float
    # The mean F1 with every failed page counted as 0. The same as mean_f1 when none failed. A page
    # with no answer is in neither: there is nothing to say it scored.
    mean_f1_counting_failures: float


def headline(eval_results: list[dict[str, Any]], *, failed: int, unanswered: int) -> Headline:
    """The headline scores of a run's evaluation records.

    Args:
        eval_results: The records the stage writes, one a page.
        failed: How many of them are pages whose extraction failed.
        unanswered: How many of them are pages with no answer in the key.
    """
    scored = [r for r in eval_results if "median_f1" in r and not records_of.failed(r)]
    total_mean = 0.0
    total_median = 0.0
    for record in scored:
        total_mean += record.get("overall_accuracy", 0.0)
        total_median += record.get("median_f1", 0.0)
    return Headline(
        pages=len(eval_results),
        scored=len(scored),
        failed=failed,
        unanswered=unanswered,
        mean_f1=total_mean / len(scored) if scored else 0.0,
        median_f1=total_median / len(scored) if scored else 0.0,
        mean_f1_counting_failures=total_mean / (len(scored) + failed) if scored or failed else 0.0,
    )


def run(checked: "Checked", output_dir: Path) -> Path:
    """Evaluate cleaned extractions against ground truth.

    For each record in cleaned_extractions.jsonl:
      1. Find the page's answer in the key
      2. Score its fields (common.scoring.score_page)
      3. Write per-image evaluation results to evaluation_results.jsonl

    Args:
        checked: The cleaned records of Stage 3 and the answer key, read: what ``checked`` gave.
        output_dir: Directory for evaluation output files.

    Returns:
        Path to the written evaluation_results.jsonl.
    """
    records, answers, ground_truth = checked.records, checked.answers, checked.ground_truth

    # Evaluate each record
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_results: list[dict[str, Any]] = []
    failed: list[str] = []
    unanswered: list[str] = []

    for record in map(records_of.as_cleaned, records):
        image_name = record["image_name"]

        if records_of.failed(record):
            failed.append(image_name)
            eval_results.append(records_of.scored_but_failed(record))
            continue

        answer = scoring.find_answer(answers, record.get("image_path", ""))
        if answer:
            evaluation = scoring.score_page(
                dict(record.get("extracted_data", {})),
                answer,
                record["document_type"],
            )
        else:
            unanswered.append(image_name)
            evaluation = {"error": f"No ground truth for {image_name}", "overall_accuracy": 0}

        eval_results.append(records_of.scored(record, evaluation))

        if evaluation and "median_f1" in evaluation:
            logger.debug(
                "%s: median_f1=%.3f, overall_accuracy=%.3f",
                image_name,
                evaluation.get("median_f1", 0),
                evaluation.get("overall_accuracy", 0),
            )

    # Write evaluation results
    output_path = output_dir / "evaluation_results.jsonl"
    count = write_jsonl(output_path, eval_results)
    logger.info("Wrote %d evaluation results to %s", count, output_path)

    # Summary statistics
    scores = headline(eval_results, failed=len(failed), unanswered=len(unanswered))
    if scores.scored:
        logger.info(
            "Summary: %d of %d images scored, avg F1 (median)=%.3f, avg F1 (mean)=%.3f",
            scores.scored,
            scores.pages,
            scores.median_f1,
            scores.mean_f1,
        )
    else:
        logger.warning("No images scored -- check ground truth alignment")
    # The pages the scores leave out are named, so that a mean over part of a run is not read as the run's.
    if failed:
        logger.warning(
            "%d page(s) failed extraction and are in no score above: %s. Counting each as 0, the mean "
            "F1 is %.3f.",
            len(failed),
            _some(failed),
            scores.mean_f1_counting_failures,
        )
    if unanswered:
        logger.warning(
            "%d page(s) have no answer in %s and are not scored: %s. Check that the answer key is the "
            "one of this set of pages.",
            len(unanswered),
            ground_truth,
            _some(unanswered),
        )
    # Rich Execution Summary table (equivalent to pre-staged-pipeline output)
    _print_summary_table(eval_results, records, output_dir, scores)

    return output_path


def score_rows(scores: Headline) -> list[tuple[str, str]]:
    """The summary table's rows of scores. Pages left out of the scores have a row of their own."""
    rows = [("Images Scored", f"{scores.scored} of {scores.pages}")]
    if scores.failed:
        rows.append(("Failed Extraction", f"{scores.failed} (in no score below)"))
    if scores.unanswered:
        rows.append(("No Answer In Key", f"{scores.unanswered} (not scored)"))
    if scores.scored:
        rows.append(("Avg F1 (mean)", f"{scores.mean_f1:.3f}"))
        rows.append(("Avg F1 (median)", f"{scores.median_f1:.3f}"))
    if scores.failed:
        rows.append(("Avg F1 (mean), failed as 0", f"{scores.mean_f1_counting_failures:.3f}"))
    return rows


def _some(names: list[str], most: int = 10) -> str:
    """The first few names, and how many more there are."""
    shown = ", ".join(names[:most])
    return shown if len(names) <= most else f"{shown} and {len(names) - most} more"


# What a stage that does not read pages reads, for its row of time.
_READS = {"split": "photographs"}


def time_rows(time: RunTime, pages: int) -> list[tuple[str, str]]:
    """The summary table's rows of time. Every figure says which stages it is of, and where it is from.

    The model's time is given for each stage and for the stages together, and the rate is of the stages
    together: pages a minute through every stage that used the cards.
    """
    # A stage that does not read pages, such as split, has a row of its own and is in no figure below
    # it: the rate is of pages, and a photograph is not a page.
    rows = [
        (
            f"{t.stage.capitalize()}, model time ({t.images} {_READS.get(t.stage, 'items')}, not in the rate)",
            f"{t.inference_seconds:.1f}s",
        )
        for t in time.others
    ]
    rows += [(f"{t.stage.capitalize()}, model time", f"{t.inference_seconds:.1f}s") for t in time.stages]
    of = " + ".join(t.stage for t in time.stages)
    # "not recorded" beats 0.0s: a rate made from nothing reads as a measurement.
    seconds = time.model_seconds
    rows.append(
        (f"Model Time ({of})" if of else "Model Time", f"{seconds:.1f}s" if seconds else "not recorded")
    )
    rows.append(("Throughput", f"{pages / seconds * 60.0:.2f} images/min" if seconds else "not recorded"))
    rows.append(("Timing Source", time.source))
    if time.stages:
        # Shown only when the stages recorded their own time: they are not known otherwise.
        rows.append(("Execution Mode", time.execution_mode))
        rows.append(("Total Wall Clock", f"{time.wall_clock:.1f}s"))
        rows.append(("Engine Startup", f"{time.startup_seconds:.1f}s"))
    return rows


def _print_summary_table(
    eval_results: list[dict[str, Any]],
    cleaned_records: list[dict[str, Any]],
    output_dir: Path,
    scores: Headline,
) -> None:
    """Print the spread of each field's scores, the summary table and the count of each document type."""
    console = Console()

    num = len(eval_results)
    total_processing = sum(r.get("processing_time", 0.0) for r in cleaned_records)
    time = run_time(
        read_stage_timings(output_dir.parent),
        per_image_total=total_processing,
        pages=num,
        read=num,
    )

    doc_types: dict[str, int] = {}
    for r in eval_results:
        dt = r.get("document_type", "UNKNOWN")
        doc_types[dt] = doc_types.get(dt, 0) + 1

    # Per-field dispersion, printed BEFORE the summary so the headline mean
    # is read with the spread already in view. Built from field_scores the
    # evaluator already stores: no scorer change involved.
    from common.field_report import field_distributions, render_field_table

    render_field_table(field_distributions(eval_results))

    table = Table(title="Execution Summary", show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Images Processed", str(num))
    for metric, value in time_rows(time, num):
        table.add_row(metric, value)
    for metric, value in score_rows(scores):
        table.add_row(metric, value)
    table.add_row("Output Directory", str(output_dir))

    console.print()
    console.print(table)

    if doc_types:
        console.print("\n[bold]Document Types:[/bold]")
        for doc_type, count in sorted(doc_types.items()):
            console.print(f"  {doc_type}: {count}")


BANNER = "Stage 4: evaluate, the fields scored against the answer key (CPU)"


@dataclass(frozen=True)
class Checked:
    """What the stage is run with, read and checked."""

    # The cleaned records, as the clean stage left them.
    records: list[dict[str, Any]]
    # The answer key: a page's file name to its answer.
    answers: dict[str, dict[str, Any]]
    ground_truth: Path


def checked(cleaned_extractions_path: Path, ground_truth: Path) -> Checked:
    """The cleaned records and the answer key, read, and the schema file with its thresholds checked.

    The one place the stage reads what it is run with. What it gives is what ``run`` is handed.

    Raises:
        ConfigError: With what is wrong, where, what is expected and how to mend it. A file of
            cleaned records, or an answer key, that is not there or cannot be read is one such: until
            26 November 2026 each ended the task with a traceback.
    """
    get_field_schema()
    records = read_input(cleaned_extractions_path, written_by="clean")
    logger.info("Read %d cleaned extractions from %s", len(records), cleaned_extractions_path)
    return Checked(records, scoring.load_answer_key(Path(ground_truth)), Path(ground_truth))


def check(task: StageSettings) -> Checked:
    """Read and check everything the stage will use, and give it for ``start``. Nothing is written.

    Raises:
        ConfigError: With what is wrong, where, what is expected and how to mend it.
    """
    assert task.ground_truth is not None
    return checked(task.cleaned_extractions, task.ground_truth)


def start(task: StageSettings, checked: Checked) -> str:
    """Run the stage for a task with what ``check`` gave, and give the line that says it is done."""
    task.evaluation_dir.mkdir(parents=True, exist_ok=True)
    start_afresh(task.clear_previous_output, task.evaluation_dir / "evaluation_results.jsonl")
    run(checked, task.evaluation_dir)
    return "Evaluation complete."
