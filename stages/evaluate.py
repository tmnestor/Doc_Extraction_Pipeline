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

from .io import read_jsonl, write_jsonl

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
    scored = [r for r in eval_results if "median_f1" in r and not r.get("error")]
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


def run(
    cleaned_extractions_path: Path,
    ground_truth: Path,
    output_dir: Path,
    *,
    inference_seconds: float | None = None,
) -> Path:
    """Evaluate cleaned extractions against ground truth.

    For each record in cleaned_extractions.jsonl:
      1. Find the page's answer in the key
      2. Score its fields (common.scoring.score_page)
      3. Write per-image evaluation results to evaluation_results.jsonl

    Args:
        cleaned_extractions_path: Path to cleaned_extractions.jsonl from Stage 3.
        ground_truth: The answer key, a JSON lines file.
        output_dir: Directory for evaluation output files.
        inference_seconds: GPU inference wall-clock seconds (extract phase
            elapsed time, set by entrypoint.sh). Used for throughput in the
            Execution Summary. Falls back to sum of per-image processing_time
            when not provided.

    Returns:
        Path to the written evaluation_results.jsonl.
    """
    # Read cleaned extractions
    records = read_jsonl(cleaned_extractions_path)
    if not records:
        msg = f"No records found in {cleaned_extractions_path}"
        raise FileNotFoundError(msg)

    logger.info(
        "Read %d cleaned extractions from %s",
        len(records),
        cleaned_extractions_path,
    )

    answers = scoring.load_answer_key(Path(ground_truth))

    # Evaluate each record
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_results: list[dict[str, Any]] = []
    failed: list[str] = []
    unanswered: list[str] = []

    for record in records:
        image_name = record["image_name"]
        error = record.get("error")

        if error:
            failed.append(image_name)
            eval_results.append(
                {
                    "image_name": image_name,
                    "document_type": record["document_type"],
                    "error": error,
                    "overall_accuracy": 0,
                    "median_f1": 0,
                }
            )
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

        eval_results.append(
            {
                "image_name": image_name,
                "document_type": record["document_type"],
                **evaluation,
            }
        )

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
    _print_summary_table(eval_results, records, output_dir, inference_seconds, scores)

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


def _print_summary_table(
    eval_results: list[dict[str, Any]],
    cleaned_records: list[dict[str, Any]],
    output_dir: Path,
    inference_seconds: float | None,
    scores: Headline,
) -> None:
    """Render an Execution Summary table and document-type breakdown.

    Mimics the pre-staged-pipeline `cli.print_summary` output so runs still
    end with the familiar throughput/accuracy summary.

    Args:
        inference_seconds: GPU inference wall-clock (extract phase elapsed).
            When provided, used for both "Inference Time" and throughput.
            Falls back to sum of per-image processing_time (correct for
            single-GPU sequential, but inflated for multi-GPU DP).
    """
    console = Console()

    num = len(eval_results)
    total_processing = sum(r.get("processing_time", 0.0) for r in cleaned_records)

    # Prefer the GPU stages' own timing when they recorded it: the
    # entrypoint measures from outside the process, so its number includes
    # engine startup, and under data parallelism the sum of per-image times
    # is total COMPUTE rather than elapsed.
    from common.stage_timing import read_stage_timings, resolve_timing

    resolved = resolve_timing(
        read_stage_timings(output_dir.parent),
        entrypoint_seconds=inference_seconds,
        per_image_total=total_processing,
        images=num,
    )
    timing_summary = resolved.summary
    inference_time = resolved.inference_seconds
    throughput = (num / inference_time * 60.0) if inference_time and inference_time > 0 else None

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
    # "not recorded" beats 0.0s: a fabricated rate reads as a measurement.
    table.add_row("Inference Time", f"{inference_time:.1f}s" if inference_time else "not recorded")
    table.add_row("Throughput", f"{throughput:.2f} images/min" if throughput else "not recorded")
    table.add_row("Timing Source", resolved.source)
    if timing_summary is not None:
        # Only shown when the GPU stages recorded their own timing. The
        # entrypoint's number bundles engine startup in and knows nothing
        # about data parallelism, so these rows would be misleading if
        # derived from it.
        table.add_row("Execution Mode", timing_summary.execution_mode)
        table.add_row("Total Wall Clock", f"{timing_summary.wall_clock:.1f}s")
        table.add_row("Engine Startup", f"{timing_summary.startup_seconds:.1f}s")
    for metric, value in score_rows(scores):
        table.add_row(metric, value)
    table.add_row("Output Directory", str(output_dir))

    console.print()
    console.print(table)

    if doc_types:
        console.print("\n[bold]Document Types:[/bold]")
        for doc_type, count in sorted(doc_types.items()):
            console.print(f"  {doc_type}: {count}")
