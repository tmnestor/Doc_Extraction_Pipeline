"""SROIE benchmark stage.

Runs the ICDAR 2019 SROIE Task 3 key-information benchmark through the
repo's own model stack, so the tokenizer and chat-template
handling under test are the ones production uses.

The stage is deliberately thin: everything model-independent lives in
``common/sroie/`` and is unit-tested without a GPU. Swapping the model is a
``bootstrap.model.type`` change in run_config.yml, which is what keeps runs of
different models comparable: same prompt, same parser, same scorer, byte for byte.

Run it through the entrypoint, never directly:

    KFP_TASK=sroie bash entrypoint.sh
"""

import logging
import time
from pathlib import Path
from typing import Any


from common.engine_calls import groups_of, in_engine_calls
from common.sroie.config import SroieSettings
from common.sroie.ground_truth import SroieRecord, load_sroie_split
from common.sroie.prompt import SROIE_PROMPT
from common.sroie.report import (
    execution_summary_rows,
    render_execution_summary,
    render_per_field_table,
    write_per_image_csv,
    write_summary_json,
)
from common.sroie.runner import benchmark_from_worker_responses
from common.stage_timing import inference_seconds_from_records
from common.sroie.scoring import MatchPolicy, score_records

logger = logging.getLogger(__name__)


def run(
    *,
    model_type: str | None = None,
    max_images: int | None = None,
    config_path: Path | None = None,
) -> Path:
    """Run the SROIE benchmark and write its artefacts.

    Args:
        model_type: Model type override. None = bootstrap.model.type.
        max_images: Score only the first N records. None = the whole split.
        config_path: Optional path to run_config.yml.

    Returns:
        Path to the summary JSON.

    Raises:
        SroieConfigError: If the pipeline.sroie block is missing a key.
        SroieDatasetError: If the split on disk is unusable.
    """
    from common.settings import Settings
    from common.vllm_dp import read_on_the_cards

    cli_args: dict[str, object] = {}
    if model_type:
        cli_args["model_type"] = model_type

    app_cfg = Settings.load(cli_args, config_path=config_path)
    settings = _settings(app_cfg)

    records = load_sroie_split(settings.data_dir)
    if max_images is not None:
        records = records[:max_images]
    logger.info("SROIE: %d records from %s", len(records), settings.data_dir)
    image_paths = [str(record.image_path) for record in records]

    # One engine on each card where the model allows it, else one engine across the cards. Either way the
    # receipts are read by read_sroie; only where the engine is started differs.
    started = time.time()
    read = read_on_the_cards(app_cfg, image_paths, read_sroie)
    whole = time.time() - started
    # The whole time is shown only with an engine on each card, where it and the model's time differ by
    # the engines' starting.
    wall_clock: float | None = whole if read.engines else None

    # The model's time, apart from starting the engine: the sum of the receipts' times, and with an
    # engine on each card the slowest card's sum.
    elapsed = inference_seconds_from_records(read.records, fallback=whole)
    return _finish(
        records=records,
        result=benchmark_from_worker_responses(records, read.records),
        settings=settings,
        model_name=app_cfg.model_type,
        elapsed=elapsed,
        execution_mode=read.arrangement,
        wall_clock=wall_clock,
    )


def _settings(app_config: Any) -> SroieSettings:
    """The pipeline.sroie block of the run's settings, which are already read."""
    return SroieSettings.from_raw(app_config.raw, config_path=app_config.config_path)


def read_sroie(processor: Any, image_paths: list[str], app_config: Any) -> list[dict[str, Any]]:
    """Ask the model for each receipt's four fields. The one place receipts are read.

    One engine across the cards calls this once with every receipt. With an engine on each card, each
    worker calls it with its share. Replies come back as the model wrote them: parsing and scoring happen
    once, in ``run``.

    Receipts go to the engine several to a call, by ``pipeline.sroie.batch_size``. A call that fails is
    tried again one receipt at a time, and a receipt that still fails is recorded with its error, so one
    receipt never ends the run.

    Args:
        processor: The document processor.
        image_paths: The receipts, in order.
        app_config: The run's settings.

    Returns:
        One record per receipt, in the order given: image_id, image_path, raw_response, error,
        processing_time.
    """
    from PIL import Image

    settings = _settings(app_config)

    def ask(paths: list[str]) -> list[str]:
        images = [Image.open(path).convert("RGB") for path in paths]
        try:
            return processor.ask(images, [SROIE_PROMPT] * len(images), settings.max_new_tokens)
        finally:
            for image in images:
                image.close()

    records: list[dict[str, Any]] = []
    total = len(image_paths)
    calls = in_engine_calls(groups_of(image_paths, settings.batch_size), ask, what="receipts")
    for outcomes in calls:
        for outcome in outcomes:
            if outcome.error is not None:
                logger.error("%s FAILED: %s", Path(outcome.item).stem, outcome.error)
            records.append(
                {
                    "image_id": Path(outcome.item).stem,
                    "image_path": outcome.item,
                    "raw_response": outcome.reply if outcome.error is None else "",
                    "error": outcome.error,
                    "processing_time": outcome.seconds,
                }
            )
        logger.info(
            "[%d/%d] call of %d in %.1fs",
            len(records),
            total,
            len(outcomes),
            sum(outcome.seconds for outcome in outcomes),
        )

    return records


def _finish(
    *,
    records: list[SroieRecord],
    result: Any,
    settings: SroieSettings,
    model_name: str,
    elapsed: float,
    execution_mode: str,
    wall_clock: float | None,
) -> Path:
    """Score, write both artefacts, and surface any inference failures."""
    scores = {policy: score_records(records, result.predictions, policy=policy) for policy in MatchPolicy}

    csv_path = settings.output_dir / "sroie_per_image.csv"
    summary_path = settings.output_dir / "sroie_summary.json"
    write_per_image_csv(csv_path, records, result.predictions)
    write_summary_json(
        summary_path,
        model_name=model_name,
        scores=scores,
        records=records,
        predictions=result.predictions,
        image_count=len(records),
        elapsed_seconds=elapsed,
        execution_mode=execution_mode,
    )

    for policy in MatchPolicy:
        logger.info("SROIE %s overall F1: %.4f", policy.value, scores[policy].overall_f1)
    logger.info("Wrote %s and %s", csv_path, summary_path)

    render_per_field_table(scores, records, result.predictions)
    render_execution_summary(
        execution_summary_rows(
            image_count=len(records),
            elapsed_seconds=elapsed,
            scores=scores,
            output_dir=settings.output_dir,
            execution_mode=execution_mode,
            failed_images=len(result.errors),
            wall_clock_seconds=wall_clock,
        )
    )

    # Artefacts are written first so a partly-failed run is still
    # inspectable, then the failure is surfaced. A run with inference
    # errors must not be read as a clean measurement of the model.
    if result.errors:
        logger.error(
            "%d image(s) failed inference: %s",
            len(result.errors),
            ", ".join(sorted(result.errors)[:10]),
        )
        raise SystemExit(1)

    return summary_path
