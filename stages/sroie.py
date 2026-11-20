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

import typer

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
from common.sroie.runner import benchmark_from_worker_responses, inference_seconds_from_responses
from common.sroie.scoring import MatchPolicy, score_records

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)

# run_dp resolves this by dotted string in a subprocess, so a typo here
# would only surface minutes into a GPU run. Named as a constant so a test
# can import it and check it resolves.
SROIE_WORKER_FN = "common.vllm_dp_workers.sroie_worker"


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
    from common.app_config import AppConfig
    from common.pipeline_ops import open_processor
    from common.vllm_dp import resolve_dp_gpus, run_dp

    cli_args: dict[str, object] = {}
    if model_type:
        cli_args["model_type"] = model_type

    app_cfg = AppConfig.load(cli_args, config_path=config_path)
    config = app_cfg.pipeline
    settings = _settings(app_cfg.config_path)

    records = load_sroie_split(settings.data_dir)
    if max_images is not None:
        records = records[:max_images]
    logger.info("SROIE: %d records from %s", len(records), settings.data_dir)
    image_paths = [str(record.image_path) for record in records]

    # One engine on each card where the model allows it, else one engine across the cards. Either way the
    # receipts are read by read_sroie; only where the engine is started differs.
    started = time.time()
    engines = resolve_dp_gpus(config, config.model_type)
    if engines:
        responses = run_dp(
            num_gpus=engines,
            images=[record.image_path for record in records],
            worker_fn=SROIE_WORKER_FN,
            worker_kwargs={
                "config_path": str(config_path) if config_path else None,
                "cli_overrides": cli_args,
            },
            app_config=app_cfg,
        )
        wall_clock: float | None = time.time() - started
        execution_mode = f"data-parallel ({engines} GPUs)"
    else:
        with open_processor(app_cfg) as processor:
            started = time.time()
            responses = read_sroie(processor, image_paths, app_cfg)
        wall_clock = None
        execution_mode = "single-engine"

    # The model's time, apart from starting the engine: with an engine on each card, the slowest card's.
    elapsed = inference_seconds_from_responses(responses, fallback=time.time() - started)
    return _finish(
        records=records,
        result=benchmark_from_worker_responses(records, responses),
        settings=settings,
        model_name=config.model_type,
        elapsed=elapsed,
        execution_mode=execution_mode,
        wall_clock=wall_clock,
    )


def _settings(config_path: Path | None) -> SroieSettings:
    """The pipeline.sroie block of the run's settings file."""
    from common.pipeline_config import load_yaml_config

    resolved = config_path or Path(__file__).parent.parent / "config" / "run_config.yml"
    _, raw_config = load_yaml_config(resolved)
    return SroieSettings.from_raw(raw_config, config_path=resolved)


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
        One record per receipt, in the order given: image_id, image_path, raw_response, error, elapsed.
    """
    from PIL import Image

    settings = _settings(app_config.config_path)
    records: list[dict[str, Any]] = []
    total = len(image_paths)

    for start in range(0, total, settings.batch_size):
        chunk = image_paths[start : start + settings.batch_size]
        batch_start = time.time()

        try:
            images = [Image.open(p).convert("RGB") for p in chunk]
            try:
                responses = processor.generate_batch(
                    images,
                    [SROIE_PROMPT] * len(images),
                    max_tokens=settings.max_new_tokens,
                )
            finally:
                for image in images:
                    image.close()
            if len(responses) != len(chunk):
                raise ValueError(f"backend returned {len(responses)} responses for {len(chunk)} images")
            errors: list[str | None] = [None] * len(chunk)
        except Exception as err:  # noqa: BLE001 - a batch must not end the run
            logger.warning("Batch of %d failed (%s): retrying one at a time", len(chunk), err)
            responses, errors = [], []
            for image_path in chunk:
                try:
                    with Image.open(image_path) as image:
                        responses.append(
                            processor.generate(
                                image.convert("RGB"),
                                SROIE_PROMPT,
                                max_tokens=settings.max_new_tokens,
                            )
                        )
                    errors.append(None)
                except Exception as inner:  # noqa: BLE001 - recorded, not raised
                    logger.error("%s FAILED: %s", Path(image_path).stem, inner)
                    responses.append("")
                    errors.append(str(inner))

        # Inside a call the requests overlap, so a receipt's own time is not separable: each is charged an
        # equal share of the call.
        share = (time.time() - batch_start) / len(chunk)
        for image_path, response, error in zip(chunk, responses, errors, strict=True):
            records.append(
                {
                    "image_id": Path(image_path).stem,
                    "image_path": image_path,
                    "raw_response": response,
                    "error": error,
                    "elapsed": share,
                }
            )

        logger.info(
            "[%d/%d] batch of %d in %.1fs", start + len(chunk), total, len(chunk), time.time() - batch_start
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
        raise typer.Exit(1) from None

    return summary_path


@app.command()
def main(
    model_type: str = typer.Option(None, "--model-type", help="Override bootstrap.model.type."),
    max_images: int = typer.Option(None, "--max-images", help="Score only the first N records."),
    config_path: Path = typer.Option(None, "--config", help="Path to run_config.yml."),
) -> None:
    """Run the SROIE benchmark stage."""
    # Per-image progress is always at INFO — it is the only sign the user
    # has that inference is advancing. A 347-image run takes ~35 minutes
    # and writes its artefacts only at the end, so without this the stage
    # is indistinguishable from a hang.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    run(model_type=model_type, max_images=max_images, config_path=config_path)


if __name__ == "__main__":
    app()
