"""vLLM data-parallel worker functions.

Each function is a top-level callable invoked in a subprocess by
``vllm_dp.run_dp``. Workers build their own vLLM engine (TP=1)
inside a process pinned to a single GPU via CUDA_VISIBLE_DEVICES.

All arguments must be picklable (strings, dicts, ints -- no model
objects, no Path objects across the process boundary).
"""

import logging
import time
from pathlib import Path
from typing import Any

from common import prompt_trace

logger = logging.getLogger(__name__)


def classify_worker(
    gpu_id: int,
    image_paths: list[str],
    *,
    config_path: str | None,
    cli_overrides: dict[str, Any],
) -> list[dict[str, Any]]:
    """Worker: build vLLM engine + processor, classify each image.

    Args:
        gpu_id: GPU rank (for logging).
        image_paths: Absolute paths to images (strings).
        config_path: Path to run_config.yml (or None).
        cli_overrides: CLI args dict for AppConfig.load().

    Returns:
        List of classification record dicts.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(levelname)s [GPU {gpu_id}] %(name)s: %(message)s",
    )

    from common.pipeline_ops import load_pipeline_configs
    from common.app_config import AppConfig
    from common.pipeline_ops import create_processor, load_model

    # Force single-GPU in this worker
    overrides = dict(cli_overrides)
    overrides["num_gpus"] = 1

    cfg_path = Path(config_path) if config_path else None
    app_cfg = AppConfig.load(overrides, config_path=cfg_path).for_stage("classify")
    config = app_cfg.pipeline

    prompt_config, universal_fields, field_definitions = load_pipeline_configs(config.model_type)

    logger.info("Loading model: %s (GPU %d)", config.model_type, gpu_id)
    model_cm = load_model(config, app_config=app_cfg)
    model, tokenizer = model_cm.__enter__()

    try:
        processor = create_processor(
            model,
            tokenizer,
            config,
            prompt_config,
            universal_fields,
            field_definitions,
            app_config=app_cfg,
        )

        from stages.classify import _classify_batch_size

        records = _classify_records(processor, image_paths, _classify_batch_size(cfg_path), config.verbose)

        # Tag every record with the rank that produced it. Workers run
        # concurrently, so the parent needs this to report the SLOWEST
        # worker's inference time — summing across workers would give total
        # compute and understate throughput by roughly the GPU count.
        for record in records:
            record["gpu_id"] = gpu_id
        return records
    finally:
        model_cm.__exit__(None, None, None)


class _Records:
    """Collects what a stage function writes, for a worker that returns its records instead of writing them.

    The one-engine stage writes each record to its output file as it goes. A worker hands its records back
    to the parent process, which writes them. Giving the stage's own functions this in place of a file
    writer lets both paths run the same code, so they cannot drift apart.
    """

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def write(self, record: dict[str, Any]) -> None:
        self.records.append(record)


def _classify_records(
    processor: Any, image_paths: list[str], batch_size: int, verbose: bool
) -> list[dict[str, Any]]:
    """Classify a worker's pages, several to an engine call, exactly as the one-engine stage does.

    Each record also carries a processing time, which the parent sums for each worker to report inference
    apart from wall clock. Inside a batch the requests overlap, so a page's own time is not separable: each
    is charged an equal share of the whole.

    Args:
        processor: The document processor.
        image_paths: This worker's pages, in order.
        batch_size: Pages per engine call, from pipeline.batching.classify.
        verbose: Passed through to the processor.

    Returns:
        One classification record per page, in input order.
    """
    from stages.classify import _classify_images

    started = time.time()
    records = _classify_images(processor, image_paths, batch_size, verbose)
    share = (time.time() - started) / len(records) if records else 0.0
    for record in records:
        record["processing_time"] = share
    return records


def _bank_batch_records(bank_adapter: Any, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Read a batch of bank statements single pass in one engine call, as the one-engine stage does.

    Args:
        bank_adapter: The bank reader, able to share an engine call (`can_batch`).
        batch: The statements' classification records.

    Returns:
        One extraction record per statement, in the shape the stage writes.
    """
    from stages.extract import _extract_bank_single_pass_batch

    collected = _Records()
    _extract_bank_single_pass_batch(collected, bank_adapter, batch)
    return collected.records


def _extract_batch_records(
    processor: Any,
    batch: list[dict[str, Any]],
    *,
    verbose: bool,
) -> list[dict[str, Any]]:
    """Extract a batch of standard documents in one engine call.

    Returns the same record shape the per-image path writes — RAW
    responses, which the clean stage parses. The two paths must stay
    indistinguishable in raw_extractions.jsonl, or resume and re-cleaning
    break.

    A batch that raises is retried one document at a time, so an
    out-of-memory costs only the document that genuinely fails.
    """
    started = time.time()
    image_paths = [c["image_path"] for c in batch]

    try:
        results = processor.extract_batch(image_paths, batch, verbose=verbose)
        if len(results) != len(batch):
            raise ValueError(f"backend returned {len(results)} results for {len(batch)} images")
    except Exception as err:  # noqa: BLE001 - one batch must not end the run
        logger.warning(
            "Batch of %d %s failed (%s) — retrying one at a time",
            len(batch),
            batch[0]["document_type"],
            err,
        )
        records = []
        for classification in batch:
            img_start = time.time()
            try:
                result = processor.process_document_aware(
                    classification["image_path"], classification, verbose=verbose
                )
                raw, error = result.get("raw_response", ""), None
            except Exception as inner:  # noqa: BLE001
                logger.error("Error extracting %s: %s", classification["image_name"], inner)
                raw, error = "", str(inner)
            records.append(
                {
                    "image_name": classification["image_name"],
                    "image_path": classification["image_path"],
                    "document_type": classification["document_type"],
                    "raw_response": raw,
                    "processing_time": time.time() - img_start,
                    "prompt_used": classification["document_type"].lower() if error is None else "error",
                    "error": error,
                }
            )
        return records

    # Inside a batch the requests overlap, so per-image timing is not
    # separable; charge each an equal share of the call's wall clock.
    share = (time.time() - started) / len(batch)
    return [
        {
            "image_name": classification["image_name"],
            "image_path": classification["image_path"],
            "document_type": classification["document_type"],
            "raw_response": result.get("raw_response", ""),
            "processing_time": share,
            "prompt_used": classification["document_type"].lower(),
            "error": None,
        }
        for classification, result in zip(batch, results, strict=True)
    ]


def classified_extract_worker(
    gpu_id: int,
    image_paths: list[str],
    *,
    config_path: str | None,
    cli_overrides: dict[str, Any],
    classifications: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Worker: build vLLM engine + processor, run classified extraction.

    Handles standard documents via ``process_document_aware`` and bank
    statements via ``UnifiedBankExtractor``.  Each worker receives the
    full classifications list and filters to its own image chunk.

    Args:
        gpu_id: GPU rank (for logging).
        image_paths: Absolute paths to images assigned to this GPU.
        config_path: Path to run_config.yml (or None).
        cli_overrides: CLI args dict for AppConfig.load().
        classifications: Full list of classification records (filtered
            inside the worker to match ``image_paths``).

    Returns:
        List of extraction record dicts.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(levelname)s [GPU {gpu_id}] %(name)s: %(message)s",
    )

    from common.pipeline_ops import load_pipeline_configs
    from common.app_config import AppConfig
    from common.pipeline_ops import create_processor, load_model
    from common.unified_bank_extractor import UnifiedBankExtractor, serialise_bank_response

    # Force single-GPU in this worker
    overrides = dict(cli_overrides)
    overrides["num_gpus"] = 1

    cfg_path = Path(config_path) if config_path else None
    app_cfg = AppConfig.load(overrides, config_path=cfg_path)
    config = app_cfg.pipeline
    effective_verbose = config.verbose

    prompt_config, universal_fields, field_definitions = load_pipeline_configs(config.model_type)

    # Filter classifications to this worker's image chunk
    chunk_set = set(image_paths)
    my_classifications = [c for c in classifications if c["image_path"] in chunk_set]

    logger.info("Loading model: %s (GPU %d)", config.model_type, gpu_id)
    model_cm = load_model(config, app_config=app_cfg)
    model, tokenizer = model_cm.__enter__()

    try:
        processor = create_processor(
            model,
            tokenizer,
            config,
            prompt_config,
            universal_fields,
            field_definitions,
            app_config=app_cfg,
        )

        # Set up bank adapter if needed
        bank_adapter = None
        has_bank = any(c["document_type"].upper() == "BANK_STATEMENT" for c in my_classifications)
        if has_bank and config.bank_v2 and getattr(processor, "supports_multi_turn", True):
            bank_budget = app_cfg.get_image_budget("bank_statement")
            bank_adapter = UnifiedBankExtractor(
                generate_fn=processor.generate,
                generate_many_fn=(
                    processor.generate_many if getattr(processor, "supports_generate_many", False) else None
                ),
                verbose=effective_verbose,
                use_balance_correction=config.balance_correction,
                max_tiles=(bank_budget["max_tiles"] if config.pre_tiling_enabled else None),
                min_tiles=(bank_budget["min_tiles"] if config.pre_tiling_enabled else None),
                bank_strategy=config.bank_strategy,
            )
            logger.info(
                "Bank adapter enabled (GPU %d, bank_strategy=%s)", gpu_id, bank_adapter.bank_strategy
            )

        records: list[dict[str, Any]] = []
        total = len(my_classifications)

        # Group into engine calls. Batching happens INSIDE the worker, after
        # run_dp has partitioned across GPUs, so each rank batches only its
        # own share and the two levels of parallelism compose.
        from common.extraction_batching import plan_extraction_batches, read_batch_sizes
        from common.pipeline_config import load_yaml_config
        from stages.extract import _unbatchable_types

        _, raw_config = load_yaml_config(
            cfg_path or Path(__file__).resolve().parent.parent / "config" / "run_config.yml"
        )
        batch_sizes = read_batch_sizes(
            raw_config,
            config_path=cfg_path or Path("config/run_config.yml"),
        )
        batches = plan_extraction_batches(
            my_classifications,
            batch_sizes=batch_sizes,
            unbatchable=_unbatchable_types(bank_adapter),
        )
        logger.info(
            "GPU %d: %d documents in %d engine calls", gpu_id, len(my_classifications), len(batches)
        )

        idx = 0
        for batch in batches:
            # A multi-document batch is one engine call; singletons fall
            # through to the per-image path below unchanged.
            if len(batch) > 1:
                if batch[0]["document_type"].upper() == "BANK_STATEMENT" and bank_adapter is not None:
                    records.extend(_bank_batch_records(bank_adapter, batch))
                else:
                    records.extend(_extract_batch_records(processor, batch, verbose=effective_verbose))
                idx += len(batch)
                logger.info("[%d/%d] batch of %d %s", idx, total, len(batch), batch[0]["document_type"])
                continue

            classification = batch[0]
            idx += 1
            image_path = classification["image_path"]
            image_name = classification["image_name"]
            doc_type = classification["document_type"]

            # Progress up front: a dense bank statement can take a while, so log
            # the start (not just the end) to track which of N is in flight.
            logger.info("[%d/%d] extracting %s (%s)...", idx, total, image_name, doc_type)

            img_start = time.time()

            # Attribution for the raw-prompt trace. Shared by both branches
            # below; a bank statement makes several VLM calls per image, so
            # without it the trace cannot be tied back to a document.
            trace_ctx = {
                "image_name": image_name,
                "pipeline": "information_extraction",
                "label": f"extract_{doc_type.lower()}",
            }

            try:
                if doc_type.upper() == "BANK_STATEMENT" and bank_adapter is not None:
                    with prompt_trace.trace_context(**trace_ctx):
                        schema_fields, metadata = bank_adapter.extract_bank_statement(image_path)
                    img_time = time.time() - img_start
                    raw_response_str = serialise_bank_response(schema_fields, metadata)
                    strategy = metadata.get("strategy_used", "unknown")
                    records.append(
                        {
                            "image_name": image_name,
                            "image_path": image_path,
                            "document_type": doc_type,
                            "raw_response": raw_response_str,
                            "processing_time": img_time,
                            "prompt_used": f"unified_bank_{strategy}",
                            "error": None,
                        }
                    )
                else:
                    with prompt_trace.trace_context(**trace_ctx):
                        result = processor.process_document_aware(
                            image_path, classification, verbose=effective_verbose
                        )
                    img_time = time.time() - img_start
                    records.append(
                        {
                            "image_name": image_name,
                            "image_path": image_path,
                            "document_type": doc_type,
                            "raw_response": result.get("raw_response", ""),
                            "processing_time": img_time,
                            "prompt_used": doc_type.lower(),
                            "error": None,
                        }
                    )
            except Exception as e:
                logger.error("Error extracting %s: %s", image_name, e)
                img_time = time.time() - img_start
                records.append(
                    {
                        "image_name": image_name,
                        "image_path": image_path,
                        "document_type": doc_type,
                        "raw_response": "",
                        "processing_time": img_time,
                        "prompt_used": "error",
                        "error": str(e),
                    }
                )

            logger.info(
                "[%d/%d] %s: %s (%.1fs)",
                idx,
                total,
                image_name,
                doc_type,
                time.time() - img_start,
            )

        # Tag every record with the rank that produced it. Workers run
        # concurrently, so the parent needs this to report the SLOWEST
        # worker's inference time — summing across workers would give total
        # compute and understate throughput by roughly the GPU count.
        for record in records:
            record["gpu_id"] = gpu_id
        return records
    finally:
        model_cm.__exit__(None, None, None)


def sroie_worker(
    gpu_id: int,
    image_paths: list[str],
    *,
    config_path: str | None,
    cli_overrides: dict[str, Any],
    max_new_tokens: int,
    tile_budget: dict[str, int],
    batch_size: int,
) -> list[dict[str, Any]]:
    """Worker: build vLLM engine, ask each receipt for its four fields.

    Returns RAW responses only. Parsing and scoring stay in the parent, so
    the serial and data-parallel paths share one parser and cannot drift.

    A failure on one image is recorded and the worker continues — losing a
    whole partition to one bad receipt would waste the run.

    Args:
        gpu_id: GPU rank (for logging).
        image_paths: Absolute paths to receipt images (strings).
        config_path: Path to run_config.yml (or None).
        cli_overrides: CLI args dict for AppConfig.load().
        max_new_tokens: Generation budget per receipt.
        tile_budget: GenerationParams.extra carrying BOTH min_tiles and
            max_tiles, from common.sroie.tiling.tile_extra. Inert unless
            pre-tiling is on. min_tiles is the lever — sending only the
            ceiling leaves the backend on its min_tiles=1 default.
        batch_size: Receipts submitted to the engine per chat() call.

    Returns:
        One dict per image: image_id, image_path, raw_response, error.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(levelname)s [GPU {gpu_id}] %(name)s: %(message)s",
    )

    from PIL import Image

    from common.pipeline_ops import load_pipeline_configs
    from common.app_config import AppConfig
    from common.pipeline_ops import create_processor, load_model
    from common.sroie.prompt import SROIE_PROMPT

    # Force single-GPU in this worker
    overrides = dict(cli_overrides)
    overrides["num_gpus"] = 1

    cfg_path = Path(config_path) if config_path else None
    app_cfg = AppConfig.load(overrides, config_path=cfg_path)
    config = app_cfg.pipeline

    prompt_config, universal_fields, field_definitions = load_pipeline_configs(config.model_type)

    logger.info("Loading model: %s (GPU %d)", config.model_type, gpu_id)
    model_cm = load_model(config, app_config=app_cfg)
    model, tokenizer = model_cm.__enter__()

    try:
        processor = create_processor(
            model,
            tokenizer,
            config,
            prompt_config,
            universal_fields,
            field_definitions,
            app_config=app_cfg,
        )

        records: list[dict[str, Any]] = []
        total = len(image_paths)

        for start in range(0, total, batch_size):
            chunk = image_paths[start : start + batch_size]
            batch_start = time.time()

            try:
                images = [Image.open(p).convert("RGB") for p in chunk]
                try:
                    responses = processor.generate_batch(
                        images,
                        [SROIE_PROMPT] * len(images),
                        max_tokens=max_new_tokens,
                        extra=dict(tile_budget),
                    )
                finally:
                    for image in images:
                        image.close()
                if len(responses) != len(chunk):
                    raise ValueError(f"backend returned {len(responses)} responses for {len(chunk)} images")
                errors: list[str | None] = [None] * len(chunk)
            except Exception as err:  # noqa: BLE001 - a batch must not end the run
                # Retry singly so an out-of-memory on a large batch costs
                # only the image that genuinely fails.
                logger.warning("Batch of %d failed (%s) — retrying one at a time", len(chunk), err)
                responses, errors = [], []
                for image_path in chunk:
                    try:
                        with Image.open(image_path) as image:
                            responses.append(
                                processor.generate(
                                    image.convert("RGB"),
                                    SROIE_PROMPT,
                                    max_tokens=max_new_tokens,
                                    extra=dict(tile_budget),
                                )
                            )
                        errors.append(None)
                    except Exception as inner:  # noqa: BLE001
                        logger.error("%s FAILED: %s", Path(image_path).stem, inner)
                        responses.append("")
                        errors.append(str(inner))

            # Charge each image an equal share of the batch's wall clock:
            # inside a batch the requests overlap, so per-image timing is
            # not separable, and the parent only needs the per-GPU total.
            share = (time.time() - batch_start) / len(chunk)
            for image_path, response, error in zip(chunk, responses, errors, strict=True):
                records.append(
                    {
                        "image_id": Path(image_path).stem,
                        "image_path": image_path,
                        "raw_response": response,
                        "error": error,
                        # Tagged with the rank so the parent can report the
                        # SLOWEST worker's inference time rather than wall
                        # clock that includes engine startup.
                        "gpu_id": gpu_id,
                        "elapsed": share,
                    }
                )

            logger.info(
                "[%d/%d] batch of %d in %.1fs",
                start + len(chunk),
                total,
                len(chunk),
                time.time() - batch_start,
            )

        return records
    finally:
        model_cm.__exit__(None, None, None)
