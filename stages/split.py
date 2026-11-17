# ruff: noqa: B008 - typer.Option in defaults is the standard Typer pattern
"""Stage 0: Split photographs of several receipts into one image per receipt (GPU).

Runs before classify. Each photograph is shown to the model, which says where each receipt is. A photograph
holding one receipt, or one the model's answer is unclear about, is copied through unchanged; a photograph the
model clearly says holds several is cut into one image per receipt. See `common.photo_split` for the rules.

Writes into the output directory:

    images/         one image per document: unchanged photographs under their own names, and cut images named
                    <photograph>__r<n>.png. The later stages read their images from here.
    splits.jsonl    one record per photograph: what was done, why, the boxes, the images written and the
                    model's reply.

With `pipeline.photo_split.enabled: false` every photograph is copied through and the model is not loaded.

Every image in the directory is examined: the selection the later stages apply (document types, a limit on the
number of images) is applied by them, to what this stage writes.

Usage:
    python -m stages.split --data-dir /data --output-dir /artifacts/split
"""

import logging
import shutil
import time
from pathlib import Path
from typing import Any

import typer

from .io import append_jsonl, read_completed_images

logger = logging.getLogger(__name__)
app = typer.Typer()

IMAGES_DIR = "images"
RECORDS_FILE = "splits.jsonl"


def cut_name(photograph: Path, number: int) -> str:
    """The file name of the n-th image cut from a photograph (n counts from 1)."""
    return f"{photograph.stem}__r{number}.png"


def _record(
    photograph: Path, action: str, reason: str, boxes: list[list[float]], outputs: list[str], reply: str
):
    return {
        "image_path": str(photograph),
        "image_name": photograph.name,
        "action": action,
        "reason": reason,
        "boxes": boxes,
        "outputs": outputs,
        "raw_response": reply,
    }


def apply_reply(photograph: Path, reply: str, images_dir: Path, config: Any) -> dict[str, Any]:
    """Act on the model's reply about one photograph: copy it through, or cut it up.

    Args:
        photograph: The photograph's path.
        reply: The model's reply to the photo split prompt.
        images_dir: Where the resulting image or images are written.
        config: The photo split settings (`common.photo_split.PhotoSplitConfig`).

    Returns:
        The record for splits.jsonl.
    """
    from PIL import Image

    from common.photo_split import cut, decide

    decision = decide(reply, config)
    boxes = [[box.left, box.top, box.right, box.bottom] for box in decision.boxes]
    if decision.action != "split":
        shutil.copy2(photograph, images_dir / photograph.name)
        return _record(photograph, "pass_through", decision.reason, boxes, [photograph.name], reply)
    outputs = []
    with Image.open(photograph) as image:
        for number, piece in enumerate(
            cut(image.convert("RGB"), decision.boxes, config.margin_frac), start=1
        ):
            name = cut_name(photograph, number)
            piece.save(images_dir / name)
            outputs.append(name)
    return _record(photograph, "split", decision.reason, boxes, outputs, reply)


def _ask(processor: Any, photographs: list[Path], config: Any) -> list[str]:
    """Put the photo split prompt to the model for each photograph, replies in the same order."""
    replies: list[str] = []
    share = getattr(processor, "supports_generate_many", False)
    step = config.batch_size if share else 1
    for start in range(0, len(photographs), step):
        batch = photographs[start : start + step]
        images = [processor.load_document_image(str(p)) for p in batch]
        if share:
            replies.extend(
                processor.generate_many(images, [config.prompt] * len(images), config.max_tokens)
            )
        else:
            replies.extend(processor.generate(image, config.prompt, config.max_tokens) for image in images)
        logger.info("Asked about %d/%d photographs", min(start + step, len(photographs)), len(photographs))
    return replies


def run(
    image_dir: Path,
    output_dir: Path,
    *,
    model_type: str | None = None,
    config_path: Path | None = None,
) -> Path:
    """Split every photograph in a directory, writing images/ and splits.jsonl.

    Args:
        image_dir: Directory containing the photographs.
        output_dir: Directory to write images/ and splits.jsonl into.
        model_type: Model type (e.g. "gemma4-31b-w4a16-vllm"). None = from the run config.
        config_path: Optional path to run_config.yml.

    Returns:
        The directory the later stages read their images from.

    Raises:
        FileNotFoundError: The directory holds no images.
        common.photo_split.PhotoSplitConfigError: The photo split settings are missing or malformed.
    """
    from common.photo_split import photo_split_config_from
    from common.pipeline_config import discover_images, load_yaml_config

    resolved_config = config_path or Path(__file__).resolve().parent.parent / "config" / "run_config.yml"
    _, raw_config = load_yaml_config(resolved_config)
    split_config = photo_split_config_from(raw_config, config_path=resolved_config)

    # Every image in the directory is examined. The selection the later stages apply (document types, a
    # limit on the number of images) is theirs to apply, to what this stage writes.
    photographs = list(discover_images(image_dir))
    if not photographs:
        msg = f"No images found in {image_dir}"
        raise FileNotFoundError(msg)

    images_dir = output_dir / IMAGES_DIR
    images_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / RECORDS_FILE

    # Resume: photographs already recorded by an earlier, interrupted run are not asked about again.
    done = read_completed_images(records_path)
    todo = [p for p in photographs if p.name not in done]
    logger.info("Found %d photographs in %s (%d already done)", len(photographs), image_dir, len(done))
    if not todo:
        logger.info("All photographs already split -- nothing to do")
        return images_dir

    if not split_config.enabled:
        records = []
        for photograph in todo:
            shutil.copy2(photograph, images_dir / photograph.name)
            reason = "photo_split is not enabled"
            records.append(_record(photograph, "pass_through", reason, [], [photograph.name], ""))
        append_jsonl(records_path, records)
        logger.info("photo_split is not enabled: copied %d photographs through unchanged", len(records))
        return images_dir

    from cli import load_pipeline_configs
    from common.app_config import AppConfig
    from common.pipeline_ops import create_processor, load_model

    cli_args: dict[str, Any] = {"data_dir": str(image_dir), "output_dir": str(output_dir)}
    if model_type is not None:
        cli_args["model_type"] = model_type
    app_cfg = AppConfig.load(cli_args, config_path=config_path)
    config = app_cfg.pipeline

    prompt_config, universal_fields, field_definitions = load_pipeline_configs(config.model_type)
    logger.info("Loading model: %s", config.model_type)
    model_cm = load_model(config, app_config=app_cfg)
    model, tokenizer = model_cm.__enter__()
    try:
        processor = create_processor(
            model, tokenizer, config, prompt_config, universal_fields, field_definitions, app_config=app_cfg
        )
        start = time.time()
        replies = _ask(processor, todo, split_config)
        elapsed = time.time() - start
        logger.info(
            "Split questions complete: %d photographs in %.1fs (%.2fs/photograph)",
            len(todo),
            elapsed,
            elapsed / len(todo),
        )
    finally:
        model_cm.__exit__(None, None, None)

    records = []
    for number, (photograph, reply) in enumerate(zip(todo, replies, strict=True), start=1):
        record = apply_reply(photograph, reply, images_dir, split_config)
        records.append(record)
        logger.info(
            "[%d/%d] %s -> %s (%s)", number, len(todo), photograph.name, record["action"], record["reason"]
        )
    append_jsonl(records_path, records)
    cut_up = sum(record["action"] == "split" for record in records)
    written = sum(len(record["outputs"]) for record in records)
    logger.info(
        "Split %d of %d photographs; wrote %d images to %s", cut_up, len(records), written, images_dir
    )
    return images_dir


@app.command()
def main(
    image_dir: Path = typer.Option(..., "--data-dir", "-d", help="Directory containing the photographs"),
    output: Path = typer.Option(
        ..., "--output-dir", "-o", help="Directory to write images/ and splits.jsonl"
    ),
    model: str | None = typer.Option(None, "--model", help="Model type"),
    config: Path | None = typer.Option(None, "--config", help="YAML configuration file"),
) -> None:
    """Stage 0: Split photographs of several receipts into one image per receipt."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(image_dir, output, model_type=model, config_path=config)


if __name__ == "__main__":
    app()
