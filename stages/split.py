"""Stage 0: Split photographs of several receipts into one image per receipt (GPU).

Runs before classify. Each photograph is shown to the model, which says where each receipt is. A photograph
holding one receipt, or one the model's answer is unclear about, is copied through unchanged; a photograph the
model clearly says holds several is cut into one image per receipt. Each image, cut or whole, is then left
alone unless its rows of print drift far enough to mislead a reader. One that does is rectified from its four
corners when they can be trusted, and otherwise turned so its rows run level. See `common.photo_split` for
the rules.

Writes into the output directory:

    images/         one image per document: unchanged photographs under their own names, turned or rectified
                    ones as <photograph>.png, and cut images as <photograph>__r<n>.png. Anything the stage
                    alters is written as PNG, so a JPEG is never compressed twice. The later stages read
                    their images from here.
    splits.jsonl    one record per photograph: what was done, why, the boxes, the images written, how far
                    each was turned to level its print, whether it was rectified, how far its rows of
                    print drift (in text heights), and the model's reply.

With `pipeline.photo_split.enabled: false` every photograph is copied through and the model is not loaded.

Every image in the directory is examined: the selection the later stages apply (document types, a limit on the
number of images) is applied by them, to what this stage writes.

Started by ``python -m stages`` with ``KFP_TASK`` naming it:

    KFP_TASK=split model=<a registered model> image_dir=<folder> output=<folder> bash entrypoint.sh
"""

import logging
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any


from common.engine_calls import EngineWatch, Outcome, groups_of, in_engine_calls

from .io import append_jsonl, failed, read_completed_images, take_out

logger = logging.getLogger(__name__)

IMAGES_DIR = "images"
RECORDS_FILE = "splits.jsonl"


def cut_name(photograph: Path, number: int) -> str:
    """The file name of the n-th image cut from a photograph (n counts from 1)."""
    return f"{photograph.stem}__r{number}.png"


def altered_name(photograph: Path) -> str:
    """The file name a photograph is written under once it has been turned or rectified.

    Always PNG. A photograph usually arrives as a JPEG, and writing the altered image back as a JPEG would
    compress it a second time, at whatever quality the library defaults to, softening the small print the
    stage exists to make readable. An unaltered photograph is copied under its own name instead.
    """
    return f"{photograph.stem}.png"


def _record(
    photograph: Path,
    action: str,
    reason: str,
    boxes: list[list[float]],
    outputs: list[str],
    turned_deg: list[float],
    rectified: list[bool],
    reply: str,
    drift_heights: list[float | None],
) -> dict[str, Any]:
    return {
        "image_path": str(photograph),
        "image_name": photograph.name,
        "action": action,
        "reason": reason,
        "boxes": boxes,
        "outputs": outputs,
        "turned_deg": [round(degrees, 2) for degrees in turned_deg],
        "rectified": any(rectified),
        "rectified_each": rectified,
        "drift_heights": [None if drift is None else round(drift, 2) for drift in drift_heights],
        "raw_response": reply,
    }


def apply_reply(
    photograph: Path, reply: str, images_dir: Path, config: Any, finder: Any | None = None
) -> dict[str, Any]:
    """Act on the model's reply about one photograph: copy it through, turn it, or cut it up.

    Args:
        photograph: The photograph's path.
        reply: The model's reply to the photo split prompt.
        images_dir: Where the resulting image or images are written.
        config: The photo split settings (`common.photo_split.PhotoSplitConfig`).
        finder: The settings for finding print and its direction (`common.sheet_finder.FinderParams`).
            Needed when `config.straighten` is true; None leaves every image as it lies.

    Returns:
        The record for splits.jsonl.
    """
    from PIL import Image

    from common.photo_split import (
        box_within,
        crop_rectangles,
        decide,
        level_turn,
        rectify,
        row_measures,
        turn,
        worth_straightening,
    )

    straighten = bool(config.straighten and finder is not None)

    def measured(part: Any) -> tuple[float, float | None]:
        """The turn that would level an image's print (0 when it should be left), and its rows' drift."""
        if not straighten or finder is None:
            return 0.0, None
        direction, drift = row_measures(part, finder)
        return (level_turn(direction, config) if worth_straightening(drift, config) else 0.0), drift

    decision = decide(reply, config)
    boxes = [[box.left, box.top, box.right, box.bottom] for box in decision.boxes]
    if decision.action != "split":
        degrees, rectified, drift = 0.0, False, None
        # One document, or none the model called a receipt (a page of another kind, such as a bank
        # statement). Its rows are measured first: by the print inside the receipt's box, or by all the
        # print when there is no box. Unless they drift far enough across the page to mislead a reader, the
        # photograph is left exactly as it is. Otherwise its corners are tried: when they can be trusted the
        # homography is reversed, which removes tilt and foreshortening together; failing that the
        # photograph is turned whole. Several boxes that were not trusted are left as they lie, since they
        # may be sheets at different angles.
        if straighten and finder is not None and len(decision.boxes) <= 1:
            with Image.open(photograph) as image:
                rgb = image.convert("RGB")
                region = rgb
                if decision.boxes:
                    (rectangle,) = crop_rectangles(
                        decision.boxes, rgb.width, rgb.height, config.margin_frac
                    )
                    region = rgb.crop(rectangle)
                degrees, drift = measured(region)
                if worth_straightening(drift, config):
                    only_box = decision.boxes[0] if decision.boxes else None
                    page = rectify(rgb, finder, config, only_box) if config.rectify_pages else None
                    if page is not None:
                        page.save(images_dir / altered_name(photograph))
                        rectified, degrees = True, 0.0
                    elif degrees:
                        turn(rgb, degrees, config).save(images_dir / altered_name(photograph))
        altered = bool(degrees) or rectified
        written = altered_name(photograph) if altered else photograph.name
        if not altered:
            shutil.copy2(photograph, images_dir / written)
        return _record(
            photograph,
            "pass_through",
            decision.reason,
            boxes,
            [written],
            [degrees],
            [rectified],
            reply,
            [drift],
        )
    # Several receipts: one image is cut per box, and each is then treated as a photograph of one receipt
    # is. Its rows are measured, and unless they drift far enough to mislead a reader it is left as cut.
    # Otherwise its corners are tried, and failing that it is turned.
    outputs, turns, drifts, rectifieds = [], [], [], []
    with Image.open(photograph) as image:
        rgb = image.convert("RGB")
        rectangles = crop_rectangles(decision.boxes, rgb.width, rgb.height, config.margin_frac)
        for number, (box, rectangle) in enumerate(zip(decision.boxes, rectangles, strict=True), start=1):
            piece = rgb.crop(rectangle)
            degrees, drift = measured(piece)
            page = None
            if straighten and finder is not None and config.rectify_pieces:
                if worth_straightening(drift, config):
                    page = rectify(piece, finder, config, box_within(box, rectangle, rgb.width, rgb.height))
            name = cut_name(photograph, number)
            if page is not None:
                page.save(images_dir / name)
                degrees = 0.0
            else:
                turn(piece, degrees, config).save(images_dir / name)
            outputs.append(name)
            turns.append(degrees)
            drifts.append(drift)
            rectifieds.append(page is not None)
    return _record(photograph, "split", decision.reason, boxes, outputs, turns, rectifieds, reply, drifts)


def _ask(
    processor: Any, photographs: list[Path], config: Any, watch: EngineWatch
) -> Iterator[list[Outcome[Path, str]]]:
    """Put the photo split prompt to the model for each photograph, one engine call at a time.

    Photographs go ``batch_size`` to an engine call, and each call's outcomes are given as soon as it is
    done, in the photographs' order. A call that fails is tried again one photograph at a time, and a
    photograph that still fails comes back with its error and no reply (``common.engine_calls``).
    """

    def ask(batch: list[Path]) -> list[str]:
        images = [processor.load_document_image(str(photograph)) for photograph in batch]
        return processor.ask(images, [config.prompt] * len(images), config.max_tokens)

    return in_engine_calls(groups_of(photographs, config.batch_size), ask, watch=watch, what="photographs")


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
    from common.settings import DEFAULT_CONFIG, read_settings_file
    from stages.io import discover_images

    resolved_config = config_path or DEFAULT_CONFIG
    raw_config = read_settings_file(resolved_config)
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

    # A photograph the model could not be asked about by an earlier run is asked about again. Its
    # record is taken out, and with it the image that run passed through in its place, which a
    # photograph cut into several this time would otherwise leave beside the cuts.
    asked_again = take_out(records_path, failed)
    for record in asked_again:
        for name in record.get("outputs", []):
            if Path(name).name == name:
                (images_dir / name).unlink(missing_ok=True)
    if asked_again:
        logger.info(
            "%d photograph(s) an earlier run could not ask about will be asked about again",
            len(asked_again),
        )

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
            records.append(
                _record(
                    photograph, "pass_through", reason, [], [photograph.name], [0.0], [False], "", [None]
                )
            )
        append_jsonl(records_path, records)
        logger.info("photo_split is not enabled: copied %d photographs through unchanged", len(records))
        return images_dir

    from common.settings import Settings
    from common.pipeline_ops import open_processor

    cli_args: dict[str, Any] = {"data_dir": str(image_dir), "output_dir": str(output_dir)}
    if model_type is not None:
        cli_args["model_type"] = model_type
    app_cfg = Settings.load(cli_args, config_path=config_path)

    # Read before the engine is started, so that a setting that cannot be used stops the run first.
    finder = None
    if split_config.straighten:
        from common.sheet_finder_config import finder_params_from

        finder_section = raw_config.get("pipeline", {}).get("photo_split", {}).get("finder")
        finder = finder_params_from(finder_section, config_path=resolved_config)

    # Each engine call's photographs are dealt with, and their records written, before the next call is
    # made: a run that stops part way keeps every photograph it had finished, and carries on from there.
    records = []
    model_seconds = 0.0
    with open_processor(app_cfg) as processor:
        watch = EngineWatch(app_cfg.stop_after_failed_requests)
        for outcomes in _ask(processor, todo, split_config, watch):
            model_seconds += sum(outcome.seconds for outcome in outcomes)
            for outcome in outcomes:
                photograph = outcome.item
                record = apply_reply(photograph, outcome.reply or "", images_dir, split_config, finder)
                if outcome.error is not None:
                    # The model could not be asked about this photograph, in a shared call or alone. It
                    # is treated as one the model said nothing clear about, and says why.
                    record["reason"] = f"the model could not be asked: {outcome.error}"
                    record["error"] = outcome.error
                append_jsonl(records_path, [record])
                records.append(record)
                logger.info(
                    "[%d/%d] %s -> %s (%s)",
                    len(records),
                    len(todo),
                    photograph.name,
                    record["action"],
                    record["reason"],
                )
    logger.info(
        "Split questions complete: %d photographs in %.1fs (%.2fs/photograph)",
        len(todo),
        model_seconds,
        model_seconds / len(todo),
    )
    cut_up = sum(record["action"] == "split" for record in records)
    written = sum(len(record["outputs"]) for record in records)
    logger.info(
        "Split %d of %d photographs; wrote %d images to %s", cut_up, len(records), written, images_dir
    )
    could_not = sum(1 for record in records if failed(record))
    if could_not:
        logger.warning(
            "The model could not be asked about %d of %d photograph(s). Each is passed through as it "
            "is and recorded with its error, and is asked about again the next time the stage is run.",
            could_not,
            len(records),
        )
    return images_dir
