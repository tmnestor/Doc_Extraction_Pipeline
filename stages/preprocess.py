"""Stage 0: preprocess. Hand the later stages one upright, level image for each document (GPU).

Until 27 November 2026 the stage was called ``split``, after one of the things it does. It sets a photograph
the right way up, cuts a photograph of several receipts into one image for each, and levels or rectifies an
image whose print drifts: the name described the second and hid the rest.

Runs before classify. Each photograph is first set the right way up: one lying on its side is set on end by
its print, and the model says whether it is then upside down. A photograph that arrives turned a quarter or
a half with no orientation tag is read worse, and this puts it right. A photograph so turned is written as
<photograph>.png, and how far it was turned is recorded as `upright_turn`.

Each photograph is then shown to the model, which says where each receipt is. A photograph
holding one receipt, or one the model's answer is unclear about, is copied through unchanged; a photograph the
model clearly says holds several is cut into one image per receipt. Each image, cut or whole, is then left
alone unless its rows of print drift far enough to mislead a reader. One that does is rectified from its four
corners when they can be trusted, and otherwise turned so its rows run level. See
`common.image_preprocessing` for the rules.

Writes into the output directory:

    images/         one image per document: unchanged photographs under their own names, turned or rectified
                    ones as <photograph>.png, and cut images as <photograph>__r<n>.png. Anything the stage
                    alters is written as PNG, so a JPEG is never compressed twice. The later stages read
                    their images from here.
    preprocess.jsonl
                    one record per photograph: what was done, why, the boxes, the images written, how far
                    each was turned to level its print, whether it was rectified, how far its rows of
                    print drift (in text heights), and the model's reply. A photograph that was cut has
                    the action "split", as it always had.

Unless the job sets `preprocess_photographs=true`, every photograph is copied through and the model is not
loaded.

Every image in the directory is examined: the selection the later stages apply (document types, a limit on the
number of images) is applied by them, to what this stage writes.

Started by ``python -m stages`` with ``KFP_TASK`` naming it:

    KFP_TASK=preprocess model=<a registered model> image_dir=<folder> output=<folder> bash entrypoint.sh
"""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import image_preprocessing
from common.engine_calls import EngineWatch, groups_of, in_engine_calls
from common.errors import ConfigError
from common.image_preprocessing import PreprocessConfig
from common.pipeline_ops import check_processor
from common.prompt_catalog import PromptCatalog
from common.sheet_finder import FinderParams
from common.stage_settings import StageSettings
from common.vllm_dp import RecordSink

from .carry_on import Work, carry_on
from .io import discover_images, no_pages_in, start_afresh
from .records import failed

logger = logging.getLogger(__name__)

IMAGES_DIR = "images"
RECORDS_FILE = "preprocess.jsonl"


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
    upright_turn: int,
) -> dict[str, Any]:
    return {
        "image_path": str(photograph),
        "image_name": photograph.name,
        "action": action,
        "reason": reason,
        "boxes": boxes,
        "outputs": outputs,
        "upright_turn": upright_turn,
        "turned_deg": [round(degrees, 2) for degrees in turned_deg],
        "rectified": any(rectified),
        "rectified_each": rectified,
        "drift_heights": [None if drift is None else round(drift, 2) for drift in drift_heights],
        "raw_response": reply,
    }


def apply_reply(
    photograph: Path,
    reply: str,
    images_dir: Path,
    config: PreprocessConfig,
    finder: FinderParams | None = None,
    upright_turn: int = 0,
) -> dict[str, Any]:
    """Act on the model's reply about one photograph: copy it through, turn it, or cut it up.

    Args:
        photograph: The photograph's path.
        reply: The model's reply to the question of where the receipts are.
        images_dir: Where the resulting image or images are written.
        config: The stage's rules.
        finder: The values for finding print and its direction.
            Needed when `config.straighten` is true; None leaves every image as it lies.
        upright_turn: How far the photograph was turned anticlockwise to set its print the right way
            up before the model was asked where the receipts are: 0, 90, 180 or 270 degrees. The
            reply's boxes are of the photograph so turned, and so is every image written.

    Returns:
        The record for preprocess.jsonl.
    """
    from PIL import Image, ImageOps

    from common.image_preprocessing import (
        box_within,
        crop_rectangles,
        decide,
        level_turn,
        rectify,
        row_measures,
        turn,
        turned_upright,
        worth_straightening,
    )

    straighten = bool(config.straighten and finder is not None)

    def the_right_way_up(image: Any) -> Any:
        """The photograph the way up its tag says, then turned as it was to set its print upright."""
        return turned_upright(ImageOps.exif_transpose(image).convert("RGB"), upright_turn)

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
                rgb = the_right_way_up(image)
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
        written = altered_name(photograph) if altered or upright_turn else photograph.name
        if upright_turn and not altered:
            # Set the right way up and otherwise left as it is.
            with Image.open(photograph) as image:
                the_right_way_up(image).save(images_dir / written)
        elif not altered:
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
            upright_turn,
        )
    # Several receipts: one image is cut per box, and each is then treated as a photograph of one receipt
    # is. Its rows are measured, and unless they drift far enough to mislead a reader it is left as cut.
    # Otherwise its corners are tried, and failing that it is turned.
    outputs, turns, drifts, rectifieds = [], [], [], []
    with Image.open(photograph) as image:
        rgb = the_right_way_up(image)
        rectangles = crop_rectangles(decision.boxes, rgb.width, rgb.height, config.margin_frac)
        for number, (box, rectangle) in enumerate(zip(decision.boxes, rectangles, strict=True), start=1):
            piece = rgb.crop(rectangle)
            degrees, drift = measured(piece)
            page = None
            if (
                straighten
                and finder is not None
                and config.rectify_pieces
                and worth_straightening(drift, config)
            ):
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
    return _record(
        photograph, "split", decision.reason, boxes, outputs, turns, rectifieds, reply, drifts, upright_turn
    )


def ask_about_photographs(
    processor: Any, photographs: list[Path], app_config: Any, *, sink: RecordSink | None = None
) -> list[dict[str, Any]]:
    """Ask the model where the documents are in each photograph. The one place photographs are asked about.

    One engine across the cards calls this once with every photograph. With an engine on each card, each
    worker calls it with its share. It loads no model, cuts no image and writes no file: it gives the
    model's reply for each photograph, and the stage's ``run`` acts on the replies.

    A photograph is first set the right way up, since one that arrives turned with no orientation tag
    is read worse. One lying on its side is set on end by its print (``image_preprocessing.lies_on_its_side``),
    which cannot tell the right way up from upside down. The model is then asked whether the print is
    the right way up, and on NO the photograph is turned half way round. Only then is it asked where the
    receipts are, so its boxes are of the photograph the right way up. So a call here is two engine calls.

    Photographs go ``image_preprocessing.CONFIG.batch_size`` to an engine call. A call that fails is tried again
    one photograph at a time, and a photograph that still fails comes back with its error and no reply
    (``common.engine_calls``).

    Args:
        processor: The document processor.
        photographs: The photographs, in order.
        app_config: The run's settings.
        sink: Where to write each photograph's reply as soon as it is given, if anywhere.

    Returns:
        For each photograph, in the order given: image_name, image_path, raw_response, error,
        processing_time and upright_turn, how far it was turned anticlockwise to set it the right way up.
    """
    config = image_preprocessing.CONFIG
    prompt, is_it_upright = (PromptCatalog().preprocess_prompt(key) for key in ("prompt", "upright"))
    watch = EngineWatch()
    turns: dict[Path, int] = {}

    def ask(batch: list[Path]) -> list[str]:
        images = []
        for photograph in batch:
            image = processor.load_document_image(str(photograph))
            turns[photograph] = 90 * image_preprocessing.lies_on_its_side(
                image, image_preprocessing.FINDER, config
            )
            images.append(image_preprocessing.turned_upright(image, turns[photograph]))
        said = processor.ask(images, [is_it_upright] * len(images), config.upright_max_tokens)
        for index, (photograph, reply) in enumerate(zip(batch, said, strict=True)):
            if image_preprocessing.said_upright(reply) is False:
                turns[photograph] += 180
                images[index] = image_preprocessing.turned_upright(images[index], 180)
        return processor.ask(images, [prompt] * len(images), config.max_tokens)

    asked: list[dict[str, Any]] = []
    calls = in_engine_calls(groups_of(photographs, config.batch_size), ask, watch=watch, what="photographs")
    for outcomes in calls:
        for outcome in outcomes:
            record = {
                "image_name": outcome.item.name,
                "image_path": str(outcome.item),
                "raw_response": outcome.reply or "",
                "error": outcome.error,
                "processing_time": outcome.seconds,
                # A photograph the model could not be asked about is left the way up it came.
                "upright_turn": turns.get(outcome.item, 0) if outcome.error is None else 0,
            }
            asked.append(record)
            if sink is not None:
                sink.write(record)
    return asked


def _remove_the_images_of(records: list[dict[str, Any]], images_dir: Path) -> None:
    """Remove the images earlier records name, of photographs that are to be asked about again.

    A photograph the model could not be asked about was passed through as it is. Cut into several this
    time, it would otherwise leave that image beside the cuts.
    """
    for record in records:
        for name in record.get("outputs", []):
            if Path(name).name == name:
                (images_dir / name).unlink(missing_ok=True)


@dataclass(frozen=True)
class Checked:
    """What the stage is run with, read and checked."""

    # Whether photographs are preprocessed: the model is asked where the receipts are, and a photograph of
    # several is cut. Off, every photograph is passed through and no model is asked.
    enabled: bool
    # The run's settings. None when they are not: no model is asked, and none is looked for.
    settings: StageSettings | None
    # Every image of the folder, in the order of their names. The selection the later stages apply
    # (document types, a limit on the number of images) is theirs to apply, to what this stage writes.
    photographs: list[Path]


def checked(settings: StageSettings, image_dir: Path) -> Checked:
    """Whether photographs are preprocessed, the run's settings and the photographs, each read and checked.

    The one place the stage works out what it is run with. What it gives is what ``run`` is handed:
    nothing is read or worked out a second time. The stage's own rules are not settings: they
    are ``common.image_preprocessing.CONFIG`` and ``FINDER``.

    Args:
        settings: The run's settings. Its ``preprocess_photographs`` says whether they are: with
            it off no model is asked, and a model's folder that is not there does not stop the stage.
        image_dir: The folder of photographs.

    Raises:
        ConfigError: With what is wrong, where, what is expected and how to mend it.
    """
    photographs = list(discover_images(image_dir))
    if not photographs:
        raise no_pages_in(image_dir, "")
    logger.info("Found %d photographs in %s", len(photographs), image_dir)
    if not settings.preprocess_photographs:
        return Checked(False, None, photographs)
    # Read here, before any engine is started: each is asked for again where the model is asked.
    PromptCatalog().preprocess_prompt()
    PromptCatalog().preprocess_prompt("upright")
    return Checked(True, settings, photographs)


def check(task: StageSettings) -> Checked:
    """Read and check everything the stage will use, and give it for ``start``. Nothing is written.

    Raises:
        ConfigError: With what is wrong, where, what is expected and how to mend it.
    """
    assert task.image_dir is not None
    # The folder the stage wrote to while it was called split. Left beside this run's, a later stage
    # pointed at it would read the images of that earlier run.
    earlier = task.output / "split"
    if earlier.exists():
        raise ConfigError(
            f"What: the output folder holds a folder named split, which this stage wrote before it was "
            f"renamed preprocess on 27 November 2026. Its images are an earlier run's.\n"
            f"Where: {earlier.resolve()}, under the folder the job's variable output names.\n"
            f"Expected: an output folder with no split folder in it. This stage writes "
            f"{task.preprocess_dir.name}/images and {task.preprocess_dir.name}/{RECORDS_FILE}.\n"
            f"How to fix: give this run a folder of its own in `output`, or delete {earlier}."
        )
    found = checked(task, task.image_dir)
    if found.settings is not None:
        check_processor(found.settings)
    return found


def run(output_dir: Path, checked: Checked, *, timing: Path | None = None) -> Path:
    """Preprocess every photograph in a directory, writing images/ and preprocess.jsonl.

    A run carries on from what an earlier one left (``stages.carry_on``): photographs already dealt with
    are not asked about again, a photograph an earlier run could not ask about is, and preprocess.jsonl is
    left in the photographs' order however the run went. The model's reply about each photograph goes
    with the record of what was done about it into a journal, the moment it has been acted on: a run
    that stops part way keeps both, and the next run has nothing to do about those photographs again.

    Args:
        output_dir: Directory to write images/ and preprocess.jsonl into.
        checked: Whether photographs are preprocessed, the run's settings and the photographs: what
            ``checked`` gave.
        timing: The run's output folder, to note a row of the stage's time in. None: no row.

    Returns:
        The directory the later stages read their images from.
    """
    stage_config, app_cfg, photographs = image_preprocessing.CONFIG, checked.settings, checked.photographs
    finder = image_preprocessing.FINDER if stage_config.straighten else None

    images_dir = output_dir / IMAGES_DIR
    images_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / RECORDS_FILE
    order = [photograph.name for photograph in photographs]

    def asked_again(records: list[dict[str, Any]]) -> None:
        _remove_the_images_of(records, images_dir)
        logger.info(
            "%d photograph(s) an earlier run could not ask about will be asked about again", len(records)
        )

    if not checked.enabled:
        # Every photograph is copied through, and no engine is started. The stage carries on the one
        # way all the same: what a run that preprocessed photographs had done before it stopped is kept as
        # that run left it, and a photograph such a run could not ask about is passed through.
        def copy_through(photograph: Path) -> dict[str, Any]:
            shutil.copy2(photograph, images_dir / photograph.name)
            reason = "preprocess_photographs is not enabled"
            return _record(
                photograph, "pass_through", reason, [], [photograph.name], [0.0], [False], "", [None], 0
            )

        copied = carry_on(
            None,
            Work(
                stage="preprocess",
                output=records_path,
                items=photographs,
                name_of=lambda photograph: photograph.name,
                tell=copy_through,
                taken_out=asked_again,
                thing="photograph",
            ),
        )
        logger.info(
            "preprocess_photographs is not enabled: copied %d photographs through unchanged",
            len(copied.records),
        )
        return images_dir

    assert app_cfg is not None
    acted_on = 0

    def act_on(asked: dict[str, Any]) -> dict[str, Any]:
        """Act on the model's reply about one photograph, and give the stage's record of it."""
        nonlocal acted_on
        photograph = Path(asked["image_path"])
        record = apply_reply(
            photograph, asked["raw_response"], images_dir, stage_config, finder, asked["upright_turn"]
        )
        if asked.get("error"):
            # The model could not be asked about this photograph, in a shared call or alone. It is
            # treated as one the model said nothing clear about, and says why.
            record["reason"] = f"the model could not be asked: {asked['error']}"
            record["error"] = asked["error"]
        acted_on += 1
        logger.info("[%d] %s -> %s (%s)", acted_on, photograph.name, record["action"], record["reason"])
        return record

    # One engine on each card where the model allows it, else one engine across the cards, as for every
    # other stage that uses the model. Either way the photographs are asked about by
    # ask_about_photographs, and each reply is acted on as it arrives, from whichever card.
    carried = carry_on(
        app_cfg,
        Work(
            stage="preprocess",
            output=records_path,
            items=photographs,
            name_of=lambda photograph: photograph.name,
            read=ask_about_photographs,
            taken_out=asked_again,
            made_of=act_on,
            timing=timing,
            thing="photograph",
        ),
    )
    if carried.arrangement is None:
        return images_dir

    records = carried.records
    logger.info("Engines: %s", carried.arrangement)
    logger.info(
        "Preprocess questions complete: %d photographs in %.1fs (%.2fs/photograph)",
        len(records),
        carried.model_seconds,
        carried.model_seconds / len(records),
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


BANNER = "Stage 0: preprocess, one upright, level image for each document in a photograph (GPU)"


def start(task: StageSettings, checked: Checked) -> str:
    """Run the stage for a task, and give the line that says it is done.

    Started afresh, the stage first deletes its earlier records and the images they name. Until
    26 November 2026 it deleted nothing: an image an earlier run had cut was then read by the stages
    after this one, beside whatever the new run made of the same photograph.
    """
    task.preprocess_dir.mkdir(parents=True, exist_ok=True)
    deleted = start_afresh(task.clear_previous_output, task.preprocess_dir / RECORDS_FILE)
    _remove_the_images_of(deleted, task.preprocess_dir / IMAGES_DIR)
    images = run(
        task.preprocess_dir,
        checked,
        # The row of the stage's time goes with the other stages', in the run's output folder.
        timing=task.output,
    )
    return f"Preprocess complete. Later stages read their images from: {images}"
