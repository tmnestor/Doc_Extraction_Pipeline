"""A photograph is cut up only on a clear answer that it holds several receipts.

The model is not run here: its replies are written out, in the form it gives. The photographs are plain images
made in the test, with a coloured patch where each receipt would be, so a cut image can be checked by colour.
"""

import dataclasses
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import yaml
from PIL import Image

from common.photo_split import (
    Box,
    PhotoSplitConfig,
    PhotoSplitConfigError,
    box_within,
    crop_rectangles,
    cut,
    decide,
    level_turn,
    parse_boxes,
    photo_split_config_from,
    rectify,
    row_direction,
    row_measures,
    turn,
    worth_straightening,
)
from common.sheet_finder_config import load_finder_params
from stages.split import apply_reply, checked, cut_name
from stages.split import run as run_the_stage

RUN_CONFIG = Path(__file__).resolve().parents[2] / "config" / "run_config.yml"
RAW = yaml.safe_load(RUN_CONFIG.read_text())
CONFIG = photo_split_config_from(RAW, config_path=RUN_CONFIG)
RED, GREEN, DESK = (200, 40, 40), (40, 160, 40), (120, 120, 120)

# The replies below are in the form the model gave on real photographs.
TWO = (
    '```json\n[\n  {"box_2d": [100, 100, 900, 400], "label": "receipt"},\n'
    '  {"box_2d": [120, 520, 880, 760], "label": "receipt"}\n]\n```'
)
ONE = '```json\n[\n  {"box_2d": [131, 363, 923, 621], "label": "receipt"}\n]\n```'


def run(image_dir: Path, output_dir: Path) -> Path:
    """The stage as shipped: the photo split is off, so no model is asked and none is looked for."""

    def never() -> Any:
        raise AssertionError("with the photo split off the run's settings are not asked for")

    return run_the_stage(output_dir, checked(RAW, RUN_CONFIG, never, image_dir))


def reply_for(*boxes: list[int]) -> str:
    return json.dumps([{"box_2d": box, "label": "receipt"} for box in boxes])


def photograph(path: Path) -> Path:
    """A 1000 by 1000 desk with a red receipt on the left and a green one on the right, as TWO describes."""
    image = Image.new("RGB", (1000, 1000), DESK)
    image.paste(RED, (100, 100, 400, 900))
    image.paste(GREEN, (520, 120, 760, 880))
    image.save(path)
    return path


def test_a_reply_with_two_boxes_is_read_in_the_models_own_order_of_coordinates() -> None:
    assert parse_boxes(TWO) == [Box(100, 100, 400, 900), Box(520, 120, 760, 880)]
    assert parse_boxes(ONE) == [Box(363, 131, 621, 923)]


@pytest.mark.parametrize(
    "reply",
    [
        "There are two receipts in this photograph.",
        "2",
        '{"count": 2}',
        '[{"label": "receipt"}]',
        '[{"box_2d": [100, 100, 900]}]',
        '[{"box_2d": [100, 100, 900, "400"]}]',
        '[{"box_2d": [900, 100, 100, 400]}]',
        '[{"box_2d": [100, 100, 900, 1400]}]',
        '[{"box_2d": [100, 100, 900, 400]}, "receipt"]',
        "",
    ],
)
def test_a_reply_that_is_not_the_boxes_asked_for_is_not_read(reply: str) -> None:
    assert parse_boxes(reply) is None
    assert decide(reply, CONFIG).action == "pass_through"
    assert "not the list of boxes" in decide(reply, CONFIG).reason


def test_two_clear_boxes_mean_split() -> None:
    decision = decide(TWO, CONFIG)

    assert (decision.action, decision.reason, len(decision.boxes)) == ("split", "2 receipts", 2)


def test_one_receipt_is_passed_through() -> None:
    decision = decide(ONE, CONFIG)

    assert (decision.action, decision.reason) == ("pass_through", "one receipt")


def test_no_receipt_is_passed_through() -> None:
    assert decide("[]", CONFIG).reason == "no receipt was found"


def test_a_speck_of_a_box_is_ignored_so_one_receipt_and_a_speck_is_one_receipt() -> None:
    reply = reply_for([100, 100, 900, 400], [500, 600, 520, 640])

    assert decide(reply, CONFIG).reason == "one receipt"


def test_boxes_that_overlap_are_not_trusted() -> None:
    reply = reply_for([100, 100, 900, 400], [200, 300, 800, 700])

    assert (decide(reply, CONFIG).action, decide(reply, CONFIG).reason) == (
        "pass_through",
        "two boxes overlap",
    )


def test_boxes_that_only_touch_at_an_edge_are_trusted() -> None:
    reply = reply_for([100, 100, 900, 400], [100, 400, 900, 700])

    assert decide(reply, CONFIG).action == "split"


def test_more_boxes_than_is_plausible_are_not_trusted() -> None:
    boxes = [[10 + 70 * n, 100, 60 + 70 * n, 900] for n in range(CONFIG.max_receipts + 1)]

    decision = decide(reply_for(*boxes), CONFIG)

    assert decision.action == "pass_through" and "more than max_receipts" in decision.reason


def test_each_cut_is_its_box_and_a_margin() -> None:
    boxes = (Box(100, 100, 400, 900), Box(520, 120, 760, 880))

    left, right = crop_rectangles(boxes, 2000, 1000, 0.05)

    # The left box is 300 wide and 800 high on the 0 to 1000 scale: 15 and 40 of margin, on a 2000 by 1000 photo.
    assert left == (170, 60, 830, 940)
    assert right == (1016, 82, 1544, 918)


def test_a_margin_stops_at_the_photographs_edge_and_halfway_to_a_neighbour() -> None:
    boxes = (Box(0, 0, 480, 1000), Box(500, 0, 1000, 1000))

    left, right = crop_rectangles(boxes, 1000, 1000, 0.10)

    assert left == (0, 0, 490, 1000)
    assert right == (490, 0, 1000, 1000)


def test_a_margin_stops_halfway_to_a_neighbour_above_or_below() -> None:
    boxes = (Box(100, 0, 900, 480), Box(100, 500, 900, 1000))

    upper, lower = crop_rectangles(boxes, 1000, 1000, 0.10)

    assert upper[3] == 490 and lower[1] == 490


def colours_of(image: Image.Image) -> set[Any]:
    """Every colour present in a small test image."""
    return {colour for _, colour in image.getcolors() or []}


def darkest_channel(image: Image.Image, x: int, y: int) -> int:
    """The lowest of a pixel's red, green and blue values."""
    pixel = image.getpixel((x, y))
    assert isinstance(pixel, tuple)
    return int(min(pixel))


def test_each_cut_image_shows_its_own_receipt_and_none_of_its_neighbour(tmp_path: Path) -> None:
    image = Image.open(photograph(tmp_path / "photo.png"))

    left, right = cut(image, decide(TWO, CONFIG).boxes, CONFIG.margin_frac)

    assert RED in colours_of(left) and GREEN not in colours_of(left)
    assert GREEN in colours_of(right) and RED not in colours_of(right)
    # The left box is 300 by 800; a margin of 0.10 of each adds 30 and 80 on every side.
    assert CONFIG.margin_frac == 0.10 and left.size == (360, 960)


def test_a_photograph_of_two_is_written_as_two_images_and_recorded(tmp_path: Path) -> None:
    source = photograph(tmp_path / "IMG_1.png")
    images = tmp_path / "images"
    images.mkdir()

    record = apply_reply(source, TWO, images, CONFIG)

    assert record["action"] == "split" and record["reason"] == "2 receipts"
    assert (
        record["outputs"] == ["IMG_1__r1.png", "IMG_1__r2.png"] == sorted(p.name for p in images.iterdir())
    )
    assert record["boxes"] == [[100, 100, 400, 900], [520, 120, 760, 880]]
    assert record["image_name"] == "IMG_1.png" and record["raw_response"] == TWO
    assert cut_name(source, 2) == "IMG_1__r2.png"


def test_a_photograph_of_one_is_copied_through_byte_for_byte(tmp_path: Path) -> None:
    source = photograph(tmp_path / "IMG_2.png")
    images = tmp_path / "images"
    images.mkdir()

    record = apply_reply(source, ONE, images, CONFIG)

    assert record["action"] == "pass_through" and record["outputs"] == ["IMG_2.png"]
    assert (images / "IMG_2.png").read_bytes() == source.read_bytes()


def test_the_stage_as_shipped_copies_every_photograph_through_without_a_model(tmp_path: Path) -> None:
    """Shipped with enabled false: the stage changes nothing, and never loads the model."""
    assert CONFIG.enabled is False
    data = tmp_path / "data"
    data.mkdir()
    sources = [photograph(data / f"CASE00{n}_bank_statement.png") for n in (1, 2)]

    images = run(data, tmp_path / "out")

    assert sorted(p.name for p in images.iterdir()) == [p.name for p in sources]
    assert all((images / p.name).read_bytes() == p.read_bytes() for p in sources)
    records = [json.loads(line) for line in (tmp_path / "out" / "splits.jsonl").read_text().splitlines()]
    assert [r["action"] for r in records] == ["pass_through", "pass_through"]
    assert {r["reason"] for r in records} == {"photo_split is not enabled"}


def test_a_second_run_does_not_repeat_photographs_already_recorded(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    photograph(data / "CASE001_bank_statement.png")
    run(data, tmp_path / "out")
    photograph(data / "CASE002_bank_statement.png")

    run(data, tmp_path / "out")

    records = [json.loads(line) for line in (tmp_path / "out" / "splits.jsonl").read_text().splitlines()]
    assert [r["image_name"] for r in records] == [
        "CASE001_bank_statement.png",
        "CASE002_bank_statement.png",
    ]


def test_the_shipped_prompt_is_the_one_that_was_measured() -> None:
    assert CONFIG.prompt == (
        "Detect every paper receipt in this photograph. Reply with JSON only: a list with one object for each "
        'receipt, in the form {"box_2d": [ymin, xmin, ymax, xmax], "label": "receipt"}, where the four numbers '
        "are normalised to 0-1000."
    )


def _raw(mutate) -> dict:
    raw = yaml.safe_load(RUN_CONFIG.read_text())
    mutate(raw["pipeline"]["photo_split"])
    return raw


@pytest.mark.parametrize(
    ("mutate", "said"),
    [
        (lambda s: s.pop("enabled"), "missing key enabled"),
        (lambda s: s.pop("prompt"), "missing key prompt"),
        (lambda s: s.pop("max_receipts"), "missing key max_receipts"),
        (lambda s: s.update(enabled="yes"), "enabled is 'yes', which is not true or false"),
        (lambda s: s.update(max_tokens=400.5), "max_tokens is 400.5, which is not int"),
        (lambda s: s.update(batch_size=True), "batch_size is True, which is not int"),
        (lambda s: s.update(margin_frac="wide"), "margin_frac is 'wide', which is not float"),
        (lambda s: s.update(prompt="   "), "prompt is empty"),
        (lambda s: s.update(max_tokens=0), "max_tokens is 0, which is below 1"),
        (lambda s: s.update(max_receipts=1), "max_receipts is 1, which is below 2"),
        (lambda s: s.update(margin_frac=0.9), "margin_frac is 0.9, which is outside 0.0 to 0.5"),
        (
            lambda s: s.update(max_overlap_frac=-0.1),
            "max_overlap_frac is -0.1, which is outside 0.0 to 1.0",
        ),
    ],
)
def test_a_missing_or_malformed_setting_is_refused_with_all_four_elements(mutate, said: str) -> None:
    with pytest.raises(PhotoSplitConfigError) as raised:
        photo_split_config_from(_raw(mutate), config_path=RUN_CONFIG)
    message = str(raised.value)
    assert said in message
    assert all(element in message for element in ("What:", "Where:", "Expected:", "How to fix:"))
    assert str(RUN_CONFIG.resolve()) in message and "pipeline.photo_split" in message


def test_a_config_without_the_section_is_refused() -> None:
    with pytest.raises(PhotoSplitConfigError, match="section is missing"):
        photo_split_config_from({"pipeline": {}}, config_path=RUN_CONFIG)


def test_the_settings_are_what_the_stage_reads() -> None:
    assert isinstance(CONFIG, PhotoSplitConfig)
    assert (CONFIG.max_tokens, CONFIG.batch_size, CONFIG.max_receipts) == (400, 16, 12)


# ---- straightening -----------------------------------------------------------------------------------------

FINDER = load_finder_params(RUN_CONFIG)
# Turning alone: rectification, which is tried first for a whole photograph and for a cut image, is off.
TURN_ONLY = dataclasses.replace(CONFIG, rectify_pages=False, rectify_pieces=False)


def printed_receipt(turn_deg: float) -> Image.Image:
    """A white receipt printed with rows of letter-sized marks, turned so its rows fall `turn_deg` to the right."""
    page = np.full((560, 280, 3), 245, np.uint8)
    for row in range(30, 530, 20):
        for column in range(20, 250, 9):
            if (row + column) % 7:
                cv2.rectangle(page, (column, row), (column + 5, row + 10), (30, 30, 30), -1)
    # OpenCV turns anticlockwise for a positive angle, and rows that fall to the right are a clockwise turn.
    matrix = cv2.getRotationMatrix2D((140, 280), -turn_deg, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    width, height = int(560 * sin + 280 * cos), int(560 * cos + 280 * sin)
    matrix[0, 2] += width / 2 - 140
    matrix[1, 2] += height / 2 - 280
    turned = cv2.warpAffine(page, matrix, (width, height), borderValue=DESK)
    return Image.fromarray(turned)


def photograph_of(path: Path, *turns: float) -> tuple[Path, str]:
    """A desk with one printed receipt per turn, side by side, and the reply a model would give for it."""
    image = Image.new("RGB", (500 * len(turns), 800), DESK)
    boxes = []
    for index, turn_deg in enumerate(turns):
        receipt = printed_receipt(turn_deg)
        left, top = 500 * index + (500 - receipt.width) // 2, (800 - receipt.height) // 2
        image.paste(receipt, (left, top))
        boxes.append(
            [
                round(1000 * top / 800),
                round(1000 * left / image.width),
                round(1000 * (top + receipt.height) / 800),
                round(1000 * (left + receipt.width) / image.width),
            ]
        )
    image.save(path)
    return path, reply_for(*boxes)


@pytest.mark.parametrize("turn_deg", [-18.0, -6.0, 9.0, 22.0])
def test_the_direction_of_the_rows_is_found_with_its_sign(turn_deg: float) -> None:
    assert row_direction(printed_receipt(turn_deg), FINDER) == pytest.approx(turn_deg, abs=1.0)


@pytest.mark.parametrize("turn_deg", [-12.0, 9.0, 14.0])
def test_a_turned_image_comes_out_with_its_rows_level(turn_deg: float) -> None:
    receipt = printed_receipt(turn_deg)

    level = turn(receipt, level_turn(row_direction(receipt, FINDER), CONFIG), CONFIG)

    after = row_direction(level, FINDER)
    assert after is not None and abs(after) < 1.0
    # The image grew to hold its corners, which are the fill colour.
    assert level.width > receipt.width or level.height > receipt.height
    assert level.getpixel((0, 0)) == CONFIG.straighten_fill_rgb


def test_a_turn_too_small_or_too_large_to_believe_is_left_alone() -> None:
    assert level_turn(None, CONFIG) == 0.0
    assert level_turn(CONFIG.straighten_min_deg / 2, CONFIG) == 0.0
    assert level_turn(-CONFIG.straighten_max_deg - 1, CONFIG) == 0.0
    assert level_turn(-12.5, CONFIG) == -12.5
    # Real receipts lie within a few degrees of level, so a large direction is a mismeasurement.
    assert CONFIG.straighten_max_deg == 15.0 and level_turn(22.0, CONFIG) == 0.0
    image = printed_receipt(0.0)
    assert turn(image, 0.0, CONFIG) is image


def test_each_cut_image_is_turned_level_and_the_turn_is_recorded(tmp_path: Path) -> None:
    source, reply = photograph_of(tmp_path / "IMG_3.png", 14.0, -9.0)
    images = tmp_path / "images"
    images.mkdir()

    record = apply_reply(source, reply, images, TURN_ONLY, FINDER)

    assert record["action"] == "split" and record["outputs"] == ["IMG_3__r1.png", "IMG_3__r2.png"]
    assert record["rectified"] is False and record["rectified_each"] == [False, False]
    assert record["turned_deg"] == pytest.approx([14.0, -9.0], abs=1.0)
    for name in record["outputs"]:
        after = row_direction(Image.open(images / name), FINDER)
        assert after is not None and abs(after) < 1.0


def test_a_photograph_of_one_turned_receipt_is_turned_whole(tmp_path: Path) -> None:
    source, reply = photograph_of(tmp_path / "IMG_4.png", 12.0)
    images = tmp_path / "images"
    images.mkdir()

    record = apply_reply(source, reply, images, TURN_ONLY, FINDER)

    assert record["action"] == "pass_through" and record["reason"] == "one receipt"
    assert record["outputs"] == ["IMG_4.png"] and record["turned_deg"] == pytest.approx([12.0], abs=1.0)
    written = Image.open(images / "IMG_4.png")
    # Nothing is cut away: the whole photograph is there, turned, so it is larger than it was.
    assert written.width > 500 and written.height > 800
    after = row_direction(written, FINDER)
    assert after is not None and abs(after) < 1.0


def test_a_photograph_of_one_level_receipt_is_still_copied_byte_for_byte(tmp_path: Path) -> None:
    source, reply = photograph_of(tmp_path / "IMG_5.png", 0.0)
    images = tmp_path / "images"
    images.mkdir()

    record = apply_reply(source, reply, images, TURN_ONLY, FINDER)

    assert record["turned_deg"] == [0.0]
    assert (images / "IMG_5.png").read_bytes() == source.read_bytes()


def test_with_straightening_off_nothing_is_turned(tmp_path: Path) -> None:
    source, reply = photograph_of(tmp_path / "IMG_6.png", 14.0, -9.0)
    images = tmp_path / "images"
    images.mkdir()

    record = apply_reply(
        source, reply, images, dataclasses.replace(CONFIG, straighten=False, rectify_pages=False), FINDER
    )

    assert record["turned_deg"] == [0.0, 0.0]
    first = row_direction(Image.open(images / "IMG_6__r1.png"), FINDER)
    assert first == pytest.approx(14.0, abs=1.0)


@pytest.mark.parametrize(
    ("mutate", "said"),
    [
        (lambda s: s.pop("straighten"), "missing key straighten"),
        (lambda s: s.pop("straighten_max_deg"), "missing key straighten_max_deg"),
        (lambda s: s.update(straighten="yes"), "straighten is 'yes', which is not true or false"),
        (lambda s: s.update(straighten_min_deg=50.0), "they must run from 0 up to at most 90"),
        (lambda s: s.update(straighten_max_deg=120.0), "they must run from 0 up to at most 90"),
        (lambda s: s.pop("straighten_fill_rgb"), "straighten_fill_rgb is None"),
        (lambda s: s.update(straighten_fill_rgb=[255, 255]), "not three whole numbers from 0 to 255"),
        (lambda s: s.update(straighten_fill_rgb=[255, 255, 300]), "not three whole numbers from 0 to 255"),
    ],
)
def test_a_missing_or_malformed_straightening_setting_is_refused(mutate, said: str) -> None:
    with pytest.raises(PhotoSplitConfigError) as raised:
        photo_split_config_from(_raw(mutate), config_path=RUN_CONFIG)
    message = str(raised.value)
    assert said in message
    assert all(element in message for element in ("What:", "Where:", "Expected:", "How to fix:"))


def test_a_page_with_no_receipt_found_is_turned_whole_by_all_its_print(tmp_path: Path) -> None:
    """A photographed page of another kind, such as a bank statement: the model finds no receipt in it."""
    source, _ = photograph_of(tmp_path / "IMG_7.png", -11.0)
    images = tmp_path / "images"
    images.mkdir()

    record = apply_reply(source, "[]", images, TURN_ONLY, FINDER)

    assert record["action"] == "pass_through" and record["reason"] == "no receipt was found"
    assert record["turned_deg"] == pytest.approx([-11.0], abs=1.0)
    after = row_direction(Image.open(images / "IMG_7.png"), FINDER)
    assert after is not None and abs(after) < 1.0


def test_a_level_page_with_no_receipt_found_is_copied_byte_for_byte(tmp_path: Path) -> None:
    source, _ = photograph_of(tmp_path / "IMG_8.png", 0.0)
    images = tmp_path / "images"
    images.mkdir()

    record = apply_reply(source, "[]", images, TURN_ONLY, FINDER)

    assert record["turned_deg"] == [0.0]
    assert (images / "IMG_8.png").read_bytes() == source.read_bytes()


def test_a_photograph_whose_several_boxes_are_not_trusted_is_left_as_it_lies(tmp_path: Path) -> None:
    """Overlapping boxes may be sheets at different angles, so no one turn is right for the photograph."""
    source, _ = photograph_of(tmp_path / "IMG_9.png", 14.0, -9.0)
    images = tmp_path / "images"
    images.mkdir()
    overlapping = reply_for([100, 100, 900, 400], [200, 300, 800, 700])

    record = apply_reply(source, overlapping, images, CONFIG, FINDER)

    assert record["reason"] == "two boxes overlap" and record["turned_deg"] == [0.0]
    assert (images / "IMG_9.png").read_bytes() == source.read_bytes()


# ---- rectification -----------------------------------------------------------------------------------------


def photographed_page(
    turn_deg: float,
    *,
    squeeze: int = 60,
    size: tuple[int, int] = (1000, 1400),
    surface: np.ndarray | None = None,
) -> Image.Image:
    """A printed page on a desk, turned and foreshortened: its top edge is `squeeze` pixels narrower each side.

    The desk is one plain colour unless `surface` gives another, as an image of the frame's size.
    """
    page = np.full((900, 600, 3), 245, np.uint8)
    for row in range(40, 860, 22):
        for column in range(30, 560, 9):
            if (row + column) % 7:
                cv2.rectangle(page, (column, row), (column + 5, row + 10), (30, 30, 30), -1)
    width, height = size
    theta = np.radians(turn_deg)
    rotate = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    flat = np.array([[0, 0], [600, 0], [600, 900], [0, 900]], np.float32)
    quad = flat.copy()
    quad[0][0] += squeeze
    quad[1][0] -= squeeze
    quad = ((quad - [300, 450]) @ rotate.T + [width / 2, height / 2]).astype(np.float32)
    desk = np.full((height, width, 3), DESK, np.uint8) if surface is None else surface
    warped = cv2.warpPerspective(
        np.dstack([page, np.full(page.shape[:2], 255, np.uint8)]),
        cv2.getPerspectiveTransform(flat, quad),
        (width, height),
        flags=cv2.INTER_CUBIC,
    )
    alpha = warped[:, :, 3:4].astype(float) / 255.0
    return Image.fromarray((desk * (1 - alpha) + warped[:, :, :3] * alpha).astype(np.uint8))


def clean_statement() -> Image.Image:
    """A page that is not a photograph: a heading, then a table inside a ruled border."""
    page = np.full((1400, 1000, 3), 255, np.uint8)
    # A heading of well-spaced lines, as a statement has: too slight for the sheet finder to take for a sheet,
    # but a tenth of the page's print.
    for row in range(60, 260, 40):
        for column in range(80, 620, 9):
            cv2.rectangle(page, (column, row), (column + 2, row + 10), (30, 30, 30), -1)
    # The table is lightly shaded inside a ruled border, as the statements' transaction tables are.
    cv2.rectangle(page, (70, 330), (930, 1330), (226, 226, 226), -1)
    cv2.rectangle(page, (70, 330), (930, 1330), (40, 40, 40), 3)
    for row in range(360, 1300, 24):
        for column in range(90, 900, 9):
            if (row + column) % 5:
                cv2.rectangle(page, (column, row), (column + 2, row + 10), (30, 30, 30), -1)
    return Image.fromarray(page)


def test_a_photographed_page_is_rectified_upright_at_its_own_size() -> None:
    photo = photographed_page(9.0)

    page = rectify(photo, FINDER, CONFIG)

    assert page is not None
    # The page is 600 by 900; foreshortened, its sides measure a little under that.
    assert abs(page.width - 600) <= 12 and abs(page.height - 900) <= 12
    after = row_direction(page, FINDER)
    assert after is not None and abs(after) < 0.6
    # Paper to every corner: no desk is left.
    for x, y in ((8, 8), (page.width - 9, 8), (8, page.height - 9), (page.width - 9, page.height - 9)):
        assert darkest_channel(page, x, y) > 200


def test_rectifying_removes_the_foreshortening_that_turning_leaves() -> None:
    """Foreshortened, the page's left and right edges lean towards each other. Turning cannot undo that."""
    photo = photographed_page(0.0, squeeze=70)

    page = rectify(photo, FINDER, CONFIG)

    assert page is not None

    def left_margin(image: Image.Image, y: int) -> int:
        row = np.asarray(image.convert("L"))[y]
        return int(np.argmax(row < 120))

    # In the photograph the print starts further in near the top than near the bottom; rectified, it does not.
    lean_before = left_margin(photo, 330) - left_margin(photo, 1050)
    lean_after = left_margin(page, 80) - left_margin(page, page.height - 80)
    assert lean_before > 30 and abs(lean_after) <= 6


def test_a_clean_page_is_never_cropped_to_its_ruled_table() -> None:
    """The fault this guards against: a table's border taken for the page's edge, and the heading cut off."""
    page = clean_statement()

    assert rectify(page, FINDER, CONFIG) is None
    # With the print test switched off, the table is taken for a sheet: this is what the test is for.
    without = dataclasses.replace(CONFIG, rectify_min_print_frac=0.01)
    cropped = rectify(page, FINDER, without)
    assert cropped is not None and cropped.height < 1100


def test_a_small_sheet_is_not_rectified() -> None:
    """A receipt covering little of the frame: its corners are not relied on, and it is turned instead."""
    photo = photographed_page(9.0, size=(2400, 2400))

    assert rectify(photo, FINDER, CONFIG) is None


def test_a_photograph_of_one_page_is_rectified_and_recorded(tmp_path: Path) -> None:
    source = tmp_path / "IMG_10.png"
    photographed_page(-7.0).save(source)
    images = tmp_path / "images"
    images.mkdir()

    record = apply_reply(source, "[]", images, CONFIG, FINDER)

    assert (
        record["action"] == "pass_through" and record["rectified"] is True and record["turned_deg"] == [0.0]
    )
    written = Image.open(images / "IMG_10.png")
    assert abs(written.width - 600) <= 12 and abs(written.height - 900) <= 12


def test_a_clean_page_goes_through_the_stage_byte_for_byte(tmp_path: Path) -> None:
    source = tmp_path / "IMG_11.png"
    clean_statement().save(source)
    images = tmp_path / "images"
    images.mkdir()

    record = apply_reply(source, "[]", images, CONFIG, FINDER)

    assert record["rectified"] is False and record["turned_deg"] == [0.0]
    assert (images / "IMG_11.png").read_bytes() == source.read_bytes()


def test_a_cut_receipt_is_rectified_as_a_photograph_of_one_receipt_is(tmp_path: Path) -> None:
    """Each cut image gets the same repair a whole photograph gets: its corners are found and it is set upright."""
    source, reply = photograph_of(tmp_path / "IMG_12.png", 14.0, -9.0)
    images = tmp_path / "images"
    images.mkdir()

    record = apply_reply(source, reply, images, CONFIG, FINDER)

    assert record["action"] == "split" and record["rectified"] is True
    assert record["rectified_each"] == [True, True] and record["turned_deg"] == [0.0, 0.0]
    for name in record["outputs"]:
        written = Image.open(images / name)
        # The receipt alone, at its own size (280 by 560), with its rows level.
        assert written.width == pytest.approx(280, abs=8) and written.height == pytest.approx(560, abs=8)
        after = row_direction(written, FINDER)
        assert after is not None and abs(after) < 1.0


def test_with_rectify_pieces_off_a_cut_receipt_is_turned_not_rectified(tmp_path: Path) -> None:
    source, reply = photograph_of(tmp_path / "IMG_12.png", 14.0, -9.0)
    images = tmp_path / "images"
    images.mkdir()

    record = apply_reply(source, reply, images, dataclasses.replace(CONFIG, rectify_pieces=False), FINDER)

    assert record["action"] == "split" and record["rectified"] is False
    assert record["turned_deg"] == pytest.approx([14.0, -9.0], abs=1.0)


def test_a_cut_receipt_lying_further_over_than_is_believed_is_not_turned(tmp_path: Path) -> None:
    """Past straighten_max_deg a measured direction is not acted on by turning: the image stays as cut."""
    source, reply = photograph_of(tmp_path / "IMG_13.png", 22.0, -9.0)
    images = tmp_path / "images"
    images.mkdir()

    record = apply_reply(source, reply, images, TURN_ONLY, FINDER)

    assert record["turned_deg"] == pytest.approx([0.0, -9.0], abs=1.0)
    with Image.open(source) as photo:
        (first, _) = cut(photo.convert("RGB"), decide(reply, CONFIG).boxes, CONFIG.margin_frac)
    assert np.array_equal(np.asarray(Image.open(images / "IMG_13__r1.png")), np.asarray(first))


def test_the_drift_of_the_rows_is_the_prints_width_times_the_tangent_of_their_angle() -> None:
    """The stand-in receipt's print is 230 px wide and 10 px tall: 23 text heights, or thereabouts."""
    direction, drift = row_measures(printed_receipt(9.0), FINDER)
    _, level = row_measures(printed_receipt(0.0), FINDER)

    assert direction == pytest.approx(9.0, abs=1.0)
    assert drift == pytest.approx(23 * np.tan(np.radians(9.0)), rel=0.25)
    assert level is not None and level < 0.5


def test_only_rows_that_drift_far_enough_to_mislead_a_reader_are_worth_straightening() -> None:
    limit = CONFIG.straighten_min_drift_heights

    assert limit == 2.0
    assert worth_straightening(limit, CONFIG) and worth_straightening(limit + 3, CONFIG)
    assert not worth_straightening(limit - 0.1, CONFIG) and not worth_straightening(0.0, CONFIG)
    # A drift that could not be measured is no reason to hold back.
    assert worth_straightening(None, CONFIG)


def test_a_narrow_receipt_at_a_small_tilt_is_left_exactly_as_it_is(tmp_path: Path) -> None:
    """At 3 degrees the stand-in's rows drift about 1.2 text heights: a reader stays on the row."""
    source, reply = photograph_of(tmp_path / "IMG_14.png", 3.0)
    images = tmp_path / "images"
    images.mkdir()

    record = apply_reply(source, reply, images, CONFIG, FINDER)

    assert record["rectified"] is False and record["turned_deg"] == [0.0]
    assert 0.5 < record["drift_heights"][0] < CONFIG.straighten_min_drift_heights
    assert (images / "IMG_14.png").read_bytes() == source.read_bytes()


def test_cut_receipts_at_a_small_tilt_are_left_as_cut(tmp_path: Path) -> None:
    source, reply = photograph_of(tmp_path / "IMG_15.png", 3.0, -2.0)
    images = tmp_path / "images"
    images.mkdir()

    record = apply_reply(source, reply, images, CONFIG, FINDER)

    assert record["action"] == "split" and record["rectified_each"] == [False, False]
    assert record["turned_deg"] == [0.0, 0.0]
    assert all(drift < CONFIG.straighten_min_drift_heights for drift in record["drift_heights"])
    with Image.open(source) as photo:
        pieces = cut(photo.convert("RGB"), decide(reply, CONFIG).boxes, CONFIG.margin_frac)
    for name, piece in zip(record["outputs"], pieces, strict=True):
        assert np.array_equal(np.asarray(Image.open(images / name)), np.asarray(piece))


def test_a_box_is_restated_on_the_scale_of_the_image_cut_around_it() -> None:
    box = Box(left=200, top=100, right=600, bottom=500)
    # A 2000 by 1000 photograph: the box is 400..1200 across and 100..500 down, cut with 100 px to spare.
    inner = box_within(box, (300, 0, 1300, 600), 2000, 1000)

    assert (inner.left, inner.right) == pytest.approx((100.0, 900.0))
    assert (inner.top, inner.bottom) == pytest.approx((1000 * 100 / 600, 1000 * 500 / 600))
    # A box that runs past the cut is kept inside it.
    outside = box_within(box, (500, 200, 1000, 400), 2000, 1000)
    assert (outside.left, outside.top, outside.right, outside.bottom) == (0.0, 0.0, 1000.0, 1000.0)


@pytest.mark.parametrize(
    ("mutate", "said"),
    [
        (lambda s: s.pop("rectify_pages"), "missing key rectify_pages"),
        (lambda s: s.pop("rectify_pieces"), "missing key rectify_pieces"),
        (lambda s: s.update(rectify_pieces="yes"), "rectify_pieces is 'yes', which is not true or false"),
        (
            lambda s: s.update(straighten=False, rectify_pages=False),
            "rectify_pieces is true but straighten is false",
        ),
        (lambda s: s.pop("rectify_min_print_frac"), "missing key rectify_min_print_frac"),
        (lambda s: s.update(rectify_min_edges=5), "rectify_min_edges is 5, which is not from 1 to 4"),
        (lambda s: s.update(rectify_min_edges=3.5), "rectify_min_edges is 3.5, which is not int"),
        (
            lambda s: s.update(rectify_min_area_frac=0.0),
            "rectify_min_area_frac is 0.0, which is not above 0",
        ),
        (
            lambda s: s.update(rectify_min_print_frac=1.5),
            "rectify_min_print_frac is 1.5, which is not above 0",
        ),
        (lambda s: s.update(straighten=False), "rectify_pages is true but straighten is false"),
    ],
)
def test_a_missing_or_malformed_rectification_setting_is_refused(mutate, said: str) -> None:
    with pytest.raises(PhotoSplitConfigError) as raised:
        photo_split_config_from(_raw(mutate), config_path=RUN_CONFIG)
    message = str(raised.value)
    assert said in message
    assert all(element in message for element in ("What:", "Where:", "Expected:", "How to fix:"))


def test_an_altered_jpeg_photograph_is_written_as_png_not_compressed_again(tmp_path: Path) -> None:
    """A photograph usually arrives as a JPEG. Turned or rectified, it must not be saved as a JPEG again."""
    source = tmp_path / "IMG_13.jpg"
    photographed_page(-7.0).save(source, quality=95)
    images = tmp_path / "images"
    images.mkdir()

    record = apply_reply(source, "[]", images, CONFIG, FINDER)

    assert record["rectified"] is True and record["outputs"] == ["IMG_13.png"]
    assert record["image_name"] == "IMG_13.jpg"
    assert [p.name for p in images.iterdir()] == ["IMG_13.png"]
    assert Image.open(images / "IMG_13.png").format == "PNG"


def test_a_turned_jpeg_photograph_is_written_as_png(tmp_path: Path) -> None:
    source = tmp_path / "IMG_14.jpg"
    photograph_of(tmp_path / "scratch.png", 12.0)
    Image.open(tmp_path / "scratch.png").save(source, quality=95)
    _, reply = photograph_of(tmp_path / "scratch.png", 12.0)
    images = tmp_path / "images"
    images.mkdir()

    record = apply_reply(source, reply, images, TURN_ONLY, FINDER)

    assert record["outputs"] == ["IMG_14.png"] and record["turned_deg"] == pytest.approx([12.0], abs=1.0)


def test_an_unaltered_jpeg_photograph_is_copied_under_its_own_name(tmp_path: Path) -> None:
    source = tmp_path / "IMG_15.jpg"
    clean_statement().save(source, quality=95)
    images = tmp_path / "images"
    images.mkdir()

    record = apply_reply(source, "[]", images, CONFIG, FINDER)

    assert record["outputs"] == ["IMG_15.jpg"] and record["rectified"] is False
    assert (images / "IMG_15.jpg").read_bytes() == source.read_bytes()


# ---- rectification by the paper method ---------------------------------------------------------------------
# The desk method needs a desk that is smooth and of one colour. These surfaces are neither.

NO_PAPER_METHOD = dataclasses.replace(CONFIG, rectify_paper_model=False)


def two_coloured_surface(size: tuple[int, int] = (1000, 1400)) -> np.ndarray:
    """A surface of two colours meeting under the page, softly mottled: a mat on a table, say."""
    width, height = size
    rng = np.random.default_rng(5)
    xx = np.mgrid[0:height, 0:width][1]
    base = np.where((xx < 420)[..., None], np.array([70.0, 74.0, 82.0]), np.array([92.0, 132.0, 100.0]))
    mottle = cv2.GaussianBlur(rng.normal(0, 40.0, (height, width, 3)), (0, 0), 6.0)
    return np.clip(base + mottle, 0, 255).astype(np.uint8)


def speckled_surface(size: tuple[int, int] = (1000, 1400)) -> np.ndarray:
    """A surface speckled as finely as print, like felt."""
    width, height = size
    rng = np.random.default_rng(6)
    speckle = cv2.GaussianBlur(rng.normal(0, 60.0, (height, width, 3)), (0, 0), 1.4)
    return np.clip(np.array([60.0, 118.0, 74.0]) + speckle, 0, 255).astype(np.uint8)


def test_a_page_on_a_two_coloured_surface_is_rectified_only_by_the_paper_method() -> None:
    photo = photographed_page(8.0, surface=two_coloured_surface())

    assert rectify(photo, FINDER, NO_PAPER_METHOD) is None
    page = rectify(photo, FINDER, CONFIG)

    assert page is not None
    assert abs(page.width - 600) <= 12 and abs(page.height - 900) <= 12
    after = row_direction(page, FINDER)
    assert after is not None and abs(after) < 0.6
    for x, y in ((8, 8), (page.width - 9, 8), (8, page.height - 9), (page.width - 9, page.height - 9)):
        assert darkest_channel(page, x, y) > 200  # paper to every corner: no surface is left


def test_the_paper_method_is_not_tried_when_the_desk_method_is_trusted() -> None:
    """On a plain desk both would do; the desk method's result is the one used, so nothing there changes."""
    photo = photographed_page(9.0)

    with_paper = rectify(photo, FINDER, CONFIG)
    without = rectify(photo, FINDER, NO_PAPER_METHOD)

    assert with_paper is not None and without is not None
    assert np.array_equal(np.asarray(with_paper), np.asarray(without))


def test_a_page_on_a_surface_speckled_like_print_is_not_trusted() -> None:
    """A known limit, held on purpose. Speckle as fine as print looks like print outside the sheet, which is
    also what a clean page's heading looks like beside its ruled table. So the page is turned, not rectified."""
    photo = photographed_page(8.0, surface=speckled_surface())

    assert rectify(photo, FINDER, CONFIG) is None


def test_a_clean_page_is_not_cropped_by_the_paper_method_either() -> None:
    """The shaded, ruled table is a sheet to the paper method; the heading left outside it says otherwise."""
    page = clean_statement()

    assert rectify(page, FINDER, CONFIG) is None
    without_print_test = dataclasses.replace(CONFIG, rectify_min_print_frac=0.01)
    assert rectify(page, FINDER, without_print_test) is not None  # so the print test is what holds it


def test_a_photograph_on_a_two_coloured_surface_is_rectified_by_the_stage(tmp_path: Path) -> None:
    source = tmp_path / "IMG_20.jpg"
    photographed_page(-6.0, surface=two_coloured_surface()).save(source, quality=95)
    images = tmp_path / "images"
    images.mkdir()
    reply = '```json\\n[{"box_2d": [150, 150, 850, 850], "label": "receipt"}]\\n```'

    record = apply_reply(source, reply, images, CONFIG, FINDER)

    assert record["action"] == "pass_through" and record["rectified"] is True
    assert record["outputs"] == ["IMG_20.png"]


@pytest.mark.parametrize(
    ("mutate", "said"),
    [
        (lambda s: s.pop("rectify_paper_model"), "missing key rectify_paper_model"),
        (lambda s: s.pop("rectify_paper_min_fit"), "missing key rectify_paper_min_fit"),
        (
            lambda s: s.update(rectify_paper_model="yes"),
            "rectify_paper_model is 'yes', which is not true or false",
        ),
        (
            lambda s: s.update(rectify_paper_min_fit=0.0),
            "rectify_paper_min_fit is 0.0, which is not above 0",
        ),
        (
            lambda s: s.update(rectify_paper_min_fit=1.2),
            "rectify_paper_min_fit is 1.2, which is not above 0",
        ),
    ],
)
def test_a_missing_or_malformed_paper_method_setting_is_refused(mutate, said: str) -> None:
    with pytest.raises(PhotoSplitConfigError) as raised:
        photo_split_config_from(_raw(mutate), config_path=RUN_CONFIG)
    message = str(raised.value)
    assert said in message
    assert all(element in message for element in ("What:", "Where:", "Expected:", "How to fix:"))
