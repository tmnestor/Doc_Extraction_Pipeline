"""A photograph that arrives turned a quarter or a half, with no orientation tag, is set the right way up.

Until 27 November 2026 nothing in the pipeline noticed such a photograph, and the model read it worse:
on 347 receipts the answers right fell from 99.1% to between 96.0% and 97.1%. Two things tell which way
up a page is, and neither can do it alone. The print itself tells a page on its side from one on end,
with no model: rows of print alternate with gaps down a page and not across it. It cannot tell the right
way up from upside down, and that the model is asked, in one question answered YES or NO.

The pages here are drawn in the test, and every reply is invented.
"""

import json
from pathlib import Path

import pytest
from PIL import Image

from common import photo_split
from common.errors import ConfigError
from common.photo_split import lies_on_its_side, said_upright, turned_upright
from common.prompt_catalog import PromptCatalog
from stages.split import apply_reply

CONFIG, FINDER = photo_split.CONFIG, photo_split.FINDER
RED, GREEN, DESK = (200, 40, 40), (40, 160, 40), (120, 120, 120)


def printed_page(tilt_px_per_100: float = 0.0) -> Image.Image:
    """A white page of print: rows of letter-sized marks, each row's letters at their own places."""
    page = Image.new("RGB", (400, 640), "white")
    for top in range(40, 600, 22):
        for left in range(24 + top % 9, 370, 9):
            if (top + left) % 7:
                drop = round(tilt_px_per_100 * left / 100)
                page.paste((30, 30, 30), (left, top + drop, left + 5, top + drop + 10))
    return page


# ---- the print tells a page on its side from one on end ------------------------------------------------


@pytest.mark.parametrize(("turned", "on_its_side"), [(0, False), (90, True), (180, False), (270, True)])
def test_a_page_on_its_side_is_told_from_one_on_end_by_its_print(turned: int, on_its_side: bool) -> None:
    page = printed_page().rotate(turned, expand=True)

    assert lies_on_its_side(page, FINDER, CONFIG) is on_its_side


def test_rows_tilted_a_few_degrees_are_still_told() -> None:
    """A fall of 8 px in 100 is about 4.6 degrees: across the page a row crosses its neighbours' bands."""
    tilted = printed_page(tilt_px_per_100=8.0)

    assert lies_on_its_side(tilted, FINDER, CONFIG) is False
    assert lies_on_its_side(tilted.rotate(90, expand=True), FINDER, CONFIG) is True


def test_an_image_with_no_print_is_left_as_it_lies() -> None:
    assert lies_on_its_side(Image.new("RGB", (300, 200), "white"), FINDER, CONFIG) is False


def test_a_quarter_turn_changes_no_pixel() -> None:
    page = printed_page()

    assert turned_upright(page, 0) is page
    assert turned_upright(turned_upright(page, 90), 270).tobytes() == page.tobytes()
    assert turned_upright(page, 90).size == (page.height, page.width)


# ---- the model says whether it is upside down ----------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "upright"),
    [
        ("YES", True),
        ("NO", False),
        ("No.", False),
        ("**Yes**", True),
        ("yes, it reads normally", True),
        ("", None),
        ("It cannot be told", None),
        ("NOT sure", None),
    ],
)
def test_the_models_answer_is_read_as_yes_no_or_neither(reply: str, upright: bool | None) -> None:
    assert said_upright(reply) is upright


def test_the_shipped_question_is_the_one_that_was_measured() -> None:
    """As put to the 31B on 27 November 2026: right on 1,024 of 1,024 pages upright or upside down."""
    assert PromptCatalog().photo_split_prompt("upright") == (
        "Is the printed text the right way up, so that it can be\n"
        "read without turning the image? Answer YES or NO."
    )


def test_a_file_with_no_question_of_which_way_up_is_refused(tmp_path: Path) -> None:
    (tmp_path / "photo_split.yaml").write_text("prompt: where is each receipt\n")

    with pytest.raises(ConfigError) as refused:
        PromptCatalog(tmp_path).photo_split_prompt("upright")

    message = str(refused.value)
    assert all(part in message for part in ("What:", "Where:", "Expected:", "How to fix:"))
    assert str(tmp_path / "photo_split.yaml") in message and "upright:" in message


# ---- what is written is the photograph the right way up ------------------------------------------------


def test_a_photograph_set_the_right_way_up_and_otherwise_left_is_written_as_png(tmp_path: Path) -> None:
    source = tmp_path / "page.jpg"
    page = printed_page()
    page.rotate(180).save(source, quality=95)
    images = tmp_path / "images"
    images.mkdir()

    record = apply_reply(source, "[]", images, CONFIG, FINDER, upright_turn=180)

    assert record["upright_turn"] == 180 and record["outputs"] == ["page.png"]
    assert record["action"] == "pass_through" and record["turned_deg"] == [0.0]
    with Image.open(source) as stored, Image.open(images / "page.png") as written:
        assert written.tobytes() == stored.convert("RGB").rotate(180).tobytes()


def test_a_photograph_that_needed_no_turn_is_still_copied_byte_for_byte(tmp_path: Path) -> None:
    source = tmp_path / "page.jpg"
    printed_page().save(source, quality=95)
    images = tmp_path / "images"
    images.mkdir()

    record = apply_reply(source, "[]", images, CONFIG, FINDER, upright_turn=0)

    assert record["upright_turn"] == 0 and record["outputs"] == ["page.jpg"]
    assert (images / "page.jpg").read_bytes() == source.read_bytes()


def test_the_boxes_are_of_the_photograph_the_right_way_up(tmp_path: Path) -> None:
    """A desk with a red receipt on the left and a green one on the right, stored upside down."""
    desk = Image.new("RGB", (1000, 1000), DESK)
    desk.paste(RED, (100, 100, 400, 900))
    desk.paste(GREEN, (520, 120, 760, 880))
    source = tmp_path / "two.png"
    desk.rotate(180).save(source)
    images = tmp_path / "images"
    images.mkdir()
    # The model was shown the photograph the right way up, and its boxes are of that.
    reply = json.dumps(
        [
            {"box_2d": [100, 100, 900, 400], "label": "receipt"},
            {"box_2d": [120, 520, 880, 760], "label": "receipt"},
        ]
    )

    record = apply_reply(source, reply, images, CONFIG, None, upright_turn=180)

    assert record["action"] == "split" and record["upright_turn"] == 180
    with Image.open(images / "two__r1.png") as first, Image.open(images / "two__r2.png") as second:
        assert first.getpixel((first.width // 2, first.height // 2)) == RED
        assert second.getpixel((second.width // 2, second.height // 2)) == GREEN
