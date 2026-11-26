"""A photograph is shown to the model the way up its orientation tag says it is.

A phone or camera held on its side stores the picture as its sensor saw it and adds a tag (EXIF
Orientation, one of eight values) saying how to turn it for viewing. The tag belongs to the EXIF standard
and not to any maker, so the files here are written by Pillow itself, in each form the pipeline accepts
that can carry one.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from models.orchestrator import DocumentOrchestrator

ORIENTATION = 0x0112
# How to get from the page as it should be viewed to the pixels a device would store under each tag value.
STORED_AS = {
    1: None,
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_90,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_270,
}


def page() -> Image.Image:
    """A page taller than it is wide, with a different colour in each corner so any turn or flip shows."""
    pixels = np.full((300, 200, 3), 255, np.uint8)
    pixels[:60, :60] = (200, 30, 30)
    pixels[:60, -60:] = (30, 160, 30)
    pixels[-60:, :60] = (30, 30, 200)
    pixels[-60:, -60:] = (20, 20, 20)
    return Image.fromarray(pixels)


def stored(path: Path, tag: int | None) -> Path:
    """The page saved as a device would store it under an orientation tag, or with no tag at all."""
    upright = page()
    if tag is None:
        upright.save(path)
        return path
    turn = STORED_AS[tag]
    pixels = upright if turn is None else upright.transpose(turn)
    exif = Image.Exif()
    exif[ORIENTATION] = tag
    # WebP is asked to keep its pixels exactly, as PNG and TIFF do, so the page can be compared whole.
    if path.suffix == ".webp":
        pixels.save(path, exif=exif, lossless=True)
    else:
        pixels.save(path, exif=exif)
    return path


def loaded(path: Path) -> Image.Image:
    return DocumentOrchestrator.load_document_image(object.__new__(DocumentOrchestrator), str(path))


@pytest.mark.parametrize("tag", sorted(STORED_AS))
@pytest.mark.parametrize("suffix", [".png", ".tiff", ".webp"])
def test_every_value_of_the_tag_gives_the_page_the_right_way_up(
    tmp_path: Path, suffix: str, tag: int
) -> None:
    path = stored(tmp_path / f"page{suffix}", tag)

    image = loaded(path)

    assert image.size == (200, 300)
    assert np.array_equal(np.asarray(image.convert("RGB")), np.asarray(page()))


@pytest.mark.parametrize("tag", sorted(STORED_AS))
def test_a_jpeg_is_turned_by_its_tag_too(tmp_path: Path, tag: int) -> None:
    """A JPEG does not keep its pixels exactly, so each corner is checked for its colour."""
    image = loaded(stored(tmp_path / "page.jpg", tag)).convert("RGB")

    assert image.size == (200, 300)
    corners = np.asarray(image)[[20, 20, -20, -20], [20, -20, 20, -20]].astype(int)
    wanted = np.array([(200, 30, 30), (30, 160, 30), (30, 30, 200), (20, 20, 20)])
    assert np.abs(corners - wanted).max() < 40


def test_a_photograph_with_no_tag_is_shown_as_it_is_stored(tmp_path: Path) -> None:
    """With no tag there is nothing to say which way up the page is: the pixels are left alone."""
    sideways = page().transpose(Image.Transpose.ROTATE_90)
    sideways.save(tmp_path / "page.png")

    image = loaded(tmp_path / "page.png")

    assert image.size == (300, 200)
    assert np.array_equal(np.asarray(image.convert("RGB")), np.asarray(sideways))
