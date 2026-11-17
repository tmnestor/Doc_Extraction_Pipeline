"""Split a photograph of several receipts into one image per receipt.

People often lay a few receipts on a table and take one photograph. Every later stage expects one document per
image, so such a photograph has to become one image per receipt first.

The vision model is asked where each receipt is, and replies with one box per receipt. What is done with the
reply is deliberately cautious:

- **One box, or none:** the photograph is passed through untouched. The model already reads a single
  photographed receipt as well as a clean one, so cropping it could only lose something.
- **Two or more boxes:** one image is cut per box, a little larger than the box.
- **A reply that cannot be trusted** (not the JSON asked for, boxes that overlap, more boxes than is
  plausible): the photograph is passed through untouched, and the reason is recorded.

So a photograph is only ever cut when the model has given a clear answer that there is more than one receipt,
and a single receipt always takes the path it took before this stage existed.

The boxes are upright rectangles, which suits receipts lying roughly upright in the photograph. A receipt turned
well away from upright next to another would bring part of its neighbour into its box.

Each image is then turned so its rows of print run level. A model follows a row across a wide gap (a label on
the left, its amount on the right) far less well on a turned page than on a level one: measured on receipts
turned up to 25 degrees, the amount on the GST line was read on 41% of them, against 100% of level ones. The
turn is found from the print itself (`common.sheet_finder.text_direction`). A photograph of one receipt is
turned whole, by the direction of the print inside its box; nothing is cut from it. A photograph in which no
receipt is found, such as a photographed bank statement, is turned whole by all of its print.

Turning removes a page's tilt but not its foreshortening: a page seen at an angle has rows that converge, and
on a wide row that is the same failure again. So a photograph that holds one document is first tried for
rectification: its four corners are found (`common.sheet_finder.find_sheets`) and the homography reversed,
which removes tilt and foreshortening together. That is done only when the corners can be trusted: exactly one
sheet found, covering enough of the frame, with enough of its edges confirmed by a sharp step, and holding
nearly all of the photograph's print (a ruled table on a clean page fails that last test). Otherwise the
photograph is turned, as above. Measured on 165 synthetic photographs of bank statements: 160 rectified, every
one with its corners within 0.1% of the page's diagonal, and 5 left to turning (pages running off the frame).

The reply's form is the one the model gives for detection prompts: a JSON list of objects, each with
`"box_2d": [ymin, xmin, ymax, xmax]` on a scale of 0 to 1000 across the photograph's height and width.

Every operating choice is in `pipeline.photo_split` of the run config. Every key is required.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from common.sheet_finder import FinderParams, find_sheets, print_marks, text_direction, warp_upright

SECTION = "pipeline.photo_split"

# The scale the model's box coordinates are on, whatever the photograph's size.
BOX_SCALE = 1000.0

_EXAMPLE = """\
pipeline:
  photo_split:
    enabled: false
    prompt: |
      Detect every paper receipt in this photograph. ...
    max_tokens: 400
    batch_size: 16
    margin_frac: 0.03
    min_box_frac: 0.01
    max_overlap_frac: 0.10
    max_receipts: 12
    straighten: true
    straighten_min_deg: 1.0
    straighten_max_deg: 40.0
    straighten_fill_rgb: [255, 255, 255]
    rectify_pages: true
    rectify_min_area_frac: 0.30
    rectify_min_edges: 4
    rectify_min_print_frac: 0.98"""

_KEYS: dict[str, type] = {
    "enabled": bool,
    "prompt": str,
    "max_tokens": int,
    "batch_size": int,
    "margin_frac": float,
    "min_box_frac": float,
    "max_overlap_frac": float,
    "max_receipts": int,
    "straighten": bool,
    "straighten_min_deg": float,
    "straighten_max_deg": float,
    "rectify_pages": bool,
    "rectify_min_area_frac": float,
    "rectify_min_edges": int,
    "rectify_min_print_frac": float,
}


class PhotoSplitConfigError(ValueError):
    """The photo split settings are missing or malformed."""


@dataclass(frozen=True)
class PhotoSplitConfig:
    """The `pipeline.photo_split` settings the stage reads.

    Attributes:
        enabled: Whether photographs are examined at all. When false every photograph is passed through and the
            model is not loaded.
        prompt: The question put to the model about each photograph.
        max_tokens: Longest reply allowed.
        batch_size: Photographs sent to the engine in one call.
        margin_frac: How much larger than its box each cut image is, on every side, as a share of the box's own
            width and height.
        min_box_frac: A box covering less than this share of the photograph is ignored.
        max_overlap_frac: Two boxes sharing more than this share of the smaller one mean the reply is not trusted.
        max_receipts: More boxes than this mean the reply is not trusted.
        straighten: Whether each image is turned so its rows of print run level.
        straighten_min_deg: A turn smaller than this is left alone.
        straighten_max_deg: A turn larger than this is not believed, and the image is left alone.
        straighten_fill_rgb: Colour of the corners a turned image gains.
        rectify_pages: Whether a photograph of one document is rectified from its corners when they can be
            trusted. Needs `straighten`, whose turning is the fallback.
        rectify_min_area_frac: The sheet found must cover at least this share of the photograph.
        rectify_min_edges: At least this many of the sheet's four edges must be confirmed by a sharp step.
        rectify_min_print_frac: At least this share of the photograph's print must lie inside the sheet.
    """

    enabled: bool
    prompt: str
    max_tokens: int
    batch_size: int
    margin_frac: float
    min_box_frac: float
    max_overlap_frac: float
    max_receipts: int
    straighten: bool
    straighten_min_deg: float
    straighten_max_deg: float
    straighten_fill_rgb: tuple[int, int, int]
    rectify_pages: bool
    rectify_min_area_frac: float
    rectify_min_edges: int
    rectify_min_print_frac: float


@dataclass(frozen=True)
class Box:
    """An upright rectangle on the model's 0 to 1000 scale."""

    left: float
    top: float
    right: float
    bottom: float

    @property
    def area(self) -> float:
        return (self.right - self.left) * (self.bottom - self.top)


@dataclass(frozen=True)
class Decision:
    """What to do with one photograph.

    Attributes:
        action: "split" or "pass_through".
        reason: Why, in a few words, for the record.
        boxes: The boxes the decision rests on (those kept after small ones were ignored).
    """

    action: str
    reason: str
    boxes: tuple[Box, ...]


def _fail(what: str, *, key: str, config_path: Path, fix: str) -> PhotoSplitConfigError:
    where = f"{config_path.resolve()} -> {SECTION}{'.' + key if key else ''}"
    return PhotoSplitConfigError(
        f"What: {what}\n"
        f"  Where: {where}\n"
        f"  Expected: every one of {', '.join(_KEYS)}, e.g.\n{_EXAMPLE}\n"
        f"  How to fix: {fix}"
    )


def photo_split_config_from(raw_config: dict[str, Any], *, config_path: Path) -> PhotoSplitConfig:
    """Read the photo split settings from a parsed run config.

    Args:
        raw_config: The whole parsed run config.
        config_path: The file it came from, named in each error.

    Returns:
        The settings.

    Raises:
        PhotoSplitConfigError: The section or a key is missing, or a value is of the wrong type or out of range.
    """
    section: Any = raw_config
    for part in SECTION.split("."):
        section = section.get(part) if isinstance(section, dict) else None
    if not isinstance(section, dict):
        raise _fail(
            f"the {SECTION} section is missing or is not a mapping",
            key="",
            config_path=config_path,
            fix=f"add the {SECTION} block to {config_path.name}, as shipped in the repository's run_config.yml",
        )
    values: dict[str, Any] = {}
    for name, kind in _KEYS.items():
        if name not in section:
            raise _fail(
                f"missing key {name}",
                key=name,
                config_path=config_path,
                fix=f"add `{name}:` under {SECTION} in {config_path.name}",
            )
        value = section[name]
        whole = isinstance(value, int) and not isinstance(value, bool)
        right_type = (
            isinstance(value, bool)
            if kind is bool
            else whole
            if kind is int
            else (whole or isinstance(value, float))
            if kind is float
            else isinstance(value, str)
        )
        if not right_type:
            raise _fail(
                f"{name} is {value!r}, which is not {'true or false' if kind is bool else kind.__name__}",
                key=name,
                config_path=config_path,
                fix=f"set `{name}` under {SECTION} in {config_path.name} as in the example",
            )
        values[name] = float(value) if kind is float else value.strip() if kind is str else value
    if not values["prompt"].strip():
        raise _fail(
            "prompt is empty",
            key="prompt",
            config_path=config_path,
            fix=f"give the question to put to the model as `prompt` under {SECTION} in {config_path.name}",
        )
    for name in ("max_tokens", "batch_size"):
        if values[name] < 1:
            raise _fail(
                f"{name} is {values[name]}, which is below 1",
                key=name,
                config_path=config_path,
                fix=f"set `{name}` under {SECTION} in {config_path.name} to 1 or more",
            )
    if values["max_receipts"] < 2:
        raise _fail(
            f"max_receipts is {values['max_receipts']}, which is below 2: no photograph could be split",
            key="max_receipts",
            config_path=config_path,
            fix=f"set `max_receipts` under {SECTION} in {config_path.name} to 2 or more",
        )
    for name, least, most in (
        ("margin_frac", 0.0, 0.5),
        ("min_box_frac", 0.0, 0.5),
        ("max_overlap_frac", 0.0, 1.0),
    ):
        if not least <= values[name] <= most:
            raise _fail(
                f"{name} is {values[name]}, which is outside {least} to {most}",
                key=name,
                config_path=config_path,
                fix=f"set `{name}` under {SECTION} in {config_path.name} to a share between {least} and {most}",
            )
    if not 0.0 <= values["straighten_min_deg"] < values["straighten_max_deg"] <= 90.0:
        raise _fail(
            f"straighten_min_deg is {values['straighten_min_deg']} and straighten_max_deg is "
            f"{values['straighten_max_deg']}: they must run from 0 up to at most 90, smallest first",
            key="straighten_min_deg",
            config_path=config_path,
            fix=f"set the two under {SECTION} in {config_path.name}, for example 1.0 and 40.0",
        )
    if not 0.0 < values["rectify_min_print_frac"] <= 1.0:
        raise _fail(
            f"rectify_min_print_frac is {values['rectify_min_print_frac']}, which is not above 0 and at most 1",
            key="rectify_min_print_frac",
            config_path=config_path,
            fix=f"set `rectify_min_print_frac` under {SECTION} in {config_path.name} to a share such as 0.98",
        )
    if not 0.0 < values["rectify_min_area_frac"] <= 1.0:
        raise _fail(
            f"rectify_min_area_frac is {values['rectify_min_area_frac']}, which is not above 0 and at most 1",
            key="rectify_min_area_frac",
            config_path=config_path,
            fix=f"set `rectify_min_area_frac` under {SECTION} in {config_path.name} to a share such as 0.30",
        )
    if not 1 <= values["rectify_min_edges"] <= 4:
        raise _fail(
            f"rectify_min_edges is {values['rectify_min_edges']}, which is not from 1 to 4",
            key="rectify_min_edges",
            config_path=config_path,
            fix=f"set `rectify_min_edges` under {SECTION} in {config_path.name} to a whole number from 1 to 4",
        )
    if values["rectify_pages"] and not values["straighten"]:
        raise _fail(
            "rectify_pages is true but straighten is false: rectification falls back on turning",
            key="rectify_pages",
            config_path=config_path,
            fix=f"set `straighten: true` under {SECTION} in {config_path.name}, or `rectify_pages: false`",
        )
    fill = section.get("straighten_fill_rgb")
    is_colour = (
        isinstance(fill, list)
        and len(fill) == 3
        and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 255 for v in fill)
    )
    if not is_colour or not isinstance(fill, list):
        raise _fail(
            f"straighten_fill_rgb is {fill!r}, which is not three whole numbers from 0 to 255",
            key="straighten_fill_rgb",
            config_path=config_path,
            fix=f"set `straighten_fill_rgb` under {SECTION} in {config_path.name} to a red, green, blue triple",
        )
    return PhotoSplitConfig(**values, straighten_fill_rgb=(fill[0], fill[1], fill[2]))


def parse_boxes(reply: str) -> list[Box] | None:
    """Read the boxes out of the model's reply.

    Args:
        reply: The model's reply, with or without a Markdown code fence round the JSON.

    Returns:
        The boxes, in the order given, or None when the reply is not a JSON list of objects each holding a
        `box_2d` of four numbers within 0 to 1000 that form a rectangle.
    """
    text = re.sub(r"```(?:json)?", "", reply).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    boxes: list[Box] = []
    for item in data:
        numbers = item.get("box_2d") if isinstance(item, dict) else None
        if not isinstance(numbers, list) or len(numbers) != 4:
            return None
        if not all(isinstance(n, int | float) and not isinstance(n, bool) for n in numbers):
            return None
        top, left, bottom, right = (float(n) for n in numbers)
        if not (0 <= left < right <= BOX_SCALE and 0 <= top < bottom <= BOX_SCALE):
            return None
        boxes.append(Box(left, top, right, bottom))
    return boxes


def _shared_area(first: Box, second: Box) -> float:
    width = min(first.right, second.right) - max(first.left, second.left)
    height = min(first.bottom, second.bottom) - max(first.top, second.top)
    return max(width, 0.0) * max(height, 0.0)


def decide(reply: str, config: PhotoSplitConfig) -> Decision:
    """Decide what to do with a photograph from the model's reply about it.

    Args:
        reply: The model's reply to the configured prompt.
        config: The photo split settings.

    Returns:
        Whether to split the photograph or pass it through, why, and the boxes.
    """
    parsed = parse_boxes(reply)
    if parsed is None:
        return Decision("pass_through", "the reply was not the list of boxes asked for", ())
    boxes = tuple(box for box in parsed if box.area >= config.min_box_frac * BOX_SCALE * BOX_SCALE)
    if len(boxes) == 0:
        return Decision("pass_through", "no receipt was found", boxes)
    if len(boxes) == 1:
        return Decision("pass_through", "one receipt", boxes)
    if len(boxes) > config.max_receipts:
        return Decision("pass_through", f"{len(boxes)} boxes, more than max_receipts", boxes)
    for i, first in enumerate(boxes):
        for second in boxes[i + 1 :]:
            if _shared_area(first, second) > config.max_overlap_frac * min(first.area, second.area):
                return Decision("pass_through", "two boxes overlap", boxes)
    return Decision("split", f"{len(boxes)} receipts", boxes)


def crop_rectangles(
    boxes: tuple[Box, ...], width: int, height: int, margin_frac: float
) -> list[tuple[int, int, int, int]]:
    """The pixel rectangle to cut for each box: the box and a margin, kept clear of its neighbours.

    Each side is pushed out by `margin_frac` of the box's own width or height, but never beyond the photograph
    and never past the halfway line to a neighbouring box on that side.

    Args:
        boxes: The boxes, on the 0 to 1000 scale.
        width: The photograph's width in pixels.
        height: The photograph's height in pixels.
        margin_frac: The margin, as a share of each box's own size.

    Returns:
        One (left, top, right, bottom) in pixels per box, in the same order.
    """
    rectangles = []
    for box in boxes:
        pad_x = margin_frac * (box.right - box.left)
        pad_y = margin_frac * (box.bottom - box.top)
        left, top = max(box.left - pad_x, 0.0), max(box.top - pad_y, 0.0)
        right, bottom = min(box.right + pad_x, BOX_SCALE), min(box.bottom + pad_y, BOX_SCALE)
        for other in boxes:
            if other is box:
                continue
            beside = min(box.bottom, other.bottom) > max(box.top, other.top)
            stacked = min(box.right, other.right) > max(box.left, other.left)
            if beside and other.left >= box.right:
                right = min(right, (box.right + other.left) / 2)
            if beside and other.right <= box.left:
                left = max(left, (box.left + other.right) / 2)
            if stacked and other.top >= box.bottom:
                bottom = min(bottom, (box.bottom + other.top) / 2)
            if stacked and other.bottom <= box.top:
                top = max(top, (box.top + other.bottom) / 2)
        rectangles.append(
            (
                int(left * width / BOX_SCALE),
                int(top * height / BOX_SCALE),
                int(round(right * width / BOX_SCALE)),
                int(round(bottom * height / BOX_SCALE)),
            )
        )
    return rectangles


def cut(image: Image.Image, boxes: tuple[Box, ...], margin_frac: float) -> list[Image.Image]:
    """Cut one image per box out of a photograph.

    Args:
        image: The photograph.
        boxes: The boxes to cut, on the 0 to 1000 scale.
        margin_frac: The margin, as a share of each box's own size.

    Returns:
        The cut images, in the order of the boxes.
    """
    return [
        image.crop(rectangle)
        for rectangle in crop_rectangles(boxes, image.width, image.height, margin_frac)
    ]


def row_direction(image: Image.Image, finder: FinderParams) -> float | None:
    """The direction of the rows of print in an image, in degrees from level.

    Args:
        image: The image: a cut receipt, or the part of a photograph inside a receipt's box.
        finder: The settings for finding print and its direction (`pipeline.photo_split.finder`).

    Returns:
        The direction, positive when the rows fall away to the right, or None when the print gives none.
    """
    rgb = np.asarray(image.convert("RGB"))
    scale = min(1.0, finder.work_long_edge / max(rgb.shape[:2]))
    if scale < 1:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    marks, text_height = print_marks(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), finder)
    return text_direction(marks, text_height, finder)


def level_turn(direction: float | None, config: PhotoSplitConfig) -> float:
    """The turn to apply for a row direction: the direction itself, or 0 when it is too small or not believed."""
    if direction is None or not config.straighten_min_deg <= abs(direction) <= config.straighten_max_deg:
        return 0.0
    return direction


def turn(image: Image.Image, degrees: float, config: PhotoSplitConfig) -> Image.Image:
    """Turn an image so that rows running at `degrees` from level become level.

    The image grows to hold its turned corners, which are filled with `straighten_fill_rgb`.

    Args:
        image: The image.
        degrees: The direction of its rows of print, as `row_direction` gives it. 0 returns the image unchanged.
        config: The photo split settings.

    Returns:
        The turned image.
    """
    if degrees == 0.0:
        return image
    return image.convert("RGB").rotate(
        degrees, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=config.straighten_fill_rgb
    )


def rectify(image: Image.Image, finder: FinderParams, config: PhotoSplitConfig) -> Image.Image | None:
    """Rectify a photograph of one document from its four corners, when they can be trusted.

    The corners are found and the homography reversed, so the page comes out upright with its rows parallel:
    its tilt and its foreshortening are both removed.

    Args:
        image: The photograph.
        finder: The sheet finder's settings (`pipeline.photo_split.finder`).
        config: The photo split settings.

    Returns:
        The rectified page, or None when the corners cannot be trusted: no sheet or more than one found, the
        sheet covering less than `rectify_min_area_frac` of the photograph, fewer than `rectify_min_edges`
        of its edges confirmed, or less than `rectify_min_print_frac` of the photograph's print lying
        inside it. The last is what tells a sheet from a ruled table on a page that is not a photograph:
        a sheet holds all the print, a table leaves the page's heading outside it.
    """
    rgb = np.asarray(image.convert("RGB"))
    sheets = find_sheets(rgb, finder)
    if len(sheets) != 1:
        return None
    sheet = sheets[0]
    covered = float(cv2.contourArea(sheet.quad)) / float(rgb.shape[0] * rgb.shape[1])
    if covered < config.rectify_min_area_frac or sheet.edges_confirmed < config.rectify_min_edges:
        return None
    scale = min(1.0, finder.work_long_edge / max(rgb.shape[:2]))
    small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    marks, _ = print_marks(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), finder)
    inside = np.zeros(marks.shape, np.uint8)
    cv2.fillConvexPoly(inside, np.round(sheet.quad * scale).astype(np.int32), 1)
    printed = int(np.count_nonzero(marks))
    if printed == 0 or np.count_nonzero(marks[inside > 0]) < config.rectify_min_print_frac * printed:
        return None
    return Image.fromarray(warp_upright(rgb, sheet.quad, config.straighten_fill_rgb))
