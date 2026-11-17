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

The reply's form is the one the model gives for detection prompts: a JSON list of objects, each with
`"box_2d": [ymin, xmin, ymax, xmax]` on a scale of 0 to 1000 across the photograph's height and width.

Every operating choice is in `pipeline.photo_split` of the run config. Every key is required.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

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
    max_receipts: 12"""

_KEYS: dict[str, type] = {
    "enabled": bool,
    "prompt": str,
    "max_tokens": int,
    "batch_size": int,
    "margin_frac": float,
    "min_box_frac": float,
    "max_overlap_frac": float,
    "max_receipts": int,
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
    """

    enabled: bool
    prompt: str
    max_tokens: int
    batch_size: int
    margin_frac: float
    min_box_frac: float
    max_overlap_frac: float
    max_receipts: int


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
    return PhotoSplitConfig(**values)


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
