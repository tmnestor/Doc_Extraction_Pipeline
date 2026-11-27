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

Every operating choice is a field of `PhotoSplitConfig`, with the value it was measured at.
"""

import json
import re
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from common.sheet_finder import (
    FinderParams,
    find_sheet_by_paper,
    find_sheets,
    print_marks,
    row_drift,
    text_direction,
    warp_upright,
)

# The scale the model's box coordinates are on, whatever the photograph's size.
BOX_SCALE = 1000.0


@dataclass(frozen=True)
class PhotoSplitConfig:
    """The photo split's rules, each with the value it was measured at.

    Until 26 November 2026 these were keys of the settings file, all required. No run ever changed one:
    they are tuned values, changed with the code and not by a job. Whether the photo split is on at all
    is the job's choice, and is not here; the question put to the model is in `prompts/photo_split.yaml`.

    Attributes:
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
        straighten_min_drift_heights: An image is turned or rectified only when its rows of print drift at
            least this many text heights off their own line across the width of the print. Below about one,
            a reader following a row stays on it, and straightening changes nothing. 0 acts on every image.
        straighten_fill_rgb: Colour of the corners a turned image gains.
        rectify_pages: Whether a photograph of one document is rectified from its corners when they can be
            trusted. Needs `straighten`, whose turning is the fallback.
        rectify_pieces: Whether each image cut from a photograph of several receipts is rectified in the same
            way, under the same tests of trust. Needs `straighten`, whose turning is the fallback.
        rectify_min_area_frac: The sheet found must cover at least this share of the photograph.
        rectify_min_edges: At least this many of the sheet's four edges must be confirmed by a sharp step.
        rectify_min_print_frac: At least this share of the photograph's print must lie inside the sheet.
        rectify_paper_model: Whether, when the desk method gives no sheet that can be trusted, the paper
            method is tried (`common.sheet_finder.find_sheet_by_paper`). It finds a sheet on a textured,
            cluttered or pale surface, where the desk method cannot.
        rectify_paper_min_fit: The paper method's sheet must agree with its quadrilateral at least this well.
        upright_max_tokens: Longest reply allowed to the question of whether the print is the right way up.
        upright_strip_heights: To tell whether a photograph lies on its side, its print is summed along
            strips this many text heights wide. Narrow, so that a tilted row stays within its own band.
        upright_smooth_heights: Each strip's profile is compared with itself smoothed over this many text
            heights.
    """

    max_tokens: int = 400
    batch_size: int = 16
    margin_frac: float = 0.1
    min_box_frac: float = 0.01
    max_overlap_frac: float = 0.1
    max_receipts: int = 12
    straighten: bool = True
    straighten_min_deg: float = 1.0
    straighten_max_deg: float = 15.0
    straighten_min_drift_heights: float = 2.0
    straighten_fill_rgb: tuple[int, int, int] = (255, 255, 255)
    rectify_pages: bool = True
    rectify_pieces: bool = True
    rectify_min_area_frac: float = 0.3
    rectify_min_edges: int = 4
    rectify_min_print_frac: float = 0.98
    rectify_paper_model: bool = True
    rectify_paper_min_fit: float = 0.97
    upright_max_tokens: int = 10
    upright_strip_heights: float = 12.0
    upright_smooth_heights: float = 4.0


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


# What the split stage goes by. A test puts another record in their place.
CONFIG = PhotoSplitConfig()
FINDER = FinderParams()


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
                round(right * width / BOX_SCALE),
                round(bottom * height / BOX_SCALE),
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


def box_within(box: Box, rectangle: tuple[int, int, int, int], width: int, height: int) -> Box:
    """A box of the photograph, restated on the 0 to 1000 scale of an image cut out of it.

    Args:
        box: The box, on the photograph's 0 to 1000 scale.
        rectangle: The (left, top, right, bottom) in pixels that was cut, as `crop_rectangles` gives it.
        width: The photograph's width in pixels.
        height: The photograph's height in pixels.

    Returns:
        The same box on the cut image's own 0 to 1000 scale, kept inside the image.
    """
    left, top, right, bottom = rectangle

    def along(value: float, size: int, start: int, end: int) -> float:
        return min(max((value / BOX_SCALE * size - start) / (end - start), 0.0), 1.0) * BOX_SCALE

    return Box(
        left=along(box.left, width, left, right),
        top=along(box.top, height, top, bottom),
        right=along(box.right, width, left, right),
        bottom=along(box.bottom, height, top, bottom),
    )


def _print_of(image: Image.Image, finder: FinderParams) -> tuple[np.ndarray, float]:
    """An image's print marks and text height, at the finder's working size."""
    rgb = np.asarray(image.convert("RGB"))
    scale = min(1.0, finder.work_long_edge / max(rgb.shape[:2]))
    if scale < 1:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return print_marks(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), finder)


def _alternation(marks: np.ndarray, text_height: float, config: PhotoSplitConfig) -> float:
    """How sharply print and gap alternate down an image of print marks, for each unit of print."""
    ink = (marks > 0).astype(np.float32)
    strip = max(round(config.upright_strip_heights * text_height), 8)
    window = max(round(config.upright_smooth_heights * text_height) | 1, 3)
    if window > ink.shape[0]:
        return 0.0
    box = np.ones(window, dtype=np.float32) / window
    energy = inked = 0.0
    for left in range(0, ink.shape[1], strip):
        part = ink[:, left : left + strip]
        if part.shape[1] < strip // 2:
            continue
        profile = part.mean(axis=1)
        energy += float(((profile - np.convolve(profile, box, mode="same")) ** 2).sum())
        inked += float(profile.sum())
    return energy / inked if inked else 0.0


def lies_on_its_side(image: Image.Image, finder: FinderParams, config: PhotoSplitConfig) -> bool:
    """Whether the rows of print in an image run up it and not across it.

    Print is rows of letters with gaps between the rows. Down a page that is the right way up, or upside
    down, print and gap alternate sharply from row to row; across it they do not, since the letters of
    different rows do not line up. On a page lying on its side it is the other way about. Measured on
    27 November 2026 on 2,151 images of eight sets, each as it was and turned a quarter: right both ways
    on 2,147. It cannot tell the right way up from upside down: that is asked of the model
    (`said_upright`).

    Args:
        image: The photograph.
        finder: The settings for finding print (`FINDER`).
        config: The photo split's rules.

    Returns:
        True when the rows run up the image. False when they run across it, or there is no print to say.
    """
    marks, text_height = _print_of(image, finder)
    if not text_height:
        return False
    return _alternation(marks.T, text_height, config) > _alternation(marks, text_height, config)


def said_upright(reply: str) -> bool | None:
    """The model's answer to whether the print is the right way up, or None when it gave neither YES nor NO."""
    found = re.search(r"\b(YES|NO)\b", reply, re.IGNORECASE)
    return None if found is None else found.group(1).upper() == "YES"


def turned_upright(image: Image.Image, quarter_turns_deg: int) -> Image.Image:
    """An image turned anticlockwise by 0, 90, 180 or 270 degrees, with no pixel changed."""
    return image.rotate(quarter_turns_deg, expand=True) if quarter_turns_deg else image


def row_direction(image: Image.Image, finder: FinderParams) -> float | None:
    """The direction of the rows of print in an image, in degrees from level.

    Args:
        image: The image: a cut receipt, or the part of a photograph inside a receipt's box.
        finder: The settings for finding print and its direction (`FINDER`).

    Returns:
        The direction, positive when the rows fall away to the right, or None when the print gives none.
    """
    marks, text_height = _print_of(image, finder)
    return text_direction(marks, text_height, finder)


def row_measures(image: Image.Image, finder: FinderParams) -> tuple[float | None, float | None]:
    """The direction of an image's rows of print, and how far they drift across the print's width.

    Args:
        image: The image: a cut receipt, or the part of a photograph inside a receipt's box.
        finder: The settings for finding print and its direction (`FINDER`).

    Returns:
        The direction in degrees from level, and the drift in text heights (see
        `common.sheet_finder.row_drift`). Each is None when the print gives none.
    """
    marks, text_height = _print_of(image, finder)
    return text_direction(marks, text_height, finder), row_drift(marks, text_height, finder)


def worth_straightening(drift: float | None, config: PhotoSplitConfig) -> bool:
    """Whether an image's rows drift far enough for turning or rectifying it to change how it reads.

    An image whose drift could not be measured is treated as worth it: nothing is known against it.
    """
    return drift is None or drift >= config.straighten_min_drift_heights


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


def _desk_quad(rgb: np.ndarray, finder: FinderParams, config: PhotoSplitConfig) -> np.ndarray | None:
    """The corners of the one sheet the desk method finds, when they pass every test of trust."""
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
    return sheet.quad


def _paper_quad(
    rgb: np.ndarray, finder: FinderParams, config: PhotoSplitConfig, box: Box | None
) -> np.ndarray | None:
    """The corners of the sheet the paper method finds, when they pass every test of trust."""
    shares = (
        None if box is None else tuple(v / BOX_SCALE for v in (box.left, box.top, box.right, box.bottom))
    )
    sheet = find_sheet_by_paper(rgb, finder, shares)  # type: ignore[arg-type]
    if sheet is None:
        return None
    covered = float(cv2.contourArea(sheet.quad)) / float(rgb.shape[0] * rgb.shape[1])
    trusted = (
        covered >= config.rectify_min_area_frac
        and sheet.edges_confirmed >= config.rectify_min_edges
        and sheet.fit >= config.rectify_paper_min_fit
        and sheet.print_inside >= config.rectify_min_print_frac
    )
    return sheet.quad if trusted else None


def rectify(
    image: Image.Image, finder: FinderParams, config: PhotoSplitConfig, box: Box | None = None
) -> Image.Image | None:
    """Rectify a photograph of one document from its four corners, when they can be trusted.

    The corners are found and the homography reversed, so the page comes out upright with its rows parallel:
    its tilt and its foreshortening are both removed.

    The corners are looked for twice. The desk method comes first: it is fast and exact, but needs a desk that
    is smooth and of one colour. When it gives no sheet that can be trusted and `rectify_paper_model` is true,
    the paper method is tried, which finds a sheet on any surface.

    Args:
        image: The photograph.
        finder: The sheet finder's settings (`FINDER`).
        config: The photo split settings.
        box: The model's box for the document, when it gave exactly one. The paper method uses it to tell the
            document's print from anything else in the photograph.

    Returns:
        The rectified page, or None when neither method's corners can be trusted. Either way the sheet must
        cover at least `rectify_min_area_frac` of the photograph, have at least `rectify_min_edges` of its
        edges confirmed, and hold at least `rectify_min_print_frac` of the photograph's print: that last is
        what tells a sheet from a ruled table on a page that is not a photograph (a sheet holds all the print,
        a table leaves the page's heading outside it). The desk method must also find exactly one sheet, and the
        paper method's sheet must agree with its quadrilateral by at least `rectify_paper_min_fit`.
    """
    rgb = np.asarray(image.convert("RGB"))
    quad = _desk_quad(rgb, finder, config)
    if quad is None and config.rectify_paper_model:
        quad = _paper_quad(rgb, finder, config, box)
    if quad is None:
        return None
    return Image.fromarray(warp_upright(rgb, quad, config.straighten_fill_rgb))
