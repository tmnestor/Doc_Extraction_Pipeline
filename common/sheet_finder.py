"""Find every sheet of paper in a photograph, and straighten each one.

A phone photograph of documents (receipts, or now and then a bank statement) is not one upright page: it holds
one sheet or several, each at its own angle, on a desk, seen off-axis. This module finds each sheet's four
corners so it can be warped upright and read as a page of its own.

The method, in the order it runs:

1. **Desk, not paper.** The desk is modelled from the frame's outer ring. Desk is then whatever can be reached
   from the frame's border through smooth, desk-coloured pixels. A shadow on the desk is desk-coloured (the desk
   darkened, same hue), but only close to lit desk: further in, that colour is shaded paper. Each sheet is what
   is left, cut off from the border by its own sharp edge.
2. **Text direction, by linear regression.** The letters of each row of print are joined sideways into a strip,
   `cv2.fitLine` fits a least-squares line through each strip, and the rows' median direction is taken. Joining
   sideways assumes the sheet is roughly upright; it cannot take the columns of a monospaced font for rows.
   A Hough vote by angle and `cv2.HoughLinesP` were both measured against it (see `text_direction`).
3. **Edges, by RANSAC.** Four straight lines are fitted to the part of a sheet's outline that lies on its convex
   hull. They are accepted only if two run along the rows and two across them. Each is then re-fitted on the
   full-size photograph from the first sharp step met walking in from outside the sheet, which is the paper's
   edge and not a shadow's (a shadow's edge is a slow ramp).
4. **Fallback.** When the four lines are not found, the sheet's own bounding rectangle takes their place and is
   refined the same way, edge by edge.

A sheet lies on the desk, not on another sheet, so a shape found inside a larger one is dropped.

That method models the DESK, and needs it smooth and of one colour. A second method, `find_sheet_by_paper`,
models the PAPER instead, for one document on any surface:

1. **Seed from the print.** Print sits on paper, so the area inside a block of print is paper for certain.
   The vision model's box for the document, when there is one, says which print is the document's.
2. **Paper, by `cv2.grabCut`.** Seeded with that area, it learns the colours of the paper and of the
   surroundings from the photograph itself, so texture and clutter do not matter.
3. **Along its own edges.** Print and box both stop where the print stops, so blank paper beyond it is at
   first taken for surroundings. A second pass offers the two strips between the sheet's opposite edge lines
   as probable paper, and the longer sheet is kept only if it is still a clean four-sided shape.
4. **Edges**, refined on the full-size photograph exactly as in the first method.

The desk method is measured on synthetic photographs only, whose desk is one plain colour. It does not hold up
on real photographs: on a textured surface (wood grain, felt), on a frame that also shows other surfaces and
objects, or on a pale surface little darker than the paper, no sheet is isolated. Tried on four real
photographs of receipts, it isolated a receipt in none; the paper method found all five receipts in them
(judged by eye: there are no true corners for real photographs).

Further limits: every edge of a sheet is expected inside the frame; sheets that touch or overlap are not separated; a
shadow whose edge is as sharp as the paper's (hard, direct light) is taken for the paper's edge; and
print is needed to say which way up a sheet is (without it the shorter edges are taken as top and bottom). Rows
read the same upside down, so of the two ways the rows allow, the one nearer upright in the photograph is taken.

Every operating choice is a field of `FinderParams`, read from YAML by the caller; none has a default here.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# Numerical details of the method, not operating choices: the choices are FinderParams.
_PLANE_FIT_ROUNDS = 6
_RANSAC_ROUNDS = 300
_EDGE_SAMPLES = 300
_MIN_EDGE_HITS = 30
_MIN_OUTLINE_POINTS = 20
_HULL_TOLERANCE_PX = 2.0
# Tolerances, as shares of the hull's perimeter, tried in turn to reduce a paper region's hull to four corners.
_HULL_QUAD_SHARES = (0.02, 0.03, 0.05, 0.08)
_PAPER_OPEN_PX = 5


@dataclass(frozen=True)
class FinderParams:
    """Every operating choice of the finder. All come from YAML; none has a default here.

    Attributes:
        work_long_edge: Long edge, in pixels, of the copy the sheets are separated on. A larger photograph is
            shrunk to it; corners are always refined on the full-size photograph.
        desk_ring_frac: Width of the frame's outer ring, as a share of its short edge, that the desk is read from.
        mask_min_distance: Least colour distance from the desk (grey levels) for a pixel to count as not desk.
        mask_noise_multiple: The same threshold as a multiple of the desk's own noise; the larger of the two is used.
        shadow_min_factor: The darkest a shadow makes the desk, as a share of its lit brightness.
        shadow_reach_heights: How far from lit desk a shadow may reach, in text heights.
        shadow_reach_frac: The same as a share of the working copy's long edge, used when no print is found.
        blur_sigma_px: Blur applied before sharp steps are looked for.
        step_span_px: Distance over which a change counts as a step.
        step_min: Least change (grey levels) over that distance for a step to count as sharp.
        step_min_area_px: A patch of sharp-step pixels smaller than this is noise, not an edge.
        edge_gap_close_px: Gaps in a line of sharp steps up to about twice this are bridged.
        edge_seal_px: Sharp steps are thickened by this much, so the desk cannot be reached through them.
        min_region_frac: Least share of the frame a sheet must cover.
        outline_tolerance_px: How far an outline point may lie from an edge line and still support it.
        min_outline_edge_frac: Least outline points on an edge line, as a share of the sheet's shorter side.
        text_agree_deg: How far an edge may be from the direction of the rows, or across them, and still be accepted.
        refine_outside_first_frac: First pass: how far outside the coarse edge to start, as a share of the diagonal.
        refine_inside_first_frac: First pass: how far inside the coarse edge to look.
        refine_outside_final_frac: Final pass: the same two reaches, shorter.
        refine_inside_final_frac: Final pass, inside.
        edge_tolerance_px: How far an edge point may lie from the refined line and still support it.
        min_edge_support: Least share of an edge that must show a sharp step for the refined line to be used.
        mark_block_px: Size of the neighbourhood a pixel is compared with to count as print (odd).
        mark_darker_by: How much darker than its neighbourhood a pixel must be to count as print.
        mark_max_heights: A mark longer than this many text heights is not print.
        text_join_heights: Width over which the letters of a row are joined sideways, in text heights.
        text_min_row_heights: Shortest strip of joined letters that counts as a row, in text heights.
        text_max_row_thickness_heights: Thickest strip that still counts as one row, in text heights.
        text_min_rows: Least number of rows for print to give a direction.
        ransac_seed: Seed for the line fits, so a photograph always gives the same corners.
        paper_work_long_edge: Long edge, in pixels, of the copy the paper method separates the sheet on.
        paper_rounds: Rounds of `cv2.grabCut` in each of its two passes.
        paper_join_heights: Letters and rows closer than this many text heights join into one block of print.
        paper_border_frac: Width of the frame's outer ring, as a share of its short edge, that is surroundings
            for certain.
        paper_min_border_difference: Least colour difference (CIE Lab) between that ring and the paper under
            the print for there to be surroundings at all. A page that fills the frame has none.
        paper_region_grow_frac: With a box from the model, the method looks only at the box and the print it
            claims, grown on every side by this share of their extent. Wide enough to hold blank paper beyond the
            print; what lies further off is neither segmented nor counted as print outside the sheet.
        paper_block_in_box_min: A block of print belongs to the box's document when at least this share of
            it lies inside the box.
        paper_block_min_share: ... and it is at least this share of the largest such block.
        paper_refine_outside_frac: How far outside the paper method's coarse edge its refinement looks, as a
            share of the diagonal.
        paper_refine_inside_frac: How far inside it starts. Short, so the walk starts in the page's margin and
            not in its print.
        paper_extension_min_fit: The second pass's longer sheet is kept when it and its quadrilateral agree
            at least this well (shared area over the area either covers).
    """

    work_long_edge: int
    desk_ring_frac: float
    mask_min_distance: float
    mask_noise_multiple: float
    shadow_min_factor: float
    shadow_reach_heights: float
    shadow_reach_frac: float
    blur_sigma_px: float
    step_span_px: int
    step_min: float
    step_min_area_px: int
    edge_gap_close_px: int
    edge_seal_px: int
    min_region_frac: float
    outline_tolerance_px: float
    min_outline_edge_frac: float
    text_agree_deg: float
    refine_outside_first_frac: float
    refine_inside_first_frac: float
    refine_outside_final_frac: float
    refine_inside_final_frac: float
    edge_tolerance_px: float
    min_edge_support: float
    mark_block_px: int
    mark_darker_by: float
    mark_max_heights: float
    text_join_heights: float
    text_min_row_heights: float
    text_max_row_thickness_heights: float
    text_min_rows: int
    ransac_seed: int
    paper_work_long_edge: int
    paper_rounds: int
    paper_join_heights: float
    paper_border_frac: float
    paper_min_border_difference: float
    paper_region_grow_frac: float
    paper_block_in_box_min: float
    paper_block_min_share: float
    paper_extension_min_fit: float
    paper_refine_outside_frac: float
    paper_refine_inside_frac: float


@dataclass(frozen=True)
class FoundSheet:
    """One sheet found in a photograph.

    Attributes:
        quad: Its corners in photograph pixels (float32, 4x2): top-left, top-right, bottom-right, bottom-left,
            where the top edge runs left to right along the rows of print.
        text_deg: Direction of its rows of print, in degrees from the x axis, or None when print gave none.
        outline: How the corners were first placed: "edges" (four fitted lines) or "box" (bounding rectangle).
        edges_confirmed: How many of the four edges were confirmed by a sharp step and refined.
    """

    quad: np.ndarray
    text_deg: float | None
    outline: str
    edges_confirmed: int


@dataclass(frozen=True)
class PaperSheet:
    """The one sheet found by the paper method.

    Attributes:
        quad: Its corners in photograph pixels (float32, 4x2): top-left, top-right, bottom-right, bottom-left.
        text_deg: Direction of its rows of print, in degrees from the x axis, or None when print gave none.
        edges_confirmed: How many of the four edges were confirmed by a sharp step and refined.
        fit: How well the paper region and its quadrilateral agree before refinement: the area they share
            over the area either covers. A sheet with desk let in, or a part missing, scores low.
        print_inside: The share of everything in the photograph that looks like print that lies inside the
            quadrilateral. Print outside it means the quadrilateral is not the whole sheet: a shaded table on
            a clean page leaves the page's heading outside. A strongly textured surface also counts against
            it, since its texture can look like print; such a sheet is then not trusted, which is the safe way
            to be wrong.
    """

    quad: np.ndarray
    text_deg: float | None
    edges_confirmed: int
    fit: float
    print_inside: float


def _apart(first_deg: float, second_deg: float) -> float:
    """The angle between two undirected lines, in degrees (0 to 90)."""
    return abs(((first_deg - second_deg) + 90.0) % 180.0 - 90.0)


def _fit_line(points: np.ndarray) -> np.ndarray:
    """Total-least-squares line through points, as (nx, ny, c) with nx*x + ny*y + c = 0 and a unit normal."""
    centre = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centre)
    normal = vt[1]
    return np.array([normal[0], normal[1], -float(normal @ centre)])


def _line_deg(line: np.ndarray) -> float:
    """Direction of a line (nx, ny, c), in degrees in [-90, 90)."""
    return float((np.degrees(np.arctan2(line[0], -line[1])) + 90.0) % 180.0 - 90.0)


def _intersect(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    """Where two lines meet, or None when they are parallel."""
    normals = np.array([first[:2], second[:2]])
    if abs(np.linalg.det(normals)) < 1e-9:
        return None
    return np.linalg.solve(normals, -np.array([first[2], second[2]]))


def _ransac_inliers(points: np.ndarray, tolerance: float, rng: np.random.Generator) -> np.ndarray:
    """Which points lie on the best-supported straight line through them."""
    best = np.zeros(len(points), bool)
    for _ in range(_RANSAC_ROUNDS):
        a, b = points[rng.choice(len(points), 2, replace=False)]
        along = b - a
        length = float(np.hypot(*along))
        if length < 1e-6:
            continue
        normal = np.array([-along[1], along[0]]) / length
        inliers = np.abs((points - a) @ normal) < tolerance
        if inliers.sum() > best.sum():
            best = inliers
    return best


def desk_distances(rgb: np.ndarray, params: FinderParams) -> tuple[np.ndarray, np.ndarray, float]:
    """Each pixel's colour distance from the desk, lit and in shadow, and the desk's noise level.

    The desk is modelled as colour = a + b*x + c*y per channel (a flat surface under a lighting gradient), fitted
    to the frame's outer ring and trimmed of whatever in the ring is not desk.

    Args:
        rgb: The photograph, uint8 (H, W, 3).
        params: The finder's parameters.

    Returns:
        The distance from the lit desk (H, W); the distance from the nearest darkened desk colour, which is small
        on a shadow; and the robust standard deviation of the desk about its model.
    """
    h, w = rgb.shape[:2]
    ring = max(int(min(h, w) * params.desk_ring_frac), 4)
    in_ring = np.zeros((h, w), bool)
    in_ring[:ring] = in_ring[-ring:] = True
    in_ring[:, :ring] = in_ring[:, -ring:] = True
    ys, xs = np.nonzero(in_ring)
    pixels = rgb[ys, xs].astype(np.float64)
    design = np.column_stack([np.ones(len(xs)), xs, ys])
    keep = np.ones(len(xs), bool)
    coefficients: np.ndarray = np.zeros((3, 3))
    sigma = 1.0
    for _ in range(_PLANE_FIT_ROUNDS):
        coefficients = np.linalg.lstsq(design[keep], pixels[keep], rcond=None)[0]
        residual = np.abs(pixels - design @ coefficients).max(axis=1)
        sigma = max(float(1.4826 * np.median(residual[keep])), 1.0)
        # Trim against the median of ALL ring pixels as well, so a sheet covering part of the ring is dropped.
        keep = residual < max(3.0 * 1.4826 * float(np.median(residual)), 3.0 * sigma)
        if keep.sum() < 100:
            break
    yy, xx = np.mgrid[0:h, 0:w]
    image = rgb.astype(np.float64)

    def model(fit: np.ndarray) -> np.ndarray:
        return np.stack([fit[0, c] + fit[1, c] * xx + fit[2, c] * yy for c in range(3)], -1)

    lit = np.linalg.norm(image - model(coefficients), axis=-1)
    # Fitted again on everything that looks like lit desk, not the ring alone, for the shadow comparison.
    is_desk = lit < 3 * sigma
    if is_desk.sum() >= 100:
        design = np.column_stack([np.ones(is_desk.sum()), xx[is_desk], yy[is_desk]])
        coefficients = np.linalg.lstsq(design, image[is_desk], rcond=None)[0]
    desk = model(coefficients)
    factor = np.clip(
        (image * desk).sum(-1) / np.maximum((desk * desk).sum(-1), 1e-9), params.shadow_min_factor, 1.0
    )
    shaded = np.linalg.norm(image - factor[..., None] * desk, axis=-1)
    return lit, shaded, sigma


def print_marks(grey: np.ndarray, params: FinderParams) -> tuple[np.ndarray, float]:
    """Small dark marks that could be print, and the typical height of one.

    Args:
        grey: The working copy in grey, uint8 (H, W).
        params: The finder's parameters.

    Returns:
        A mask of the marks (255 where print), and the median height of a letter-sized mark in pixels, or 0.0
        when too few are found to say.
    """
    ink = cv2.adaptiveThreshold(
        grey,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        params.mark_block_px,
        params.mark_darker_by,
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    widths = stats[1:, cv2.CC_STAT_WIDTH]
    areas = stats[1:, cv2.CC_STAT_AREA]
    letter = (heights >= 4) & (heights <= 60) & (widths >= 2) & (areas >= 6)
    if letter.sum() < 20:
        return np.zeros_like(ink), 0.0
    text_height = float(np.median(heights[letter]))
    longest = np.maximum(heights, widths)
    keep = (
        (longest <= params.mark_max_heights * text_height) & (longest >= 0.3 * text_height) & (areas >= 4)
    )
    lut = np.zeros(count, np.uint8)
    lut[1:][keep] = 255
    return lut[labels], text_height


def text_direction(marks: np.ndarray, text_height: float, params: FinderParams) -> float | None:
    """Direction of the rows of print among some marks, by a least-squares line through each row.

    The letters of each row are joined sideways into one strip, `cv2.fitLine` fits a least-squares line through
    each strip, and the direction is the median of the rows' directions, weighted by their length. Joining
    sideways assumes the sheet is roughly upright in the photograph, which receipts are; it cannot mistake the
    columns of a monospaced font for rows, because letters are never joined up and down.

    Chosen by measurement on 17 November 2026, on 165 synthetic receipts (right = within 2 degrees) and three
    real photographs:

    - this method: 153 right, worst 3.4 degrees, none a quarter turn out; right on all three photographs;
    - a hand-written Hough vote by angle: 159 right, worst 5.1 degrees, none a quarter turn out; right on all
      three, but custom code where this is a standard routine;
    - `cv2.HoughLinesP` at its best setting: 156 right, 2 a quarter turn out; a quarter turn out on two of the
      photographs. It follows rows or columns according to one gap setting that must lie between the letter
      spacing and the row spacing, and faint thermal print leaves no such setting.

    Args:
        marks: A mask of print marks, non-zero where print.
        text_height: The typical height of a letter, in pixels.
        params: The finder's parameters.

    Returns:
        The direction in degrees from the x axis, in [-90, 90), or None when too few rows are found.
    """
    if text_height <= 0:
        return None
    join = max(int(round(params.text_join_heights * text_height)), 2)
    strips = cv2.morphologyEx(marks, cv2.MORPH_CLOSE, np.ones((1, join), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(strips, connectivity=8)
    angles: list[float] = []
    lengths: list[float] = []
    for label in range(1, count):
        x, y, w, h, _ = stats[label]
        if w < params.text_min_row_heights * text_height:
            continue
        ys, xs = np.nonzero(labels[y : y + h, x : x + w] == label)
        points = np.column_stack([xs, ys]).astype(np.float32)
        vx, vy, _, _ = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        centred = points - points.mean(axis=0)
        # A strip is one row only if it is thin: rows that have run together are left out.
        thickness = 2.0 * float(np.percentile(np.abs(centred @ np.array([-vy, vx])), 95))
        if thickness > params.text_max_row_thickness_heights * text_height:
            continue
        angles.append(float((np.degrees(np.arctan2(vy, vx)) + 90.0) % 180.0 - 90.0))
        lengths.append(float(np.ptp(centred @ np.array([vx, vy]))))
    if len(angles) < params.text_min_rows:
        return None
    order = np.argsort(angles)
    cumulative = np.cumsum(np.array(lengths)[order])
    return float(np.array(angles)[order][np.searchsorted(cumulative, cumulative[-1] / 2)])


def _sharp_steps(blurred: np.ndarray, params: FinderParams) -> np.ndarray:
    """Pixels where the image changes sharply over the step span, across or down, with noise and gaps tidied."""
    span = params.step_span_px
    sharp = np.zeros(blurred.shape[:2], bool)
    across = np.abs(blurred[:, span:] - blurred[:, :-span]).max(axis=2) > params.step_min
    down = np.abs(blurred[span:, :] - blurred[:-span, :]).max(axis=2) > params.step_min
    half = span // 2
    sharp[:, half : half + across.shape[1]] |= across
    sharp[half : half + down.shape[0], :] |= down
    count, labels, stats, _ = cv2.connectedComponentsWithStats(sharp.astype(np.uint8), connectivity=8)
    lut = np.zeros(count, np.uint8)
    lut[1:] = stats[1:, cv2.CC_STAT_AREA] >= params.step_min_area_px
    steps = lut[labels]
    close = 2 * params.edge_gap_close_px + 1
    steps = cv2.morphologyEx(steps, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    seal = 2 * params.edge_seal_px + 1
    return cv2.dilate(steps, np.ones((seal, seal), np.uint8))


def sheet_regions(small: np.ndarray, text_height: float, params: FinderParams) -> tuple[np.ndarray, int]:
    """Label each sheet of paper on the working copy.

    Args:
        small: The working copy, uint8 (H, W, 3).
        text_height: The typical height of a letter on it, or 0.0 when no print was found.
        params: The finder's parameters.

    Returns:
        A label image (0 for desk, 1..n for the sheets) and n.
    """
    h, w = small.shape[:2]
    blurred = cv2.GaussianBlur(small, (0, 0), params.blur_sigma_px).astype(np.float32)
    lit, shaded, sigma = desk_distances(small, params)
    threshold = max(params.mask_noise_multiple * sigma, params.mask_min_distance)
    reach = (
        params.shadow_reach_heights * text_height if text_height else params.shadow_reach_frac * max(h, w)
    )
    smooth = _sharp_steps(blurred, params) == 0
    lit_desk = ((lit < threshold) & smooth).astype(np.uint8)
    shadow_or_desk = ((shaded < threshold) & smooth).astype(np.uint8)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    def joined(seed: np.ndarray, through: np.ndarray) -> np.ndarray:
        """Every pixel of `through` connected to the seed (or, with no seed, to the frame's border)."""
        _, labels = cv2.connectedComponents(through, connectivity=4)
        if seed.any():
            touching = np.unique(labels[seed > 0])
        else:
            touching = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        return np.isin(labels, touching[touching > 0]).astype(np.uint8)

    # Lit desk reached from the border; then shadow, but only as far as a shadow reaches, measured along the
    # desk and never across a sharp step; then any lit desk beyond that shadow, and its shadow in turn.
    desk = joined(np.zeros((h, w), np.uint8), lit_desk)
    # The frame's border is desk wherever it is desk-coloured, lit or not: a sheet lying close to the border
    # leaves a strip of desk there that its own shadow covers and that no lit desk reaches.
    on_border = np.zeros((h, w), np.uint8)
    on_border[0], on_border[-1], on_border[:, 0], on_border[:, -1] = 1, 1, 1, 1
    desk |= on_border & shadow_or_desk
    for _ in range(2):
        for _ in range(max(int(round(reach)), 1)):
            desk = cv2.dilate(desk, cross) & shadow_or_desk | desk
        desk = joined(desk, lit_desk | desk)
    not_desk = (desk == 0).astype(np.uint8)
    paper = cv2.morphologyEx(not_desk, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(paper, connectivity=8)
    owner = np.zeros((h, w), np.int32)
    sheets = 0
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= params.min_region_frac * h * w:
            sheets += 1
            owner[labels == label] = sheets
    return owner, sheets


def _hull_lines(
    contour: np.ndarray, shorter_side: float, params: FinderParams, rng: np.random.Generator
) -> list[np.ndarray] | None:
    """Four edge lines of a sheet, from the part of its outline that lies on its convex hull."""
    outline = contour.reshape(-1, 2)
    hull = cv2.convexHull(outline)
    on_hull = np.array(
        [
            abs(cv2.pointPolygonTest(hull, (float(x), float(y)), True)) < _HULL_TOLERANCE_PX
            for x, y in outline
        ]
    )
    points = outline[on_hull].astype(np.float64)
    lines: list[np.ndarray] = []
    for _ in range(4):
        if len(points) < _MIN_OUTLINE_POINTS:
            return None
        inliers = _ransac_inliers(points, params.outline_tolerance_px, rng)
        if inliers.sum() < params.min_outline_edge_frac * shorter_side:
            return None
        line = _fit_line(points[inliers])
        lines.append(line)
        # Drop a wider band than the inlier tolerance, so the same edge is not found twice.
        points = points[np.abs(points @ line[:2] + line[2]) > 3.0 * params.outline_tolerance_px]
    return lines


def _quad_from_lines(
    lines: list[np.ndarray], text_deg: float | None, params: FinderParams
) -> np.ndarray | None:
    """Corners from four lines: two along the rows of print and two across them."""
    degs = [_line_deg(line) for line in lines]
    if text_deg is not None:
        along = sorted(range(4), key=lambda i: _apart(degs[i], text_deg))[:2]
        if max(_apart(degs[i], text_deg) for i in along) > params.text_agree_deg:
            return None
    else:
        along = [0, min(range(1, 4), key=lambda i: _apart(degs[0], degs[i]))]
    across = [i for i in range(4) if i not in along]
    reference = text_deg if text_deg is not None else degs[along[0]]
    if max(_apart(degs[i], reference + 90.0) for i in across) > 2 * params.text_agree_deg:
        return None
    corners = [_intersect(lines[a], lines[c]) for a in along for c in across]
    if any(corner is None for corner in corners):
        return None
    return np.array(corners, dtype=np.float64)


def _ordered(quad: np.ndarray, text_deg: float | None) -> np.ndarray:
    """Top-left, top-right, bottom-right, bottom-left: the top edge runs left to right along the rows of print.

    Without a direction from print, the shorter edges are taken as top and bottom.
    """
    centre = quad.mean(axis=0)
    ring = quad[np.argsort(np.arctan2(quad[:, 1] - centre[1], quad[:, 0] - centre[0]))]
    best, best_key = ring, None
    for shift in range(4):
        candidate = np.roll(ring, -shift, axis=0)
        top = candidate[1] - candidate[0]
        if top[0] <= 0:
            continue
        if text_deg is not None:
            key = _apart(float(np.degrees(np.arctan2(top[1], top[0]))), text_deg)
        else:
            key = float(np.linalg.norm(top) - np.linalg.norm(candidate[2] - candidate[1]))
        if best_key is None or key < best_key:
            best, best_key = candidate, key
    return best


def _refine_edge(
    image: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    centre: np.ndarray,
    reach: tuple[float, float],
    params: FinderParams,
    rng: np.random.Generator,
    *,
    outward: bool = False,
) -> tuple[np.ndarray | None, float]:
    """Re-fit one edge at full size from the first sharp step met walking across it.

    The walk runs in from outside the sheet, where the desk is smooth and a shadow's edge is a slow ramp. With
    `outward` it runs out from inside the sheet instead, for a surface whose own texture is full of sharp steps:
    the paper's margin is smooth, so the first sharp step met from inside is the paper's edge.

    Returns:
        The line, or None when too little of the edge is seen, and the share of the sampled positions inside the
        frame that agree with it.
    """
    h, w = image.shape[:2]
    along = end - start
    length = float(np.hypot(*along))
    if length < 1e-6:
        return None, 0.0
    along = along / length
    inward = np.array([-along[1], along[0]])
    if (centre - start) @ inward < 0:
        inward = -inward
    offsets = np.arange(-reach[0], reach[1] + 1.0)
    span = params.step_span_px
    hits = []
    sampled = 0
    for t in np.linspace(0.03, 0.97, _EDGE_SAMPLES):
        base = start + along * (t * length)
        xs = np.round(base[0] + offsets * inward[0]).astype(int)
        ys = np.round(base[1] + offsets * inward[1]).astype(int)
        in_frame = (xs >= 0) & (ys >= 0) & (xs < w) & (ys < h)
        walk = offsets
        if not in_frame.all():
            # A sheet close to the frame's border leaves less room outside it than the walk asks for. The walk
            # is then started at the border, provided some desk is still seen outside the coarse edge and the
            # whole inside reach is in frame. Where the edge itself is off the frame, the position neither
            # supports the edge nor counts against it.
            if not in_frame[offsets >= 0].all() or not in_frame[offsets <= -2 * span].any():
                continue
            xs, ys, walk = xs[in_frame], ys[in_frame], offsets[in_frame]
        if outward:
            xs, ys, walk = xs[::-1], ys[::-1], walk[::-1]
        sampled += 1
        profile = image[ys, xs]
        step = np.abs(profile[span:] - profile[:-span]).max(axis=1)
        sharp = np.nonzero(step > params.step_min)[0]
        if len(sharp) == 0:
            continue
        first = int(sharp[0])
        peak = first + int(np.argmax(step[first : first + 2 * span]))
        hits.append(base + walk[min(peak + span // 2, len(walk) - 1)] * inward)
    if sampled == 0 or len(hits) < _MIN_EDGE_HITS:
        return None, 0.0
    points = np.array(hits)
    inliers = _ransac_inliers(points, params.edge_tolerance_px, rng)
    return _fit_line(points[inliers]), float(inliers.sum()) / sampled


def _refine(
    image: np.ndarray,
    quad: np.ndarray,
    params: FinderParams,
    rng: np.random.Generator,
    *,
    from_inside: bool = False,
) -> tuple[np.ndarray, int]:
    """Pull each edge of a quad onto the paper's edge. An edge that is not confirmed stays where it was.

    Args:
        image: The photograph, blurred, as float32.
        quad: The coarse corners, in photograph pixels.
        params: The finder's parameters.
        rng: Source of randomness for the line fits.
        from_inside: Walk out from inside the sheet over the paper method's shorter reaches, where the default
            walks in from outside over the desk method's.

    Returns:
        The refined quad, and how many of its edges the final pass confirmed.
    """
    diagonal = float(np.linalg.norm(quad[2] - quad[0]))
    if from_inside:
        reach = (params.paper_refine_outside_frac * diagonal, params.paper_refine_inside_frac * diagonal)
        passes = (reach, reach)
    else:
        passes = (
            (params.refine_outside_first_frac * diagonal, params.refine_inside_first_frac * diagonal),
            (params.refine_outside_final_frac * diagonal, params.refine_inside_final_frac * diagonal),
        )
    confirmed = 0
    for reach in passes:
        centre = quad.mean(axis=0)
        edges = []
        confirmed = 0
        for i in range(4):
            start, end = quad[i].astype(float), quad[(i + 1) % 4].astype(float)
            line, share = _refine_edge(image, start, end, centre, reach, params, rng, outward=from_inside)
            if line is None or share < params.min_edge_support:
                line = _fit_line(np.array([start, end]))
            else:
                confirmed += 1
            edges.append(line)
        # Edge i runs from corner i to corner i+1, so corner i is where edge i-1 meets edge i.
        corners = [_intersect(edges[i - 1], edges[i]) for i in range(4)]
        if any(corner is None for corner in corners):
            return quad, 0
        quad = np.array(corners, dtype=np.float64)
    return quad, confirmed


def find_sheets(rgb: np.ndarray, params: FinderParams) -> list[FoundSheet]:
    """Find every sheet of paper in a photograph.

    Args:
        rgb: The photograph, uint8 (H, W, 3), RGB.
        params: The finder's parameters.

    Returns:
        The sheets found, largest first. Empty when the frame holds none.
    """
    rng = np.random.default_rng(params.ransac_seed)
    h, w = rgb.shape[:2]
    scale = min(1.0, params.work_long_edge / max(h, w))
    small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    marks, text_height = print_marks(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), params)
    owner, sheets = sheet_regions(small, text_height, params)
    full = cv2.GaussianBlur(rgb, (0, 0), params.blur_sigma_px).astype(np.float32)

    found: list[FoundSheet] = []
    for number in range(1, sheets + 1):
        region = (owner == number).astype(np.uint8)
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea)
        x, y, box_w, box_h = cv2.boundingRect(contour)
        filled = np.zeros_like(region)
        cv2.drawContours(filled, [contour], -1, 1, -1)
        sheet_marks = np.where(filled > 0, marks, 0).astype(np.uint8)
        text_deg = text_direction(sheet_marks[y : y + box_h, x : x + box_w], text_height, params)
        lines = _hull_lines(contour, float(min(box_w, box_h)), params, rng)
        coarse = _quad_from_lines(lines, text_deg, params) if lines is not None else None
        outline = "edges"
        if coarse is None:
            coarse, outline = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float64), "box"
        quad, confirmed = _refine(full, _ordered(coarse, text_deg) / scale, params, rng)
        found.append(FoundSheet(quad.astype(np.float32), text_deg, outline, confirmed))

    # A sheet lies on the desk, not on another sheet: a shape whose centre is inside a larger one is part of it.
    kept: list[FoundSheet] = []
    for sheet in sorted(found, key=lambda s: -float(cv2.contourArea(s.quad))):
        cx, cy = sheet.quad.mean(axis=0)
        if not any(cv2.pointPolygonTest(k.quad, (float(cx), float(cy)), False) >= 0 for k in kept):
            kept.append(sheet)
    return kept


def _region_fit(region: np.ndarray, quad: np.ndarray) -> float:
    """How well a region and a quadrilateral agree: the area they share over the area either covers."""
    inside = np.zeros(region.shape, np.uint8)
    cv2.fillConvexPoly(inside, np.round(quad).astype(np.int32), 1)
    either = int(np.count_nonzero((inside > 0) | (region > 0)))
    return float(np.count_nonzero((inside > 0) & (region > 0))) / max(either, 1)


def _hull_quad(region: np.ndarray) -> np.ndarray | None:
    """The four corners of a region's convex hull, or of its bounding rectangle when the hull has no four."""
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    perimeter = cv2.arcLength(hull, True)
    for share in _HULL_QUAD_SHARES:
        approx = cv2.approxPolyDP(hull, share * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float64)
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float64)


def _cut_paper(bgr: np.ndarray, mask: np.ndarray, seed: np.ndarray, params: FinderParams) -> np.ndarray:
    """Run `cv2.grabCut` from a labelled mask and return the connected paper region that holds the seed.

    The region is empty when the mask leaves nothing that could be paper, or nothing that could be surroundings
    (print covering the whole frame, say): grabCut cannot start without a sample of each.
    """
    could_be_paper = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
    if could_be_paper.all() or not could_be_paper.any():
        return np.zeros(mask.shape, np.uint8)
    # grabCut starts its colour models with k-means, which draws at random: seeded, a photograph always gives
    # the same sheet. The rectangle argument is unused when the mask carries the labels.
    cv2.setRNGSeed(params.ransac_seed)
    models = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(bgr, mask, None, *models, params.paper_rounds, cv2.GC_INIT_WITH_MASK)  # type: ignore[call-overload]
    labelled = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    kernel = np.ones((_PAPER_OPEN_PX, _PAPER_OPEN_PX), np.uint8)
    count, labels = cv2.connectedComponents(
        cv2.morphologyEx(labelled, cv2.MORPH_OPEN, kernel), connectivity=8
    )
    holding = np.bincount(labels[seed > 0], minlength=count)
    holding[0] = 0
    if holding.max() == 0:
        return np.zeros(mask.shape, np.uint8)
    return (labels == int(np.argmax(holding))).astype(np.uint8)


def find_sheet_by_paper(
    rgb: np.ndarray, params: FinderParams, box: tuple[float, float, float, float] | None = None
) -> PaperSheet | None:
    """Find the one sheet a photograph shows, from a model of the paper and not of the desk.

    Args:
        rgb: The photograph, uint8 (H, W, 3), RGB.
        params: The finder's parameters.
        box: The vision model's box for the document, as shares of the photograph's width and height (left,
            top, right, bottom, each from 0 to 1), or None. It says which print is the document's; it is not
            trusted for where the sheet ends.

    Returns:
        The sheet, or None when the photograph shows no print, when its border is the colour of the paper under
        the print (a page that fills the frame: there is nothing to cut away), or when no shape is found.
    """
    full_h, full_w = rgb.shape[:2]
    scale = min(1.0, params.paper_work_long_edge / max(full_h, full_w))
    small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    h, w = small.shape[:2]
    marks, text_height = print_marks(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), params)
    if text_height <= 0:
        return None
    join = max(3, int(round(params.paper_join_heights * text_height)) | 1)
    blocks = cv2.morphologyEx(marks, cv2.MORPH_CLOSE, np.ones((join, join), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(blocks, connectivity=8)
    if count < 2:
        return None
    areas = stats[:, cv2.CC_STAT_AREA].astype(float)
    areas[0] = 0.0

    # Which print is the document's: the largest block, or the blocks the model's box mostly covers.
    in_box = None
    if box is None:
        chosen = [int(np.argmax(areas))]
    else:
        in_box = np.zeros((h, w), np.uint8)
        left, top, right, bottom = int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)
        in_box[max(0, top) : bottom + 1, max(0, left) : right + 1] = 1
        covered = np.bincount(labels[in_box > 0], minlength=count).astype(float)
        covered[0] = 0.0
        mostly = np.flatnonzero(
            (covered >= params.paper_block_in_box_min * np.maximum(areas, 1.0)) & (areas > 0)
        )
        if len(mostly) == 0:
            mostly = np.array([int(np.argmax(covered))])
        chosen = [int(i) for i in mostly if areas[i] >= params.paper_block_min_share * areas[mostly].max()]
    outlines = [
        contour
        for i in chosen
        for contour in cv2.findContours(
            (labels == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0]
    ]
    seed = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(seed, cv2.convexHull(np.vstack(outlines)), 1)

    # With a box, only the neighbourhood of the document is looked at: the box and the print it claims, and a
    # margin wide enough for blank paper beyond the print. What lies further off (the rest of the table, another
    # receipt) is neither segmented nor counted as print outside the sheet.
    origin = np.zeros(2)
    if in_box is not None:
        ys, xs = np.nonzero((in_box > 0) | (seed > 0))
        grow_x = params.paper_region_grow_frac * (xs.max() - xs.min() + 1)
        grow_y = params.paper_region_grow_frac * (ys.max() - ys.min() + 1)
        x0, x1 = max(0, int(xs.min() - grow_x)), min(w, int(xs.max() + grow_x) + 1)
        y0, y1 = max(0, int(ys.min() - grow_y)), min(h, int(ys.max() + grow_y) + 1)
        small = np.ascontiguousarray(small[y0:y1, x0:x1])
        marks, seed, in_box = marks[y0:y1, x0:x1], seed[y0:y1, x0:x1], in_box[y0:y1, x0:x1]
        h, w = small.shape[:2]
        origin = np.array([x0, y0], dtype=np.float64)

    # The paper's colour: what lies between the marks, inside the seed.
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    between = (seed > 0) & (cv2.dilate(marks, np.ones((3, 3), np.uint8)) == 0)
    if not between.any():
        return None
    paper_colour = np.median(lab[between], axis=0)
    # A page that fills the frame has no surroundings: its border is the colour of its own paper.
    border = max(2, int(params.paper_border_frac * min(h, w)))
    ring = np.ones((h, w), bool)
    ring[border:-border, border:-border] = False
    ring_colour = np.median(lab[ring], axis=0)
    if float(np.linalg.norm(ring_colour - paper_colour)) < params.paper_min_border_difference:
        return None

    # First pass: paper near the print (and inside the box), surroundings elsewhere, each only "probably".
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    if in_box is None:
        grow = max(3, int(round(2 * text_height)) | 1)
        mask[cv2.dilate(seed, np.ones((grow, grow), np.uint8)) > 0] = cv2.GC_PR_FGD
    else:
        mask[(in_box > 0) | (seed > 0)] = cv2.GC_PR_FGD
    mask[ring] = cv2.GC_BGD
    mask[seed > 0] = cv2.GC_FGD
    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    region = _cut_paper(bgr, mask, seed, params)
    quad = _hull_quad(region)
    if quad is None:
        return None

    # Second pass: a sheet can only continue along its own edges, so the two strips between its opposite edge
    # lines, run out to the frame, are offered as probable paper. The four corner regions outside both strips
    # stay surroundings, and the model of the surroundings is learnt from them.
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    centre = quad.mean(axis=0)
    inside = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        side = (b[0] - a[0]) * (ys - a[1]) - (b[1] - a[1]) * (xs - a[0])
        at_centre = (b[0] - a[0]) * (centre[1] - a[1]) - (b[1] - a[1]) * (centre[0] - a[0])
        inside.append(side * np.sign(at_centre) >= 0)
    strips = (inside[0] & inside[2]) | (inside[1] & inside[3])
    mask = np.where(strips, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    mask[ring] = cv2.GC_BGD
    mask[seed > 0] = cv2.GC_FGD
    longer = _cut_paper(bgr, mask, seed, params)
    longer_quad = _hull_quad(longer)
    fit = _region_fit(region, quad)
    if longer_quad is not None:
        longer_fit = _region_fit(longer, longer_quad)
        # On a surface the first pass already got right, the strips let desk in and the shape goes ragged.
        if longer_fit >= params.paper_extension_min_fit:
            region, quad, fit = longer, longer_quad, longer_fit

    within = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(within, np.round(quad).astype(np.int32), 1)
    printed = int(np.count_nonzero(marks))
    print_inside = float(np.count_nonzero(marks[within > 0])) / printed if printed else 0.0

    text_deg = text_direction(np.where(region > 0, marks, 0).astype(np.uint8), text_height, params)
    full = cv2.GaussianBlur(rgb, (0, 0), params.blur_sigma_px).astype(np.float32)
    rng = np.random.default_rng(params.ransac_seed)
    coarse = (_ordered(quad, text_deg) + origin) / scale
    refined, confirmed = _refine(full, coarse, params, rng, from_inside=True)
    return PaperSheet(refined.astype(np.float32), text_deg, confirmed, fit, print_inside)


def warp_upright(rgb: np.ndarray, quad: np.ndarray, fill_rgb: tuple[int, int, int]) -> np.ndarray:
    """Warp a sheet to an upright rectangle sized by its edge lengths.

    Args:
        rgb: The photograph, uint8 (H, W, 3).
        quad: The sheet's corners: top-left, top-right, bottom-right, bottom-left.
        fill_rgb: Colour for any part of the sheet that lay outside the frame.

    Returns:
        The upright sheet, uint8.
    """
    tl, tr, br, bl = quad.astype(np.float32)
    width = max(int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))), 1)
    height = max(int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))), 1)
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(quad.astype(np.float32), target)
    return cv2.warpPerspective(
        rgb,
        transform,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill_rgb,
    )
