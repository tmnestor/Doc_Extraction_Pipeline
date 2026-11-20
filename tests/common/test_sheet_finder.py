"""Every sheet in a photograph is found, the right way up, with its corners where the paper's are.

The photographs are made in the test: sheets printed with rows of letter-sized marks, laid on a desk at their own
angles, with a drop shadow, blur and noise. The true corners are therefore known exactly and no evaluation image
is needed.
"""

import dataclasses
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from common.sheet_finder import (
    FinderParams,
    _cut_paper,
    PaperSheet,
    find_sheet_by_paper,
    find_sheets,
    print_marks,
    text_direction,
    warp_upright,
)
from common.sheet_finder_config import FinderConfigError, finder_params_from, load_finder_params

RUN_CONFIG = Path(__file__).resolve().parents[2] / "config" / "run_config.yml"
PARAMS = load_finder_params(RUN_CONFIG)
SHEET_W, SHEET_H = 352, 704
# Placements are given on a 900 by 640 layout and scaled by this, so the sheets are about the size (in pixels)
# of a receipt in a phone photograph shrunk to the finder's working size.
LAYOUT_SCALE = 1.6
# 1% of a sheet's diagonal (about 790 px).
CORNER_TOLERANCE_PX = 8.0


def _sheet(rng: np.random.Generator, paper: int = 232) -> np.ndarray:
    """A sheet of paper printed like a receipt: rows of letter-sized marks on a monospaced grid."""
    sheet = np.full((SHEET_H, SHEET_W, 3), paper, np.uint8)
    for row in range(24, SHEET_H - 24, 17):
        start = int(rng.integers(0, 6))
        length = int(rng.integers(9, 22))
        for column in range(start, start + length):
            if rng.random() < 0.15:
                continue
            x = 14 + column * 8
            if x + 5 < SHEET_W - 12:
                cv2.rectangle(sheet, (x, row), (x + 4, row + 8), (40, 40, 40), -1)
    return sheet


def _photo(
    seed: int,
    placements: list[tuple[float, float, float]],
    *,
    desk: tuple[int, int, int] = (176, 150, 128),
    paper: int = 232,
    frame: tuple[int, int] = (900, 640),
) -> tuple[np.ndarray, list[np.ndarray]]:
    """A photograph of sheets on a desk, and each sheet's true corners (TL, TR, BR, BL).

    Each placement is (centre x, centre y, turn in degrees).
    """
    rng = np.random.default_rng(seed)
    width, height = int(frame[0] * LAYOUT_SCALE), int(frame[1] * LAYOUT_SCALE)
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.array(desk, np.float64) + (xx * 0.02 - yy * 0.015)[..., None]
    flat = np.array([[0, 0], [SHEET_W, 0], [SHEET_W, SHEET_H], [0, SHEET_H]], np.float32)
    layers, quads = [], []
    for cx, cy, turn in placements:
        theta = np.radians(turn)
        rotate = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        centre = [cx * LAYOUT_SCALE, cy * LAYOUT_SCALE]
        quad = ((flat - [SHEET_W / 2, SHEET_H / 2]) @ rotate.T + centre).astype(np.float32)
        # A slight foreshortening of the top edge, as from a camera not held square.
        quad[0][0] += 6
        quad[1][0] -= 6
        rgba = np.dstack([_sheet(rng, paper), np.full((SHEET_H, SHEET_W), 255, np.uint8)])
        layers.append(
            cv2.warpPerspective(
                rgba, cv2.getPerspectiveTransform(flat, quad), (width, height), flags=cv2.INTER_CUBIC
            )
        )
        quads.append(quad)
    for layer in layers:
        # A soft shadow, as under room light: its edge is a slow ramp, unlike the paper's.
        shadow = cv2.GaussianBlur(layer[:, :, 3], (0, 0), 14.0) * 0.45
        shadow = np.roll(np.roll(shadow, 10, axis=0), 10, axis=1)[..., None] / 255.0
        image = image * (1 - shadow) + np.array([25.0, 22.0, 20.0]) * shadow
    for layer in layers:
        alpha = layer[:, :, 3:4].astype(np.float64) / 255.0
        image = image * (1 - alpha) + layer[:, :, :3] * alpha
    image = cv2.GaussianBlur(image, (0, 0), 0.6) + rng.normal(0, 2.0, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8), quads


def _match(found, true: np.ndarray):
    """The sheet found at a true sheet's place."""
    near = [s for s in found if np.linalg.norm(s.quad.mean(axis=0) - true.mean(axis=0)) < 100]
    assert len(near) == 1, f"{len(near)} sheets found where one lies"
    return near[0]


def _worst_corner(found_quad: np.ndarray, true: np.ndarray) -> float:
    """The largest distance between a found corner and the true corner of the same name."""
    return float(np.linalg.norm(found_quad - true, axis=1).max())


def test_one_sheet_is_found_with_its_corners_in_place() -> None:
    photo, (true,) = _photo(1, [(450, 320, 8.0)])

    found = find_sheets(photo, PARAMS)

    assert len(found) == 1
    assert _worst_corner(found[0].quad, true) < CORNER_TOLERANCE_PX
    assert found[0].edges_confirmed == 4


def test_three_sheets_at_their_own_angles_are_each_found() -> None:
    photo, quads = _photo(2, [(200, 350, -14.0), (600, 360, 6.0), (1000, 340, 19.0)], frame=(1200, 700))

    found = find_sheets(photo, PARAMS)

    assert len(found) == 3
    for true in quads:
        assert _worst_corner(_match(found, true).quad, true) < CORNER_TOLERANCE_PX


def test_each_sheets_rows_of_print_give_its_direction() -> None:
    turns = [-14.0, 6.0, 19.0]
    photo, quads = _photo(
        3, [(200, 350, turns[0]), (600, 360, turns[1]), (1000, 340, turns[2])], frame=(1200, 700)
    )

    found = find_sheets(photo, PARAMS)

    for true, turn in zip(quads, turns, strict=True):
        sheet = _match(found, true)
        assert sheet.text_deg is not None
        assert abs(sheet.text_deg - turn) < 1.5


def test_a_monospaced_grid_of_letters_is_read_along_its_rows_not_its_columns() -> None:
    """Letters on a monospaced grid line up in columns as sharply as in rows; rows must still win."""
    marks = np.zeros((300, 300), np.uint8)
    for row in range(20, 280, 17):
        for column in range(10, 280, 8):
            cv2.rectangle(marks, (column, row), (column + 4, row + 8), 255, -1)
    turned = cv2.warpAffine(marks, cv2.getRotationMatrix2D((150, 150), -12.0, 1.0), (300, 300))

    assert text_direction(marks, 9.0, PARAMS) == pytest.approx(0.0, abs=1.0)
    assert text_direction(turned, 9.0, PARAMS) == pytest.approx(12.0, abs=1.0)


def test_the_first_corner_is_the_sheets_own_top_left() -> None:
    """A sheet turned 25 degrees either way is returned the right way up, not turned a quarter."""
    for turn in (-25.0, 25.0):
        photo, (true,) = _photo(4, [(450, 320, turn)])

        (sheet,) = find_sheets(photo, PARAMS)

        assert int(np.argmin(np.linalg.norm(sheet.quad - true[0], axis=1))) == 0
        assert _worst_corner(sheet.quad, true) < CORNER_TOLERANCE_PX


def test_a_shadow_joining_two_sheets_does_not_join_them() -> None:
    """Two sheets about 40 px apart: their drop shadows overlap in the gap between them."""
    photo, quads = _photo(5, [(330, 320, 0.0), (576, 320, 0.0)])

    found = find_sheets(photo, PARAMS)

    assert len(found) == 2
    for true in quads:
        assert _worst_corner(_match(found, true).quad, true) < CORNER_TOLERANCE_PX


def test_grey_paper_on_a_grey_desk_is_found_in_one_piece() -> None:
    """Paper the colour of the desk in shadow: only its sharp edge sets it apart."""
    photo, quads = _photo(6, [(300, 320, -9.0), (620, 320, 11.0)], desk=(170, 168, 166), paper=140)

    found = find_sheets(photo, PARAMS)

    assert len(found) == 2
    for true in quads:
        assert _worst_corner(_match(found, true).quad, true) < 2 * CORNER_TOLERANCE_PX


def test_a_bare_desk_holds_no_sheet() -> None:
    photo, _ = _photo(7, [])

    assert find_sheets(photo, PARAMS) == []


def test_the_same_photograph_always_gives_the_same_corners() -> None:
    photo, _ = _photo(8, [(300, 320, -9.0), (620, 320, 11.0)])

    first = find_sheets(photo, PARAMS)
    again = find_sheets(photo, PARAMS)

    assert [s.quad.tolist() for s in first] == [s.quad.tolist() for s in again]


def test_a_sheet_is_warped_upright_at_its_own_size() -> None:
    photo, (true,) = _photo(9, [(450, 320, 17.0)])
    (sheet,) = find_sheets(photo, PARAMS)

    upright = warp_upright(photo, sheet.quad, (255, 255, 255))

    assert abs(upright.shape[1] - SHEET_W) <= 8 and abs(upright.shape[0] - SHEET_H) <= 8
    # Paper to every corner: no desk is left in the frame.
    for y, x in ((6, 6), (6, -7), (-7, 6), (-7, -7)):
        assert upright[y, x].min() > 190
    # The rows of print now run straight across.
    marks, text_height = print_marks(cv2.cvtColor(upright, cv2.COLOR_RGB2GRAY), PARAMS)
    assert text_direction(marks, text_height, PARAMS) == pytest.approx(0.0, abs=1.0)


def _section() -> dict:
    return dict(yaml.safe_load(RUN_CONFIG.read_text())["pipeline"]["photo_split"]["finder"])


def assert_four_elements(error: BaseException, *fragments: str) -> None:
    message = str(error)
    for element in ("What:", "Where:", "Expected:", "How to fix:"):
        assert element in message
    assert str(RUN_CONFIG.resolve()) in message and "pipeline.photo_split.finder" in message
    for fragment in fragments:
        assert fragment in message


def test_the_shipped_config_states_every_setting_and_nothing_else() -> None:
    assert set(_section()) == {field.name for field in dataclasses.fields(FinderParams)}
    assert isinstance(PARAMS.work_long_edge, int) and isinstance(PARAMS.step_min, float)


@pytest.mark.parametrize("name", [field.name for field in dataclasses.fields(FinderParams)])
def test_a_missing_setting_is_refused_by_name(name: str) -> None:
    section = _section()
    del section[name]

    with pytest.raises(FinderConfigError) as raised:
        finder_params_from(section, config_path=RUN_CONFIG)
    assert_four_elements(raised.value, f"missing key {name}", f"pipeline.photo_split.finder.{name}")


@pytest.mark.parametrize(
    ("name", "value", "said"),
    [
        ("work_long_edge", 1600.5, "not a whole number"),
        ("step_span_px", True, "not a whole number"),
        ("step_min", "nine", "not a number"),
        ("min_edge_support", False, "not a number"),
        ("min_region_frac", 0, "not above zero"),
        ("blur_sigma_px", -1.0, "not above zero"),
        ("mark_block_px", 30, "not an odd number"),
    ],
)
def test_a_setting_of_the_wrong_kind_is_refused(name: str, value: object, said: str) -> None:
    section = _section()
    section[name] = value

    with pytest.raises(FinderConfigError) as raised:
        finder_params_from(section, config_path=RUN_CONFIG)
    assert_four_elements(raised.value, name, said)


def test_a_misspelt_setting_is_refused_not_ignored() -> None:
    section = _section()
    section["step_minimum"] = 9.0

    with pytest.raises(FinderConfigError) as raised:
        finder_params_from(section, config_path=RUN_CONFIG)
    assert_four_elements(raised.value, "unknown key(s) ['step_minimum']")


def test_a_config_without_the_section_or_the_file_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "run_config.yml"
    empty.write_text("pipeline: {}\n")

    for path, said in ((empty, "missing or is not a mapping"), (tmp_path / "absent.yml", "does not exist")):
        with pytest.raises(FinderConfigError) as raised:
            load_finder_params(path)
        message = str(raised.value)
        assert said in message and str(path.resolve()) in message
        assert all(element in message for element in ("What:", "Where:", "Expected:", "How to fix:"))


def test_a_seed_of_zero_is_allowed() -> None:
    assert PARAMS.ransac_seed == 0


# ---- the paper method --------------------------------------------------------------------------------------
# One sheet on a surface the desk method cannot model: speckled like felt and of two colours. The paper method
# learns the paper from the print and must find the same corners.

PAGE_W, PAGE_H = 520, 760


def _page(rng: np.random.Generator, printed_share: float = 0.92) -> np.ndarray:
    """A page of print. `printed_share` below 1 leaves the rest of the page blank, as below a short statement."""
    page = np.full((PAGE_H, PAGE_W, 3), 236, np.uint8)
    for row in range(30, int(PAGE_H * printed_share) - 20, 18):
        for column in range(int(rng.integers(0, 4)), 56):
            if rng.random() < 0.2:
                continue
            x = 24 + column * 8
            cv2.rectangle(page, (x, row), (x + 4, row + 8), (38, 38, 38), -1)
    return page


def _on_felt(
    seed: int, turn: float, *, printed_share: float = 0.92, shade_blank: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """A page on a speckled, two-coloured surface, and its true corners (TL, TR, BR, BL)."""
    rng = np.random.default_rng(seed)
    width, height = 900, 1100
    xx = np.mgrid[0:height, 0:width][1]
    base = np.where((xx < 330)[..., None], np.array([52.0, 54.0, 58.0]), np.array([60.0, 118.0, 74.0]))
    speckle = cv2.GaussianBlur(rng.normal(0, 60.0, (height, width, 3)), (0, 0), 1.4)
    image = base + speckle
    page = _page(rng, printed_share).astype(np.float64)
    if shade_blank:
        # A soft shadow over the blank foot of the page: darker, with no sharp edge.
        ramp = np.clip((np.mgrid[0:PAGE_H, 0:PAGE_W][0] - PAGE_H * printed_share) / 60.0, 0, 1)[..., None]
        page = page * (1 - shade_blank * ramp)
    flat = np.array([[0, 0], [PAGE_W, 0], [PAGE_W, PAGE_H], [0, PAGE_H]], np.float32)
    theta = np.radians(turn)
    rotate = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    quad = flat.copy()
    quad[0][0] += 30
    quad[1][0] -= 30
    quad = ((quad - [PAGE_W / 2, PAGE_H / 2]) @ rotate.T + [width / 2, height / 2]).astype(np.float32)
    rgba = np.dstack([page, np.full((PAGE_H, PAGE_W), 255.0)]).astype(np.uint8)
    layer = cv2.warpPerspective(
        rgba, cv2.getPerspectiveTransform(flat, quad), (width, height), flags=cv2.INTER_CUBIC
    )
    alpha = layer[:, :, 3:4].astype(np.float64) / 255.0
    image = image * (1 - alpha) + layer[:, :, :3] * alpha
    image = cv2.GaussianBlur(image, (0, 0), 0.6) + rng.normal(0, 2.0, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8), quad


def _paper_error(photo: np.ndarray, true: np.ndarray, **kwargs) -> tuple[float, PaperSheet]:
    """The worst corner's distance from the truth, as a share of the page's diagonal, and the sheet found."""
    sheet = find_sheet_by_paper(photo, PARAMS, **kwargs)
    assert sheet is not None
    return _worst_corner(sheet.quad, true) / float(np.linalg.norm(true[2] - true[0])), sheet


def test_the_desk_method_finds_no_usable_sheet_on_a_speckled_surface() -> None:
    """What the paper method is for: here the desk method gives nothing, or not the page."""
    photo, true = _on_felt(31, 7.0)

    found = find_sheets(photo, PARAMS)

    assert not any(_worst_corner(s.quad, true) < 20 for s in found)


@pytest.mark.parametrize(("seed", "turn"), [(31, 7.0), (32, -11.0), (33, 0.0)])
def test_the_paper_method_finds_the_page_on_a_speckled_surface(seed: int, turn: float) -> None:
    photo, true = _on_felt(seed, turn)

    error, sheet = _paper_error(photo, true)

    assert error < 0.01
    assert sheet.edges_confirmed == 4 and sheet.fit > 0.97
    assert sheet.text_deg == pytest.approx(turn, abs=2.0)


def test_blank_paper_below_the_print_is_part_of_the_sheet() -> None:
    """Print stops two thirds of the way down. The sheet does not: its foot is blank, and here in shadow."""
    photo, true = _on_felt(34, 5.0, printed_share=0.62, shade_blank=0.35)

    error, sheet = _paper_error(photo, true)

    assert error < 0.01 and sheet.fit > 0.97


def test_a_page_that_fills_the_frame_has_nothing_to_cut_away() -> None:
    """A clean page, not a photograph: its border is its own paper, so the paper method stops at once."""
    rng = np.random.default_rng(35)
    page = cv2.resize(_page(rng), None, fx=1.6, fy=1.6, interpolation=cv2.INTER_LINEAR)

    assert find_sheet_by_paper(page, PARAMS) is None


def test_a_photograph_with_no_print_gives_no_sheet() -> None:
    photo, _ = _photo(36, [])

    assert find_sheet_by_paper(photo, PARAMS) is None


def test_the_models_box_says_which_of_two_sheets_is_meant() -> None:
    photo, quads = _photo(37, [(250, 320, -6.0), (640, 320, 8.0)])
    height, width = photo.shape[:2]

    for true in quads:
        (x0, y0), (x1, y1) = true.min(axis=0), true.max(axis=0)
        error, _ = _paper_error(photo, true, box=(x0 / width, y0 / height, x1 / width, y1 / height))
        assert error < 0.02


def test_the_paper_method_always_gives_the_same_corners() -> None:
    photo, _ = _on_felt(38, -4.0)

    first = find_sheet_by_paper(photo, PARAMS)
    again = find_sheet_by_paper(photo, PARAMS)

    assert first is not None and again is not None
    assert first.quad.tolist() == again.quad.tolist() and first.fit == again.fit


def test_print_outside_the_sheet_counts_against_it_unless_a_box_sets_it_apart() -> None:
    """Two sheets. With no box the whole frame is looked at, and the other sheet is print outside this one,
    which says the quadrilateral is not everything. With the model box for one sheet, only its neighbourhood is
    looked at, and the other sheet, a document of its own, no longer counts against it."""
    photo, quads = _photo(39, [(250, 320, -6.0), (640, 320, 8.0)])
    height, width = photo.shape[:2]
    (x0, y0), (x1, y1) = quads[0].min(axis=0), quads[0].max(axis=0)

    alone = find_sheet_by_paper(photo, PARAMS)
    boxed = find_sheet_by_paper(photo, PARAMS, box=(x0 / width, y0 / height, x1 / width, y1 / height))

    assert alone is not None and 0.3 < alone.print_inside < 0.7
    assert boxed is not None and boxed.print_inside > 0.95


@pytest.mark.parametrize("label", [cv2.GC_PR_FGD, cv2.GC_PR_BGD])
def test_a_mask_with_nothing_of_one_kind_gives_an_empty_region_not_a_crash(label: int) -> None:
    """Met on a real photograph: print reached every edge, so nothing was left to learn the surroundings from,
    and grabCut refuses to start without a sample of paper and one of surroundings."""
    photo, _ = _on_felt(40, 0.0)
    mask = np.full(photo.shape[:2], label, np.uint8)
    seed = np.zeros(photo.shape[:2], np.uint8)
    seed[100:200, 100:200] = 1

    region = _cut_paper(cv2.cvtColor(photo, cv2.COLOR_RGB2BGR), mask, seed, PARAMS)

    assert region.shape == mask.shape and not region.any()


def test_print_to_every_edge_of_a_coloured_frame_gives_no_sheet() -> None:
    """The whole frame is print on a tinted ground: there is a border unlike paper, but nothing around the print."""
    rng = np.random.default_rng(41)
    frame = np.full((900, 700, 3), (150, 170, 190), np.uint8)
    for row in range(4, 890, 16):
        for column in range(2, 690, 8):
            if rng.random() > 0.15:
                cv2.rectangle(frame, (column, row), (column + 4, row + 8), (30, 30, 30), -1)

    assert find_sheet_by_paper(frame, PARAMS) is None
