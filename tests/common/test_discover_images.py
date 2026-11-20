"""`max_images` and `document_types` must apply on the stages path, not just the CLI.

Both keys are required in `pipeline.information_extraction.input` and are loaded
into PipelineConfig, but until 2026-09-21 only ``cli.py`` honoured them:
``stages/classify.py`` and ``stages/extract.py`` called ``discover_images``
with the directory alone. Asking for 12 images through
``KFP_TASK=run_info_extract`` silently processed all 165: the config read as
applied while doing nothing, which is the exact silent-no-op the fail-fast rule
exists to prevent.

The fix puts both filters inside ``discover_images`` so no caller can omit them.
These tests cover the function, then assert the two stage call sites actually
pass the config values through.
"""

import inspect
from pathlib import Path

import pytest

from common.pipeline_config import discover_images


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    """A directory of predictably named images across three document types."""
    for name in (
        "invoice_01.png",
        "invoice_02.png",
        "receipt_01.png",
        "receipt_02.png",
        "bank_statement_01.png",
    ):
        (tmp_path / name).write_bytes(b"")
    return tmp_path


def test_finds_everything_when_unfiltered(image_dir):
    assert len(discover_images(image_dir)) == 5


def test_max_images_truncates_after_sorting(image_dir):
    # Sorted by lowercase filename, so the limit is deterministic rather than
    # whatever order the filesystem happened to return.
    found = discover_images(image_dir, max_images=2)
    assert [p.name for p in found] == ["bank_statement_01.png", "invoice_01.png"]


def test_max_images_none_means_no_limit(image_dir):
    assert len(discover_images(image_dir, max_images=None)) == 5


def test_max_images_larger_than_the_set_is_harmless(image_dir):
    assert len(discover_images(image_dir, max_images=99)) == 5


def test_document_types_filters_by_filename(image_dir):
    found = discover_images(image_dir, document_types=["invoice"])
    assert [p.name for p in found] == ["invoice_01.png", "invoice_02.png"]


def test_document_types_and_max_images_compose(image_dir):
    # The type filter must run FIRST, or the limit eats images the filter would
    # have removed and the caller gets fewer than requested.
    found = discover_images(image_dir, document_types=["receipt"], max_images=1)
    assert [p.name for p in found] == ["receipt_01.png"]


@pytest.mark.parametrize("module_name", ["stages.classify", "stages.extract"])
def test_the_stage_call_sites_pass_both_filters(module_name):
    # Guards the regression that started this: a stage calling
    # discover_images(dir) alone silently ignores both config keys.
    import importlib

    source = inspect.getsource(importlib.import_module(module_name))
    for call in "discover_images(":
        assert call in source
    assert "max_images" in source, f"{module_name} never mentions max_images"
    assert "document_types" in source, f"{module_name} never mentions document_types"
