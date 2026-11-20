"""Config key tests for Qwen 3.8: the normalisation convention is not uniform.

``inference.vllm.models`` is keyed by the FULL registered type;
``inference.generation.models`` and ``pipeline.batch.*_sizes`` by the NORMALISED
form. A key that matches neither does not raise at lookup time: it falls through
to the ``defaults`` block, so the config reads as applied while doing nothing.
``_validate_model_override_keys`` is the guard against that, and these tests
exercise it directly rather than through ``AppConfig.load``, which validates
sandbox data directories that do not exist on a development machine.
"""

import pytest
import yaml

from common.app_config import AppConfig, _validate_model_override_keys

QWEN_FULL = "qwen38-27b-fp8-vllm"
QWEN_NORMALISED = "qwen38-27b-fp8"
GEMMA_FULL = "gemma4-31b-w4a16-vllm"
GEMMA_NORMALISED = "gemma4-31b"


@pytest.fixture(scope="module")
def raw_config(run_config_path):
    """The run config parsed as plain YAML, with no validation applied."""
    return yaml.safe_load(run_config_path.read_text())


def test_normalisation_strips_vllm_and_w4a16_but_not_fp8():
    # Traced empirically, not inferred from the Gemma case: -w4a16 is stripped,
    # -fp8 is NOT, and a trailing -fp8 also prevents the size-suffix strip.
    assert AppConfig._normalize_model_type(QWEN_FULL) == QWEN_NORMALISED
    assert AppConfig._normalize_model_type(GEMMA_FULL) == GEMMA_NORMALISED


def test_every_override_key_resolves_to_a_registered_model(raw_config):
    # The real guard from AppConfig.load. A misspelled key raises here.
    inference = raw_config["inference"]
    _validate_model_override_keys(
        vllm_models=inference["vllm"]["models"],
        generation_models=inference["generation"]["models"],
    )


def test_qwen38_vllm_block_is_keyed_by_the_full_type(raw_config):
    models = raw_config["inference"]["vllm"]["models"]
    assert QWEN_FULL in models
    assert QWEN_NORMALISED not in models, "the vllm section uses FULL types"
    block = models[QWEN_FULL]
    assert block["max_model_len"] == 16384
    assert block["limit_mm_per_prompt"] == 1
    # Pixel-area sizing knob: the sweep lever, stated explicitly rather than
    # left implicit at the checkpoint defaults.
    assert block["mm_processor_kwargs"]["min_pixels"] == 65536
    assert block["mm_processor_kwargs"]["max_pixels"] == 16777216


def test_qwen38_model_path_is_registered(raw_config):
    paths = raw_config["bootstrap"]["model"]["default_paths"]
    assert paths[QWEN_FULL] == "/home/jovyan/nfs_share/models/Qwen3.8-27B-FP8"


def test_model_path_cannot_diverge_from_model_type(raw_config):
    """An explicit `path` outranks `default_paths`, so it must not contradict `type`.

    ``merge_configs`` only consults ``default_paths`` when ``model_path`` is
    empty, and the entrypoint never passes ``--model``, so a stale ``path`` wins
    silently. On 2026-09-22 ``type`` was the 31B while ``path`` still pointed at
    the 12B checkpoint: the run would have loaded 12B weights and reported them
    as the 31B, with nothing in the output to contradict it.

    ``null`` is the correct value: it defers to ``default_paths[type]``.
    """
    model = raw_config["bootstrap"]["model"]
    explicit = model.get("path")
    if explicit is None:
        return  # deferring to default_paths, which is keyed by type, so safe

    expected = model["default_paths"].get(model["type"])
    assert expected is not None, f"{model['type']!r} has no entry in default_paths"
    assert explicit == expected, (
        f"bootstrap.model.path points at {explicit!r} but type is "
        f"{model['type']!r}, whose default_paths entry is {expected!r}. "
        "The explicit path wins, so this run would load the wrong weights."
    )


def test_input_dir_and_ground_truth_are_the_same_dataset(raw_config):
    """The answer key must belong to the images being scored.

    Filenames are identical across the clean and degraded sets and across dates,
    so a mismatch does not fail: it scores real extractions against the wrong
    answer key and reports plausible numbers. That is unrecoverable after the
    fact, because nothing in the output records which key was used.
    """
    from pathlib import Path

    source = raw_config["pipeline"]["information_extraction"]["input"]
    image_dir = Path(source["dir"])
    ground_truth = Path(source["ground_truth"])
    assert image_dir.name == ground_truth.parent.name, (
        f"dataset mismatch: images from {image_dir.name!r} would be scored "
        f"against the answer key in {ground_truth.parent.name!r}"
    )


def test_qwen38_has_a_batch_size_in_every_map(raw_config):
    batch = raw_config["pipeline"]["batch"]
    for name in ("default_sizes", "max_sizes", "conservative_sizes"):
        sizes = batch[name]
        assert QWEN_NORMALISED in sizes, f"{name} would fall back to batch.min_size"
        # Matching the Gemma 31B is the point: batching headroom is per document
        # type and the bank statements that dominate the corpus have none, so
        # raising either model alone would confound throughput.
        assert sizes[QWEN_NORMALISED] == sizes[GEMMA_NORMALISED] == 1
