"""A stage can load its engine with settings that differ from the model's own.

The classify stage needs less visual detail than extraction, and each GPU stage
loads its own engine, so the image token budget can be stated per stage. The
override must reach the engine only for that stage, leave every other setting
alone, and reject anything that would read as configured while doing nothing.
"""

import copy
from pathlib import Path

import pytest
import yaml

from common.settings import Settings
from common.vllm_stage_overrides import (
    STAGES,
    StageOverrideConfigError,
    apply_stage_overrides,
    validate_stage_overrides,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_CONFIG = REPO_ROOT / "config" / "run_config.yml"
GEMMA = "gemma4-31b-w4a16-vllm"


def model_block(tokens: int = 560, classify_tokens: int | None = 280) -> dict:
    block: dict = {
        "max_num_seqs": 16,
        "mm_processor_kwargs": {"max_soft_tokens": tokens},
        "hf_overrides": {
            "vision_soft_tokens_per_image": tokens,
            "vision_config": {"default_output_length": tokens, "other": "kept"},
        },
    }
    if classify_tokens is not None:
        block["stage_overrides"] = {
            "classify": {
                "mm_processor_kwargs": {"max_soft_tokens": classify_tokens},
                "hf_overrides": {
                    "vision_soft_tokens_per_image": classify_tokens,
                    "vision_config": {"default_output_length": classify_tokens},
                },
            }
        }
    return block


# --- applying -------------------------------------------------------------------


def test_the_stage_loads_its_own_values_and_keeps_everything_else():
    resolved = apply_stage_overrides(model_block(), "classify")

    assert resolved == {
        "max_num_seqs": 16,
        "mm_processor_kwargs": {"max_soft_tokens": 280},
        "hf_overrides": {
            "vision_soft_tokens_per_image": 280,
            "vision_config": {"default_output_length": 280, "other": "kept"},
        },
    }


def test_without_a_stage_the_models_own_values_are_loaded():
    resolved = apply_stage_overrides(model_block(), None)

    assert resolved["mm_processor_kwargs"] == {"max_soft_tokens": 560}
    assert resolved["hf_overrides"]["vision_config"]["default_output_length"] == 560
    assert "stage_overrides" not in resolved


def test_a_model_with_no_overrides_is_unchanged_for_any_stage():
    block = model_block(classify_tokens=None)

    assert apply_stage_overrides(block, "classify") == block


def test_applying_does_not_modify_the_models_block():
    block = model_block()
    before = copy.deepcopy(block)

    resolved = apply_stage_overrides(block, "classify")
    resolved["hf_overrides"]["vision_config"]["default_output_length"] = 1

    assert block == before


# --- validating -----------------------------------------------------------------


def assert_diagnostic(error: StageOverrideConfigError, *fragments: str) -> None:
    message = str(error)
    assert "What:" in message
    assert "Where:" in message and str(RUN_CONFIG) in message
    assert f"inference.vllm.models.{GEMMA}.stage_overrides" in message
    assert "Expected:" in message and "max_soft_tokens: 280" in message
    assert "How to fix:" in message
    for fragment in fragments:
        assert fragment in message


def test_a_valid_block_passes():
    validate_stage_overrides(
        {GEMMA: model_block(), "other-model": {"max_num_seqs": 2}}, config_path=RUN_CONFIG
    )


def broken(change) -> dict:
    block = model_block()
    change(block)
    return {GEMMA: block}


def _unknown_stage(block: dict) -> None:
    block["stage_overrides"]["extract"] = block["stage_overrides"].pop("classify")


def _unknown_block(block: dict) -> None:
    block["stage_overrides"]["classify"]["max_num_seqs"] = 4


def _misspelt_key(block: dict) -> None:
    block["stage_overrides"]["classify"]["mm_processor_kwargs"] = {"max_soft_token": 280}


def _misspelt_nested_key(block: dict) -> None:
    block["stage_overrides"]["classify"]["hf_overrides"]["vision_config"] = {"default_output_len": 280}


def _not_a_mapping(block: dict) -> None:
    block["stage_overrides"] = 280


def _empty_stage(block: dict) -> None:
    block["stage_overrides"]["classify"] = {}


def _value_where_a_mapping_is(block: dict) -> None:
    block["stage_overrides"]["classify"]["hf_overrides"]["vision_config"] = 280


def _block_the_model_does_not_state(block: dict) -> None:
    del block["mm_processor_kwargs"]


@pytest.mark.parametrize(
    ("change", "fragments"),
    [
        (_unknown_stage, ("'extract'", ".stage_overrides.extract")),
        (_unknown_block, ("'max_num_seqs'", ".classify.max_num_seqs")),
        (_misspelt_key, ("'max_soft_token'", ".classify.mm_processor_kwargs.max_soft_token")),
        (_misspelt_nested_key, ("'default_output_len'", ".hf_overrides.vision_config.default_output_len")),
        (_not_a_mapping, ("not a mapping",)),
        (_empty_stage, ("empty", ".stage_overrides.classify")),
        (_value_where_a_mapping_is, ("single value", ".hf_overrides.vision_config")),
        (_block_the_model_does_not_state, (".classify.mm_processor_kwargs",)),
    ],
)
def test_a_malformed_block_fails_with_a_diagnostic(change, fragments):
    with pytest.raises(StageOverrideConfigError) as caught:
        validate_stage_overrides(broken(change), config_path=RUN_CONFIG)

    assert_diagnostic(caught.value, *fragments)


# --- the config loader ----------------------------------------------------------


def app_config(vllm_config: dict) -> Settings:
    """An Settings holding only the vLLM settings, without reading the pipeline config."""
    config = object.__new__(Settings)
    config._vllm_config = vllm_config
    config._stage = None
    return config


def test_only_the_staged_config_hands_the_loader_the_stage_values():
    defaults = {"max_num_seqs": 1, "max_model_len": 8192}
    config = app_config({"__defaults__": defaults, GEMMA: {**defaults, **model_block()}})

    own = config.get_vllm_config(GEMMA)
    staged = config.for_stage("classify").get_vllm_config(GEMMA)

    assert own["mm_processor_kwargs"] == {"max_soft_tokens": 560}
    assert staged["mm_processor_kwargs"] == {"max_soft_tokens": 280}
    assert staged["hf_overrides"]["vision_soft_tokens_per_image"] == 280
    assert staged["max_num_seqs"] == own["max_num_seqs"] == 16
    assert staged["max_model_len"] == 8192
    assert "stage_overrides" not in own and "stage_overrides" not in staged
    assert config.get_vllm_config(GEMMA) == own


def test_a_model_without_its_own_block_gets_the_defaults_at_any_stage():
    config = app_config({"__defaults__": {"max_num_seqs": 1}})

    assert config.for_stage("classify").get_vllm_config("unlisted") == {"max_num_seqs": 1}


def test_an_unknown_stage_is_refused():
    with pytest.raises(ValueError, match="extract"):
        app_config({}).for_stage("extract")


# --- the shipped config ---------------------------------------------------------


def shipped_models() -> dict:
    return yaml.safe_load(RUN_CONFIG.read_text())["inference"]["vllm"]["models"]


def test_the_shipped_config_is_valid():
    validate_stage_overrides(shipped_models(), config_path=RUN_CONFIG)


def test_the_shipped_classify_values_move_together():
    block = shipped_models()[GEMMA]
    staged = apply_stage_overrides(block, "classify")

    values = {
        staged["mm_processor_kwargs"]["max_soft_tokens"],
        staged["hf_overrides"]["vision_soft_tokens_per_image"],
        staged["hf_overrides"]["vision_config"]["default_output_length"],
    }
    assert len(values) == 1
    assert values <= {70, 140, 280, 560, 1120}


def test_the_classify_stage_and_its_worker_load_through_the_staged_config():
    assert "classify" in STAGES
    assert '.for_stage("classify")' in (REPO_ROOT / "stages/classify.py").read_text()
    # An engine on a card is started with the settings the stage hands over, staged copy and all: see
    # tests/common/test_one_path.py. Nothing reads the settings again for a worker.
    assert "Settings.load" not in (REPO_ROOT / "common/vllm_dp.py").read_text()
