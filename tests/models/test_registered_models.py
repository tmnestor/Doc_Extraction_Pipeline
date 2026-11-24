"""The pipeline runs two models, and refuses any other name before doing any work.

Gemma 4 31B runs as one engine across both cards. Gemma 4 12B, quantised by its maker, runs as one engine
on each card. A model that was once registered and is no longer is a plain mistake in the settings, and
the message must say which names are allowed.
"""

from pathlib import Path

import pytest
import yaml

from models.registry import get_model, list_models

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_CONFIG = REPO_ROOT / "config" / "run_config.yml"
MODELS = ["gemma4-12b-unified-w4a16-vllm", "gemma4-31b-w4a16-vllm"]


def test_exactly_two_models_are_registered() -> None:
    assert list_models() == MODELS


@pytest.mark.parametrize("removed", ["internvl3-vllm", "qwen38-27b-fp8-vllm", "gemma4-12b-unified-vllm"])
def test_a_removed_model_is_refused_with_a_full_diagnostic(removed: str) -> None:
    with pytest.raises(ValueError) as raised:
        get_model(removed)

    message = str(raised.value)
    assert removed in message
    assert all(model in message for model in MODELS)
    assert all(part in message for part in ("What:", "Where:", "Expected:", "How to fix:"))
    assert "internvl" not in message.replace(removed, "")


def test_the_shipped_settings_name_a_registered_model() -> None:
    raw = yaml.safe_load(RUN_CONFIG.read_text())

    assert raw["bootstrap"]["model"]["type"] in MODELS
    assert sorted(raw["bootstrap"]["model"]["default_paths"]) == MODELS
    assert sorted(raw["inference"]["vllm"]["models"]) == MODELS
