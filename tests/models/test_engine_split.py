"""How one engine is spread over several cards is a setting: tensor (every layer split) or pipeline.

A model too large for one card runs as a single engine across the cards. It can be split two ways. Tensor
parallelism splits every layer, so the cards exchange results at every layer. Pipeline parallelism shares
the layers out in sequence, so they exchange once per pass. Which is faster on cards without a direct link
between them has to be measured, so the choice is a required key and not a constant in the loader.
"""

import ast
from pathlib import Path

import pytest
import yaml

from common.pipeline_config import ENGINE_SPLITS, load_yaml_config
from models import model_loader
from models.model_loader import engine_split_sizes

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_CONFIG = REPO_ROOT / "config" / "run_config.yml"


def config_with(tmp_path: Path, change) -> Path:
    raw = yaml.safe_load(RUN_CONFIG.read_text())
    change(raw["bootstrap"]["gpus"])
    path = tmp_path / "run_config.yml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_tensor_splits_every_layer_across_the_cards() -> None:
    assert engine_split_sizes(2, "tensor") == (2, 1)


def test_pipeline_shares_the_layers_out_in_sequence() -> None:
    assert engine_split_sizes(2, "pipeline") == (1, 2)


@pytest.mark.parametrize("split", ENGINE_SPLITS)
def test_on_one_card_there_is_nothing_to_split(split: str) -> None:
    assert engine_split_sizes(1, split) == (1, 1)


def test_an_unknown_split_is_refused_with_a_full_diagnostic() -> None:
    with pytest.raises(ValueError) as raised:
        engine_split_sizes(2, "both")

    message = str(raised.value)
    assert "'both'" in message
    assert all(part in message for part in ("What:", "Where:", "Expected:", "How to fix:"))
    assert "bootstrap.gpus.engine_split" in message


def test_the_shipped_config_states_the_split_and_it_is_the_one_measured() -> None:
    flat, raw = load_yaml_config(RUN_CONFIG)

    assert raw["bootstrap"]["gpus"]["engine_split"] == "tensor"
    assert flat["engine_split"] == "tensor"


def test_pipeline_is_read_from_the_config(tmp_path: Path) -> None:
    flat, _ = load_yaml_config(config_with(tmp_path, lambda gpus: gpus.update(engine_split="pipeline")))

    assert flat["engine_split"] == "pipeline"


def test_a_config_without_the_key_fails_before_any_work(tmp_path: Path) -> None:
    with pytest.raises(ValueError) as raised:
        load_yaml_config(config_with(tmp_path, lambda gpus: gpus.pop("engine_split")))

    message = str(raised.value)
    assert "engine_split" in message and "bootstrap.gpus" in message
    assert all(part in message for part in ("What:", "Where:", "Expected:", "How to fix:"))


def test_a_config_with_an_unknown_split_fails_before_any_work(tmp_path: Path) -> None:
    with pytest.raises(ValueError) as raised:
        load_yaml_config(config_with(tmp_path, lambda gpus: gpus.update(engine_split="ring")))

    message = str(raised.value)
    assert "'ring'" in message and "tensor, pipeline" in message
    assert all(part in message for part in ("What:", "Where:", "Expected:", "How to fix:"))


def test_the_loader_gives_the_engine_the_sizes_from_the_setting() -> None:
    """The fault this guards against is a setting that is read and then never reaches the engine."""
    source = Path(model_loader.__file__).read_text()

    assert "engine_split_sizes(" in source and "cfg.engine_split" in source
    assert '"tensor_parallel_size": tp_size' in source
    assert 'llm_kwargs["pipeline_parallel_size"] = pp_size' in source


def test_the_tensor_split_builds_the_engine_as_it_always_was() -> None:
    """The pipeline size is handed over only when it is more than one, so a run that does not ask for the
    pipeline split gives the engine no argument it was not given before."""
    tree = ast.parse(Path(model_loader.__file__).read_text())
    guarded = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and "pp_size > 1" in ast.unparse(node.test)
    ]

    assert len(guarded) == 1
    assert "pipeline_parallel_size" in ast.unparse(guarded[0])
    always = {
        key.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert "tensor_parallel_size" in always and "pipeline_parallel_size" not in always
