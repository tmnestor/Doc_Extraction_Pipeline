"""Engine settings that one pipeline stage loads differently from the rest.

Each GPU stage runs in its own process and loads its own engine, so a stage can
be given different engine settings without touching the others. The case this
exists for is the image token budget: classifying a page needs less visual
detail than reading every figure off it, and the engine's time on a short reply
is mostly the time it spends reading the image.

A model states its overrides in ``inference.vllm.models.<model>.stage_overrides``
of ``config/run_config.yml``. An override may only change a setting the model's
own block already states, so a misspelt key is an error instead of a new setting
the engine silently ignores.
"""

import copy
from pathlib import Path
from typing import Any

from common.config_schema import ConfigError

STAGE_OVERRIDES_KEY = "stage_overrides"

# Stages that load their engine through Settings.for_stage(). A stage listed
# in the YAML but not here would read as configured while doing nothing.
STAGES: tuple[str, ...] = ("classify",)

# Blocks of a model's vLLM settings a stage may override.
OVERRIDABLE: tuple[str, ...] = ("mm_processor_kwargs", "hf_overrides")

_SECTION = "inference.vllm.models"

_EXAMPLE = """\
inference:
  vllm:
    models:
      gemma4-31b-w4a16-vllm:
        mm_processor_kwargs:
          max_soft_tokens: 560
        hf_overrides:
          vision_soft_tokens_per_image: 560
          vision_config:
            default_output_length: 560
        stage_overrides:
          classify:
            mm_processor_kwargs:
              max_soft_tokens: 280
            hf_overrides:
              vision_soft_tokens_per_image: 280
              vision_config:
                default_output_length: 280"""


class StageOverrideConfigError(ConfigError):
    """A stage_overrides block is malformed."""


def _fail(what: str, *, model: str, key_path: str, config_path: Path | str) -> StageOverrideConfigError:
    where = f"{Path(config_path).resolve()} -> {_SECTION}.{model}.{STAGE_OVERRIDES_KEY}{key_path}"
    return StageOverrideConfigError(
        f"What: {what}\n"
        f"  Where: {where}\n"
        f"  Expected: a mapping of stage ({', '.join(STAGES)}) to the blocks it loads differently "
        f"({', '.join(OVERRIDABLE)}), each key already stated in the model's own block, e.g.\n{_EXAMPLE}\n"
        f"  How to fix: correct the entry at the path above, or delete the {STAGE_OVERRIDES_KEY} "
        f"block so every stage loads the model's own settings."
    )


def _check_leaves(override: Any, base: Any, *, model: str, key_path: str, config_path: Path | str) -> None:
    """Every key an override sets must already be set, to the same kind of value, in the model's block."""
    if not isinstance(override, dict):
        if isinstance(base, dict):
            raise _fail(
                f"the override is a single value ({override!r}) where the model's own block has a mapping.",
                model=model,
                key_path=key_path,
                config_path=config_path,
            )
        return
    if not isinstance(base, dict):
        raise _fail(
            "the override is a mapping where the model's own block has a single value or nothing.",
            model=model,
            key_path=key_path,
            config_path=config_path,
        )
    for key, value in override.items():
        if key not in base:
            raise _fail(
                f"the override sets {key!r}, which the model's own block does not state "
                f"(it states: {', '.join(sorted(map(str, base))) or 'nothing'}).",
                model=model,
                key_path=f"{key_path}.{key}",
                config_path=config_path,
            )
        _check_leaves(value, base[key], model=model, key_path=f"{key_path}.{key}", config_path=config_path)


def validate_stage_overrides(vllm_models: dict[str, Any], *, config_path: Path | str) -> None:
    """Check every model's stage_overrides block, before any engine is loaded.

    Args:
        vllm_models: The ``inference.vllm.models`` mapping as read from the YAML.
        config_path: The file it was read from, for the error message.

    Raises:
        StageOverrideConfigError: With what is wrong, where, a valid example and the fix.
    """
    for model, settings in vllm_models.items():
        if not isinstance(settings, dict) or STAGE_OVERRIDES_KEY not in settings:
            continue
        block = settings[STAGE_OVERRIDES_KEY]
        if not isinstance(block, dict):
            raise _fail(
                f"{STAGE_OVERRIDES_KEY} is {type(block).__name__}, not a mapping.",
                model=model,
                key_path="",
                config_path=config_path,
            )
        for stage, overrides in block.items():
            if stage not in STAGES:
                raise _fail(
                    f"{stage!r} is not a stage that takes overrides.",
                    model=model,
                    key_path=f".{stage}",
                    config_path=config_path,
                )
            if not isinstance(overrides, dict) or not overrides:
                raise _fail(
                    f"the {stage} entry is empty or not a mapping.",
                    model=model,
                    key_path=f".{stage}",
                    config_path=config_path,
                )
            for name, override in overrides.items():
                if name not in OVERRIDABLE:
                    raise _fail(
                        f"{name!r} is not a block a stage may override.",
                        model=model,
                        key_path=f".{stage}.{name}",
                        config_path=config_path,
                    )
                _check_leaves(
                    override,
                    settings.get(name),
                    model=model,
                    key_path=f".{stage}.{name}",
                    config_path=config_path,
                )


def _merged(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        return {key: _merged(base[key], override[key]) if key in override else base[key] for key in base}
    return override


def apply_stage_overrides(model_config: dict[str, Any], stage: str | None) -> dict[str, Any]:
    """Return a model's vLLM settings as *stage* loads them.

    The stage_overrides block itself is never part of the result. With no stage,
    or a stage the model states no override for, the model's own settings are
    returned unchanged.

    Args:
        model_config: One model's resolved vLLM settings.
        stage: The stage loading the engine, or None for the model's own settings.

    Returns:
        A new mapping; *model_config* is not modified.
    """
    resolved = {
        key: copy.deepcopy(value) for key, value in model_config.items() if key != STAGE_OVERRIDES_KEY
    }
    overrides = (model_config.get(STAGE_OVERRIDES_KEY) or {}).get(stage, {}) if stage else {}
    for name, override in overrides.items():
        resolved[name] = _merged(resolved[name], copy.deepcopy(override))
    return resolved
