"""The run's settings: one object, read once from ``config/run_config.yml``.

    settings = Settings.load({"model_type": ...})

A stage that reads pages loads its settings here and hands them to whatever needs them. A run's choices
are the job's variables, handed to ``load`` (``common.stage_settings``). What comes from the file is
where each model's weights are.

The folders a task reads and writes are not here. They are worked out once, when the task starts
(``common.stage_settings``), and handed to the stage. Until 25 November 2026 this object worked the
folder of pages out a second time, and refused to load without one: so a task that read a folder of
its own, as the pipeline then had, could not start unless the pipeline's folder of pages existed.

Nothing is defaulted in Python. A key that is needed and absent stops the run with a message that says
what is wrong, where, what it should look like and how to mend it. A key the pipeline does not know is
refused too (``common.config_schema``).
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from common.config_schema import ConfigError, check_known_keys

DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "run_config.yml"
_ABSENT = object()


def read_settings_file(path: Path) -> dict[str, Any]:
    """The settings file as a mapping, with every key checked to be one the pipeline knows.

    The one place the file is read: the task's own settings and the run's settings both come from here.

    Raises:
        ConfigError: If the file does not exist, or holds a key the pipeline does not know.
    """
    if not path.is_file():
        raise ConfigError(
            f"What: the settings file does not exist.\n"
            f"Where: {path.resolve()}\n"
            f"Expected: the pipeline's settings file, config/run_config.yml in the repository.\n"
            f"How to fix: run the entry script from the repository's own folder."
        )
    raw = yaml.safe_load(path.read_text()) or {}
    check_known_keys(raw, config_path=path.resolve())
    return raw


def value_at(raw: Mapping[str, Any], key: str, absent: Any = _ABSENT) -> Any:
    """The value at a dotted key, or ``absent`` if any part of the path is missing."""
    value: Any = raw
    for part in key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return absent
        value = value[part]
    return value


def _need(raw: dict[str, Any], key: str, path: Path, example: str) -> Any:
    """The value at a dotted key, which must be there (it may be null where null means something)."""
    value = value_at(raw, key)
    if value is _ABSENT:
        raise ConfigError(
            f"What: the required key '{key}' is missing.\n"
            f"Where: {path} -> {key}\n"
            f"Expected: {example}\n"
            f"How to fix: add '{key}' to {path}. Where a null is allowed, write it out, so that what the "
            f"pipeline is set to do can be read from the file."
        )
    return value


class Settings:
    """The run's settings. Nothing in it is changed after loading."""

    config_path: Path
    raw: dict[str, Any]
    # What a task is started with.
    model_type: str
    model_path: Path
    # The share of each card's memory the job sets (DOC_EXTRACT_GPU_MEMORY_UTILIZATION). None: the
    # model's own, in its registration.
    gpu_memory_utilization: float | None
    # Which pages of the folder are read: the job's max_images and document_types. None: every page.
    max_images: int | None
    document_types: tuple[str, ...] | None
    # The file that what is sent to the model is kept in (DOC_EXTRACT_TRACE_FILE). None: not kept.
    trace_file: Path | None

    @classmethod
    def load(
        cls,
        overrides: dict[str, Any],
        *,
        config_path: Path | None = None,
        raw: dict[str, Any] | None = None,
    ) -> "Settings":
        """Read and check the settings.

        Args:
            overrides: What the job's variables give: ``model_type``, ``gpu_memory_utilization``,
                ``max_images``, ``document_types`` and ``trace_file``. One left out, or None, is not set.
            config_path: The settings file, or None for the shipped one.
            raw: What that file holds, when it has been read already (``read_settings_file``): a task
                is worked out from one reading of its settings file. None reads the file.

        Raises:
            ConfigError: If the settings file does not exist, holds a key the pipeline does not know,
                lacks a needed key, has a value that cannot be used, or has a per-model block
                under a name no model has.
        """
        path = config_path or DEFAULT_CONFIG
        if raw is None:
            raw = read_settings_file(path)
        given = {name: value for name, value in overrides.items() if value is not None}
        self = object.__new__(cls)
        self.config_path, self.raw = path, raw

        choices = ("gpu_memory_utilization", "max_images", "document_types", "trace_file")
        unknown = sorted(set(given) - {"model_type", *choices})
        if unknown:
            raise TypeError(f"Settings.load does not take {', '.join(unknown)}")
        for name in choices:
            setattr(self, name, given.get(name))

        model = "bootstrap.model"
        self.model_type = given.get("model_type") or _need(
            raw, f"{model}.type", path, "a registered model, e.g. type: gemma4-31b-w4a16-vllm"
        )
        self.model_path = cls._model_folder(raw, path, self.model_type)
        return self

    # -- how each part is read ------------------------------------------------------------------------

    @staticmethod
    def _model_folder(raw: dict[str, Any], path: Path, model_type: str) -> Path:
        """The folder of the model's weights: the one ``bootstrap.model.default_paths`` gives for it."""
        default_paths = value_at(raw, "bootstrap.model.default_paths")
        listed = default_paths.get(model_type) if isinstance(default_paths, dict) else None
        if not listed or not (Path(listed) / "config.json").exists():
            raise ConfigError(
                f"What: no folder was found for the model {model_type!r}"
                f"{f': {listed} does not hold a config.json' if listed else ''}.\n"
                f"Where: {path} -> bootstrap.model.default_paths.{model_type}\n"
                f"Expected: the folder that holds the model's config.json, e.g.\n"
                f"  bootstrap:\n    model:\n      default_paths:\n        {model_type}: /models/{model_type}\n"
                f"How to fix: give that key the model's folder."
            )
        return Path(listed)
