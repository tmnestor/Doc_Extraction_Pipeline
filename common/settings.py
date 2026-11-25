"""The run's settings: one object, read once from ``config/run_config.yml``.

    settings = Settings.load({"model_type": ..., "output_dir": ...})

A stage that reads pages loads its settings here and hands them to whatever needs them. The model a task
is started with may be given to ``load`` and then stands in place of the file's. Everything else comes
from the file.

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


def _is_none(value: Any) -> bool:
    return value is None or str(value).strip().lower() in {"none", "null", ""}


class Settings:
    """The run's settings. Nothing in it is changed after loading."""

    config_path: Path
    raw: dict[str, Any]
    # What a task is started with.
    model_type: str
    model_path: Path
    # The folder the stage writes to, when it gave one: only to place the record of what was sent to
    # the model, when that is kept and no place is named for it.
    output_dir: Path | None
    # The share of each card's memory the job sets (DOC_EXTRACT_GPU_MEMORY_UTILIZATION), checked when
    # the task was worked out. None: the model's own, in its registration.
    gpu_memory_utilization: float | None
    # Which pages of the folder are read.
    max_images: int | None
    document_types: list[str] | None
    # Whether what is sent to the model is kept, and where.
    trace_raw_prompts: bool
    trace_path: str | None
    # The order pages are read in, and the labels that are not read.
    extraction_order: list[str]
    extraction_skip_labels: list[str]
    classification_fallback_type: str

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
            overrides: ``model_type``, to stand in place of the file's; ``output_dir``, the folder the stage writes to; and ``gpu_memory_utilization``, the
                share of each card's memory the job sets. A value of None is taken as not given.
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

        unknown = sorted(set(given) - {"model_type", "output_dir", "gpu_memory_utilization"})
        if unknown:
            raise TypeError(f"Settings.load does not take {', '.join(unknown)}")
        folders = "pipeline.information_extraction"
        self.output_dir = Path(given["output_dir"]) if "output_dir" in given else None
        self.gpu_memory_utilization = given.get("gpu_memory_utilization")
        self.max_images = _need(
            raw, f"{folders}.input.max_images", path, "null for every page, or a number"
        )
        self.document_types = _need(
            raw, f"{folders}.input.document_types", path, "null for every type, or a list such as [RECEIPT]"
        )

        model = "bootstrap.model"
        self.model_type = given.get("model_type") or _need(
            raw, f"{model}.type", path, "a registered model, e.g. type: gemma4-31b-w4a16-vllm"
        )
        self.model_path = cls._model_folder(raw, path, self.model_type)

        self.trace_raw_prompts, self.trace_path = cls._tracing(raw, path)

        self.classification_fallback_type = _need(
            raw,
            "pipeline.classification.fallback_type",
            path,
            "a document type, e.g. fallback_type: UNIVERSAL",
        )
        self.extraction_order = cls._list_of_text(
            raw, path, "pipeline.extraction.order", "order:\n  - bank_statement\n  - invoice\n  - receipt"
        )
        self.extraction_skip_labels = cls._list_of_text(
            raw, path, "pipeline.extraction.skip_labels", "skip_labels: []", empty_if_null=True
        )
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

    @staticmethod
    def _tracing(raw: dict[str, Any], path: Path) -> tuple[bool, str | None]:
        """Whether what is sent to the model is kept, and where. The block must be there: one left out
        is not taken for 'off'. Until 26 November 2026 it was."""
        tracing = raw.get("inference", {}).get("tracing")
        enabled = tracing.get("raw_prompts") if isinstance(tracing, dict) else None
        if not isinstance(enabled, bool) or "path" not in tracing:
            raise ConfigError(
                "What: the 'tracing' block needs 'raw_prompts' (true or false) and 'path'.\n"
                f"Where: {path} -> inference.tracing\n"
                "Expected:\n  inference:\n    tracing:\n      raw_prompts: false\n      path: none\n"
                "How to fix: give both keys under 'inference.tracing'."
            )
        return enabled, None if _is_none(tracing["path"]) else str(tracing["path"])

    @staticmethod
    def _list_of_text(
        raw: dict[str, Any], path: Path, key: str, example: str, *, empty_if_null: bool = False
    ) -> list[str]:
        value = _need(raw, key, path, f"a list of text, e.g.\n{example}")
        if value is None and empty_if_null:
            return []
        if not isinstance(value, list) or not all(isinstance(entry, str) for entry in value):
            raise ConfigError(
                f"What: {key} is {value!r}, not a list of text.\n"
                f"Where: {path} -> {key}\n"
                f"Expected: a list of text, e.g.\n{example}\n"
                f"How to fix: write {key} as a YAML list of names."
            )
        return list(value)
