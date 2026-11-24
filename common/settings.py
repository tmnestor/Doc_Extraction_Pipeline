"""The run's settings: one object, read once from ``config/run_config.yml``.

    settings = Settings.load({"model_type": ..., "data_dir": ..., "output_dir": ...})

A stage that reads pages loads its settings here and hands them to whatever needs them. Three things a
task is started with (the model, the folder of pages, the output folder) may be given to ``load`` and
then stand in place of the file's. Everything else comes from the file.

Nothing is defaulted in Python. A key that is needed and absent stops the run with a message that says
what is wrong, where, what it should look like and how to mend it. A key the pipeline does not know is
refused too (``common.config_schema``).
"""

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from common.config_schema import ConfigError, check_known_keys
from common.vllm_stage_overrides import STAGES, apply_stage_overrides, validate_stage_overrides

DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "run_config.yml"
_SECONDARY_SORTS = ("none", "image_area_asc", "image_area_desc")
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


def _build_vllm_config(raw: dict[str, Any], path: Path) -> dict[str, dict]:
    """The engine settings of each model: ``vllm.defaults`` with the model's own laid over them."""
    section = raw.get("inference", {}).get("vllm", {})
    defaults = section.get("defaults")
    if not isinstance(defaults, dict):
        raise ValueError(
            "What: the settings file has no inference.vllm.defaults block.\n"
            f"Where: {path}, key inference.vllm.defaults.\n"
            "Expected: a block such as\n"
            "  inference:\n"
            "    vllm:\n"
            "      defaults:\n"
            "        gpu_memory_utilization: 0.90\n"
            "        max_model_len: 8192\n"
            "        max_num_seqs: 1\n"
            "        limit_mm_per_prompt: 1\n"
            "        enable_prefix_caching: true\n"
            f"How to fix: add that block to {path}."
        )
    models = section.get("models", {})
    _check_model_names(models, path)
    validate_stage_overrides(models, config_path=str(path))
    config: dict[str, dict] = {"__defaults__": dict(defaults)}
    for model_name, own in models.items():
        config[model_name] = {**defaults, **own}
    return config


def _check_model_names(vllm_models: dict, path: Path) -> None:
    """Refuse a per-model block under a name no registered model has.

    Such a block is never looked up: the model gets the defaults, and the block reads as if it were in
    force.
    """
    from models.registry import list_models

    registered = list_models()
    unknown = sorted(set(vllm_models) - set(registered))
    if unknown:
        raise ConfigError(
            f"What: inference.vllm.models has block(s) for no registered model: {', '.join(unknown)}. "
            f"Such a block is never read, so its settings are not applied.\n"
            f"Where: {path} -> inference.vllm.models.{unknown[0]}\n"
            f"Expected: a block under a registered model's full name, one of: {', '.join(registered)}.\n"
            f"How to fix: correct the name, or delete the block if the model should have the defaults "
            f"above it."
        )


_TOKEN_BUDGETS = ("classify", "extract_base", "extract_per_field")


class Settings:
    """The run's settings. Nothing in it is changed after loading; ``for_stage`` gives a copy."""

    config_path: Path
    raw: dict[str, Any]
    # What a task is started with.
    model_type: str
    model_path: Path
    data_dir: Path
    output_dir: Path
    # Which pages of the folder are read.
    max_images: int | None
    document_types: list[str] | None
    # How the engine is started.
    enforce_eager: bool
    chat_template: str | None
    num_gpus: int
    data_parallel_size: int | None
    # Whether each page's type is printed as it is classified, and whether what is sent to the model is kept.
    verbose: bool
    trace_raw_prompts: bool
    trace_path: str | None
    # The order pages are read in, and the labels that are not read.
    extraction_order: list[str]
    secondary_sort: str
    extraction_skip_labels: list[str]
    classification_fallback_type: str

    _vllm_config: dict[str, dict]
    _token_budgets: dict[str, int]
    _infrastructure: dict[str, int | float]
    _stage: str | None

    @classmethod
    def load(cls, overrides: dict[str, Any], *, config_path: Path | None = None) -> "Settings":
        """Read and check the settings.

        Args:
            overrides: Any of ``model_type``, ``data_dir``, ``output_dir`` and ``num_gpus``, to stand
                in place of the file's. A value of None is taken as not given.
            config_path: The settings file, or None for the shipped one.

        Raises:
            ConfigError: If the settings file does not exist, holds a key the pipeline does not know,
                lacks a needed key, has a value that cannot be used, or has a per-model block
                under a name no model has.
        """
        path = config_path or DEFAULT_CONFIG
        raw = read_settings_file(path)
        given = {name: value for name, value in overrides.items() if value is not None}
        self = object.__new__(cls)
        self.config_path, self.raw, self._stage = path, raw, None

        folders = "pipeline.information_extraction"
        data_dir = given.get("data_dir") or value_at(raw, f"{folders}.input.dir")
        output_dir = given.get("output_dir") or value_at(raw, f"{folders}.output.dir")
        if data_dir in (None, _ABSENT) or output_dir in (None, _ABSENT):
            missing = "image_dir" if data_dir in (None, _ABSENT) else "output"
            key = f"{folders}.input.dir" if missing == "image_dir" else f"{folders}.output.dir"
            raise ConfigError(
                f"What: the {'folder of pages' if missing == 'image_dir' else 'output folder'} is not set.\n"
                f"Where: the variable {missing} in the job's environment, or {path} -> {key}\n"
                f"Expected: a folder, e.g. {missing}=/data/run_20261121\n"
                f"How to fix: set {missing} in the job's environment, or give {key} a value."
            )
        self.data_dir, self.output_dir = Path(data_dir), Path(output_dir)
        if not self.data_dir.exists():
            raise ConfigError(
                f"What: the folder of pages does not exist: {self.data_dir}\n"
                f"Where: the variable image_dir in the job's environment, or {path} -> {folders}.input.dir\n"
                f"Expected: a folder that holds the pages to read.\n"
                f"How to fix: correct the folder, or create it and put the pages in it."
            )
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
        self.enforce_eager = _need(raw, f"{model}.enforce_eager", path, "true or false")
        self.chat_template = cls._chat_template(raw, path)

        gpus = "bootstrap.gpus"
        num_gpus = given.get("num_gpus")
        self.num_gpus = (
            num_gpus
            if num_gpus is not None
            else _need(raw, f"{gpus}.num_gpus", path, "0 for every card, or a number")
        )
        if not isinstance(self.num_gpus, int) or self.num_gpus < 0:
            raise ConfigError(
                f"What: the number of cards is {self.num_gpus!r}.\n"
                f"Where: {path} -> {gpus}.num_gpus\n"
                f"Expected: 0 (every card), or a whole number of cards, e.g. num_gpus: 2\n"
                f"How to fix: set {gpus}.num_gpus to 0 or a positive whole number."
            )
        self.data_parallel_size = _need(
            raw, f"{gpus}.data_parallel_size", path, "null for one engine a card, or a number of engines"
        )

        self.verbose = _need(raw, "pipeline.processing.verbose", path, "true or false")
        self.trace_raw_prompts, self.trace_path = cls._tracing(raw, path)

        self._vllm_config = _build_vllm_config(raw, path)
        self._token_budgets = cls._budgets(raw, path)
        self._infrastructure = dict(raw.get("resources", {}).get("infrastructure", {}))
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
        self.secondary_sort = _need(
            raw, "pipeline.extraction.secondary_sort", path, f"one of {list(_SECONDARY_SORTS)}"
        )
        if self.secondary_sort not in _SECONDARY_SORTS:
            raise ConfigError(
                f"What: pipeline.extraction.secondary_sort is {self.secondary_sort!r}.\n"
                f"Where: {path} -> pipeline.extraction.secondary_sort\n"
                f"Expected: one of {list(_SECONDARY_SORTS)}, e.g. secondary_sort: none\n"
                f"How to fix: set it to one of those values."
            )
        return self

    # -- how each part is read ------------------------------------------------------------------------

    @staticmethod
    def _model_folder(raw: dict[str, Any], path: Path, model_type: str) -> Path:
        """The folder of the model's weights: ``bootstrap.model.path`` for the file's own model, else
        the folder ``bootstrap.model.default_paths`` gives for the model asked for.

        A path set for one model is never used for another, and no other model's folder is tried.
        """
        explicit = _need(raw, "bootstrap.model.path", path, "null, to use default_paths")
        if explicit and model_type == value_at(raw, "bootstrap.model.type"):
            folder = Path(explicit)
            if not folder.exists():
                raise ConfigError(
                    f"What: the model folder does not exist: {folder}\n"
                    f"Where: {path} -> bootstrap.model.path\n"
                    f"Expected: the folder that holds the model's config.json, or null to use default_paths.\n"
                    f"How to fix: correct bootstrap.model.path, or set it to null."
                )
            return folder
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
    def _chat_template(raw: dict[str, Any], path: Path) -> str | None:
        """A chat template to use in place of the model's own, or None for the model's own."""
        value = _need(raw, "bootstrap.model.chat_template", path, "none, or the path of a *.jinja file")
        if _is_none(value):
            return None
        template = Path(str(value)).expanduser()
        if not template.is_file():
            raise ConfigError(
                f"What: 'bootstrap.model.chat_template' points to a file that does not exist: {value}\n"
                f"Where: {path} -> bootstrap.model.chat_template\n"
                "Expected: 'none' (use the model's template) or a path to an existing *.jinja file.\n"
                f"How to fix: correct the path, or set 'chat_template: none' in {path}."
            )
        return str(template)

    @staticmethod
    def _tracing(raw: dict[str, Any], path: Path) -> tuple[bool, str | None]:
        """Whether what is sent to the model is kept, and where. No ``inference.tracing`` block is off."""
        tracing = raw.get("inference", {}).get("tracing")
        if tracing is None:
            return False, None
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
    def _budgets(raw: dict[str, Any], path: Path) -> dict[str, int]:
        """The completion budgets, each a positive whole number. Checked here, before any engine starts."""
        budgets: dict[str, int] = {}
        for name in _TOKEN_BUDGETS:
            key = f"pipeline.token_budgets.{name}"
            value = _need(raw, key, path, f"a whole number of tokens, e.g. {name}: 512")
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ConfigError(
                    f"What: {key} is {value!r}, not a positive whole number.\n"
                    f"Where: {path} -> {key}\n"
                    f"Expected: the most tokens a reply may run to, e.g. {name}: 512\n"
                    f"How to fix: set {key} to a positive whole number."
                )
            budgets[name] = value
        return budgets

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

    # -- what the rest of the pipeline asks -------------------------------------------------------------

    def for_stage(self, stage: str) -> "Settings":
        """A copy whose engine settings are the ones ``stage`` loads its engine with.

        A model may state settings one stage loads differently, in
        ``inference.vllm.models.<model>.stage_overrides``. Everything else is shared with this object.

        Raises:
            ValueError: If ``stage`` is not a stage that takes overrides.
        """
        if stage not in STAGES:
            msg = f"Unknown stage {stage!r} for vLLM stage overrides. Known stages: {', '.join(STAGES)}"
            raise ValueError(msg)
        staged = copy.copy(self)
        staged._stage = stage
        return staged

    def get_vllm_config(self, model_type: str) -> dict[str, Any]:
        """The engine settings of a model: its own block over ``inference.vllm.defaults``.

        On a copy made by ``for_stage``, the model's overrides for that stage are applied.
        """
        defaults = self._vllm_config["__defaults__"]
        own = self._vllm_config.get(model_type, {})
        return apply_stage_overrides({**defaults, **own}, self._stage) if own else dict(defaults)

    def get_token_budget(self, name: str) -> int:
        """A completion budget from ``pipeline.token_budgets``."""
        return self._named(self._token_budgets, name, "pipeline.token_budgets", "classify: 150")

    def get_infra(self, name: str) -> int | float:
        """A setting from ``resources.infrastructure``."""
        return self._named(self._infrastructure, name, "resources.infrastructure", "dp_join_timeout: 60")

    def _named(self, block: dict[str, Any], name: str, section: str, example: str) -> Any:
        if name not in block:
            raise ConfigError(
                f"What: {section}.{name} is not set.\n"
                f"Where: {self.config_path} -> {section}.{name}\n"
                f"Expected: a number, e.g. {example}. Set there now: {', '.join(sorted(block)) or 'nothing'}.\n"
                f"How to fix: add {name} under {section}."
            )
        return block[name]
