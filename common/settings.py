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
from pathlib import Path
from typing import Any

import yaml

from common.config_schema import check_known_keys
from common.vllm_stage_overrides import STAGES, apply_stage_overrides, validate_stage_overrides

DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "run_config.yml"
_SECONDARY_SORTS = ("none", "image_area_asc", "image_area_desc")
_ABSENT = object()


class ConfigError(ValueError):
    """The settings cannot be used. The message says what, where, what is expected and the fix."""


def read_settings_file(path: Path) -> dict[str, Any]:
    """The settings file as a mapping, with every key checked to be one the pipeline knows.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If it holds a key the pipeline does not know.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text())
    check_known_keys(raw, config_path=path.resolve())
    return raw


def _at(raw: dict[str, Any], key: str) -> Any:
    """The value at a dotted key, or ``_ABSENT`` if any part of the path is missing."""
    value: Any = raw
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return _ABSENT
        value = value[part]
    return value


def _need(raw: dict[str, Any], key: str, path: Path, example: str) -> Any:
    """The value at a dotted key, which must be there (it may be null where null means something)."""
    value = _at(raw, key)
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


def _build_generation_registry(raw_config: dict, config_file: str) -> dict[str, dict]:
    """The generation settings of each model: ``generation.defaults`` with the model's own laid over them.

    Raises:
        ValueError: If ``inference.generation.defaults`` is missing. Nothing is assumed in its place.
    """
    gen = raw_config.get("inference", {}).get("generation", {})
    if not isinstance(gen.get("defaults"), dict):
        raise ValueError(
            "What: the settings file has no inference.generation.defaults block.\n"
            f"Where: {config_file}, key inference.generation.defaults.\n"
            "Expected: a block such as\n"
            "  inference:\n"
            "    generation:\n"
            "      defaults:\n"
            "        max_new_tokens_base: 512\n"
            "        max_new_tokens_per_field: 64\n"
            "        temperature: 0.0\n"
            "        do_sample: false\n"
            "        top_p: 0.95\n"
            "        use_cache: true\n"
            f"How to fix: add that block to {config_file}."
        )
    defaults = dict(gen["defaults"])
    registry: dict[str, dict] = {
        model_name: {**defaults, **overrides} for model_name, overrides in (gen.get("models") or {}).items()
    }
    # What a model with no block of its own gets.
    registry["__defaults__"] = defaults
    return registry


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
    _check_model_names(models, raw.get("inference", {}).get("generation", {}).get("models", {}))
    validate_stage_overrides(models, config_path=str(path))
    config: dict[str, dict] = {"__defaults__": dict(defaults)}
    for model_name, own in models.items():
        config[model_name] = {**defaults, **own}
    return config


def normalised_model_type(model_type: str) -> str:
    """A model's name without its deployment suffixes, as ``inference.generation.models`` is keyed.

    ``"gemma4-31b-w4a16-vllm"`` -> ``"gemma4-31b"``,
    ``"gemma4-12b-unified-w4a16-vllm"`` -> ``"gemma4-12b-unified"``.
    """
    key = model_type.lower()
    if key.endswith("-vllm"):
        key = key[: -len("-vllm")]
    parts = key.rsplit("-", 1)
    if len(parts) == 2 and parts[1].endswith("b") and parts[1][:-1].isdigit():
        key = parts[0]
    if key.endswith("-w4a16"):
        key = key[: -len("-w4a16")]
    return key


def _check_model_names(vllm_models: dict, generation_models: dict) -> None:
    """Refuse a per-model block under a name no registered model has.

    Such a block is never looked up: the model gets the defaults, and the block reads as if it were in
    force. The two sections are keyed differently, which is how the mistake is made:
    ``inference.vllm.models`` by the full registered name, ``inference.generation.models`` by the name
    without its deployment suffixes.
    """
    from models.registry import list_models

    registered = list_models()
    normalised = {normalised_model_type(name) for name in registered}
    unknown_vllm = sorted(set(vllm_models) - set(registered))
    unknown_generation = sorted(set(generation_models) - normalised)
    if not unknown_vllm and not unknown_generation:
        return
    problems = []
    if unknown_vllm:
        problems.append(f"inference.vllm.models: {unknown_vllm}")
    if unknown_generation:
        problems.append(f"inference.generation.models: {unknown_generation}")
    raise ValueError(
        f"What:  per-model override key(s) match no registered model, so the "
        f"block is dead config -- the lookup silently falls through to the "
        f"`defaults` block and the tuning is never applied. "
        f"Offending: {'; '.join(problems)}\n"
        f"  Where: config/run_config.yml -> {' and '.join(p.split(':')[0] for p in problems)}\n"
        f"  Expected: inference.vllm.models is keyed by the FULL registered "
        f"type, one of: {', '.join(sorted(registered))}. "
        f"inference.generation.models is keyed by the NORMALISED type, one of: "
        f"{', '.join(sorted(normalised))}. The two conventions differ and the "
        f"normalisation is not uniform -- check it rather than inferring it.\n"
        f"  How to fix: correct the key to a listed value, or delete the block "
        f"if the model should inherit the `defaults` above it."
    )


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
    device_map: str
    num_gpus: int
    data_parallel_size: int | None
    # What is printed, and whether what is sent to the model is kept.
    verbose: bool
    debug: bool
    trace_raw_prompts: bool
    trace_path: str | None
    # The order pages are read in, and the labels that are not read.
    extraction_order: list[str]
    secondary_sort: str
    extraction_skip_labels: list[str]
    classification_fallback_type: str

    _generation_registry: dict[str, dict]
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
            ConfigError: If a needed key is missing or a value cannot be used.
            FileNotFoundError: If the settings file does not exist.
            ValueError: If the file holds a key the pipeline does not know, or a per-model block
                under a name no model has.
        """
        path = config_path or DEFAULT_CONFIG
        raw = read_settings_file(path)
        given = {name: value for name, value in overrides.items() if value is not None}
        self = object.__new__(cls)
        self.config_path, self.raw, self._stage = path, raw, None

        folders = "pipeline.information_extraction"
        data_dir = given.get("data_dir") or _at(raw, f"{folders}.input.dir")
        output_dir = given.get("output_dir") or _at(raw, f"{folders}.output.dir")
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
        self.device_map = str(_need(raw, f"{model}.device_map", path, "auto"))
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
        self.debug = _need(raw, "pipeline.processing.debug", path, "true or false")
        self.trace_raw_prompts, self.trace_path = cls._tracing(raw, path)

        self._generation_registry = _build_generation_registry(raw, str(path))
        self._vllm_config = _build_vllm_config(raw, path)
        self._token_budgets = dict(raw.get("pipeline", {}).get("token_budgets", {}))
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
        if explicit and model_type == _at(raw, "bootstrap.model.type"):
            folder = Path(explicit)
            if not folder.exists():
                raise ConfigError(
                    f"What: the model folder does not exist: {folder}\n"
                    f"Where: {path} -> bootstrap.model.path\n"
                    f"Expected: the folder that holds the model's config.json, or null to use default_paths.\n"
                    f"How to fix: correct bootstrap.model.path, or set it to null."
                )
            return folder
        default_paths = _at(raw, "bootstrap.model.default_paths")
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

    def get_generation_config(self, model_type: str) -> dict[str, Any]:
        """The generation settings of a model: its own block, else ``inference.generation.defaults``."""
        key = normalised_model_type(model_type)
        return dict(self._generation_registry.get(key) or self._generation_registry["__defaults__"])

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
