"""What a task is run with: which model, which folders, and whether to start afresh.

A task is started by ``entrypoint.sh`` with ``KFP_TASK`` naming it. Each of its settings is taken from the
environment if it is set there, and otherwise from the settings file. A setting the task needs that is in
neither stops the run before any work, with a message that says what is missing, where to put it, what it
should look like and how to mend it.

A variable that is unset, empty, or the word ``None`` counts as not set: a pipeline manifest passes a blank
parameter as the text "None".

Run as a module it prints the log folder, which the entry script needs before it starts anything else:

    python -m common.stage_settings config/run_config.yml
"""

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.settings import ConfigError, read_settings_file, value_at

TASKS = ("split", "classify", "extract", "clean", "evaluate")

# The tasks that read pages from the image folder.
_READS_IMAGES = ("split", "classify", "extract")
# The prefix the pipeline's own variables had until 21 November 2026.
_OLD_PREFIX = "L" + "MM_"

_MODEL = ("model", "bootstrap.model.type")
_IMAGE_DIR = ("image_dir", "pipeline.information_extraction.input.dir")
_GROUND_TRUTH = ("ground_truth", "pipeline.information_extraction.input.ground_truth")
_OUTPUT = ("output", "pipeline.information_extraction.output.dir")
_LOG_DIR = ("DOC_EXTRACT_LOG_DIR", "bootstrap.logging.log_dir")
_DATASETS = "pipeline.datasets"
_DATASET_KEYS = ("document_type", "image_dir")


@dataclass(frozen=True)
class StageSettings:
    """Everything a task is run with."""

    task: str
    model: str
    image_dir: Path | None
    output: Path
    ground_truth: Path | None
    log_dir: Path
    clear_previous_output: bool
    # The settings file these were worked out from. A stage that reads more of it reads this one.
    config_path: Path
    # The dataset the job names, if it names one, and the document type that dataset gives its pages.
    dataset: str | None = None
    given_type: str | None = None

    # Every file a stage writes is under the output folder, so that one stage finds what another wrote.

    @property
    def split_dir(self) -> Path:
        return self.output / "split"

    @property
    def classifications(self) -> Path:
        return self.output / "classifications.jsonl"

    @property
    def raw_extractions(self) -> Path:
        return self.output / "raw_extractions.jsonl"

    @property
    def cleaned_extractions(self) -> Path:
        return self.output / "cleaned_extractions.jsonl"

    @property
    def evaluation_dir(self) -> Path:
        return self.output / "evaluation"


def _set(environ: Mapping[str, str], name: str) -> str | None:
    """The variable's value, or None if it is unset, empty or the word None."""
    value = environ.get(name)
    return None if value is None or value == "" or value == "None" else value


def _from_file(raw: Mapping[str, Any], key: str) -> str | None:
    value = value_at(raw, key, None)
    return None if value is None or value == "" else str(value)


def _value(
    environ: Mapping[str, str],
    raw: Mapping[str, Any],
    config_path: Path,
    source: tuple[str, str],
    *,
    needed_by: str | None,
    example: str,
) -> str | None:
    """A setting from the environment, else from the settings file; refused if needed and in neither."""
    variable, key = source
    value = _set(environ, variable) or _from_file(raw, key)
    if value is None and needed_by is not None:
        raise ConfigError(
            f"What: the task {needed_by!r} needs {variable}, and it is set nowhere.\n"
            f"Where: the environment variable {variable}, or the key {key} in {config_path.resolve()}.\n"
            f"Expected: {example}\n"
            f"How to fix: set {variable} in the job's environment, or give {key} a value in that file."
        )
    return value


def _datasets(raw: Mapping[str, Any], config_path: Path) -> dict[str, tuple[str, str | None]]:
    """Every dataset of the settings file: its document type, and its folder of pages if it names one.

    All of them are checked, whichever the job names: a fault in one is found before it is needed.

    Raises:
        ConfigError: If the block is absent, a dataset is not its two keys, or its type is not a
            document type of the schema file.
    """
    from common.field_schema import get_field_schema

    where = config_path.resolve()
    example = (
        "  pipeline:\n    datasets:\n      receipts_of_a_shop:\n"
        "        document_type: RECEIPT\n        image_dir: null"
    )
    block = value_at(raw, _DATASETS, None)
    if not isinstance(block, Mapping):
        raise ConfigError(
            f"What: the settings file has no '{_DATASETS}' block.\n"
            f"Where: {where} -> {_DATASETS}\n"
            f"Expected: the datasets whose pages are all of one type, e.g.\n{example}\n"
            f"  or, with none, datasets: {{}}\n"
            f"How to fix: add the block to {where}. Write 'datasets: {{}}' if there is no such dataset."
        )
    types = get_field_schema().document_types
    found: dict[str, tuple[str, str | None]] = {}
    for name, entry in block.items():
        key = f"{_DATASETS}.{name}"
        if not isinstance(entry, Mapping) or set(entry) != set(_DATASET_KEYS):
            has = sorted(entry) if isinstance(entry, Mapping) else repr(entry)
            raise ConfigError(
                f"What: the dataset '{name}' is not its two keys, {' and '.join(_DATASET_KEYS)}: it has {has}.\n"
                f"Where: {where} -> {key}\n"
                f"Expected: both keys and no other, e.g.\n{example}\n"
                f"How to fix: give '{name}' a document_type, and an image_dir that is a folder or null."
            )
        kind, folder = entry["document_type"], entry["image_dir"]
        if not isinstance(kind, str) or kind.lower() not in types:
            raise ConfigError(
                f"What: the dataset '{name}' gives its pages the type {kind!r}, which is not a document type.\n"
                f"Where: {where} -> {key}.document_type\n"
                f"Expected: a document type of config/field_definitions.yaml, in either case: "
                f"{', '.join(sorted(types))}.\n"
                f"How to fix: name one of those, or add the type's block to the schema file and a prompt "
                f"of the same name to the prompt file."
            )
        if folder is not None and not isinstance(folder, str):
            raise ConfigError(
                f"What: the dataset '{name}' has an image_dir that is neither a folder nor null: {folder!r}.\n"
                f"Where: {where} -> {key}.image_dir\n"
                f"Expected: the folder that holds its pages, e.g. image_dir: /data/receipts, or null to "
                f"leave the job to name it.\n"
                f"How to fix: write the folder as text, or null."
            )
        found[str(name)] = (kind, folder or None)
    return found


def _refuse_old_names(environ: Mapping[str, str]) -> None:
    old = sorted(name for name in environ if name.startswith(_OLD_PREFIX))
    if old:
        raise ConfigError(
            f"What: variable(s) set under a prefix this pipeline no longer reads: {', '.join(old)}.\n"
            f"Where: the environment of this job (the pipeline manifest, or the command that started it).\n"
            f"Expected: DOC_EXTRACT_LOG_DIR, DOC_EXTRACT_CONDA_ENV, DOC_EXTRACT_GPU_MEMORY_UTILIZATION.\n"
            f"How to fix: rename each variable listed, keeping its value, and start the job again."
        )


def log_dir(environ: Mapping[str, str], config_path: Path) -> Path:
    """The folder the run's log is written to."""
    _refuse_old_names(environ)
    found = _value(
        environ,
        read_settings_file(config_path),
        config_path,
        _LOG_DIR,
        needed_by="every task",
        example="a folder that outlasts the job, e.g. DOC_EXTRACT_LOG_DIR=/data/run_20261121/logs",
    )
    assert found is not None
    return Path(found)


def resolve(task: str, environ: Mapping[str, str], config_path: Path) -> StageSettings:
    """Work out what a task is run with.

    Args:
        task: The task's name, as ``KFP_TASK`` gives it.
        environ: The job's environment.
        config_path: The settings file.

    Returns:
        The task's settings.

    Raises:
        ConfigError: If the task is not one of ``TASKS``, a setting it needs is set nowhere, or a
            value cannot be read.
    """
    _refuse_old_names(environ)
    if task not in TASKS:
        raise ConfigError(
            f"What: KFP_TASK is {task!r}, which is not a task of this pipeline.\n"
            f"Where: the environment variable KFP_TASK.\n"
            f"Expected: one of {', '.join(TASKS)}.\n"
            f"How to fix: set KFP_TASK to one of those, e.g. KFP_TASK=classify model=<a registered model> "
            f"image_dir=<folder> output=<folder> bash entrypoint.sh"
        )
    raw = read_settings_file(config_path)

    def value(source: tuple[str, str], needed: bool, example: str) -> str | None:
        return _value(
            environ, raw, config_path, source, needed_by=task if needed else None, example=example
        )

    model = value(_MODEL, True, "a registered model, e.g. model=gemma4-31b-w4a16-vllm")

    # A dataset the job names gives its pages their type, and their folder unless the job names one.
    datasets = _datasets(raw, config_path)
    dataset = _set(environ, "dataset")
    given_type: str | None = None
    of_the_dataset: str | None = None
    if dataset is not None:
        if dataset not in datasets:
            raise ConfigError(
                f"What: the job names the dataset {dataset!r}, which the settings file does not have.\n"
                f"Where: the environment variable dataset, and the block {_DATASETS} in "
                f"{config_path.resolve()}.\n"
                f"Expected: one of {', '.join(sorted(datasets)) or '(the file has none)'}, or no dataset "
                f"at all for pages whose types the model is to be asked.\n"
                f"How to fix: correct the name, add the dataset to {_DATASETS}, or leave dataset unset."
            )
        given_type, of_the_dataset = datasets[dataset]
    image_dir = _set(environ, _IMAGE_DIR[0]) or of_the_dataset
    if image_dir is None:
        image_dir = value(
            _IMAGE_DIR, task in _READS_IMAGES, "the folder of pages to read, e.g. image_dir=/data/pages"
        )
    output = value(_OUTPUT, True, "the folder the stages write to, e.g. output=/data/run_20261121")
    ground_truth = value(
        _GROUND_TRUTH,
        task == "evaluate",
        "the answer key, e.g. ground_truth=/data/pages/ground_truth.jsonl",
    )
    assert model is not None and output is not None
    if task == "evaluate" and ground_truth is not None and Path(ground_truth).suffix != ".jsonl":
        variable, key = _GROUND_TRUTH
        raise ConfigError(
            f"What: the answer key {ground_truth!r} is not a JSON lines file. That is the one format read.\n"
            f"Where: the environment variable {variable}, or the key {key} in {config_path.resolve()}.\n"
            f"Expected: a file ending .jsonl with one answer a line, each holding the page's file name "
            f"and its own document type's fields, e.g. {variable}=/data/pages/ground_truth.jsonl\n"
            f"How to fix: name the set's ground_truth.jsonl, or write the key out as JSON lines."
        )

    return StageSettings(
        task=task,
        model=model,
        image_dir=Path(image_dir) if image_dir else None,
        output=Path(output),
        ground_truth=Path(ground_truth) if ground_truth else None,
        log_dir=log_dir(environ, config_path),
        clear_previous_output=_start_afresh(environ),
        config_path=config_path,
        dataset=dataset,
        given_type=given_type,
    )


def check_folder_of_pages(settings: StageSettings, config_path: Path) -> None:
    """Refuse to start a task that reads pages when its folder of pages is not there.

    Only the tasks that read pages are asked: the others are given no such folder, and need none.

    Raises:
        ConfigError: If the task reads pages and its folder of pages does not exist.
    """
    if settings.task not in _READS_IMAGES:
        return
    if settings.image_dir is None or not settings.image_dir.is_dir():
        variable, key = _IMAGE_DIR
        raise ConfigError(
            f"What: the folder of pages does not exist: {settings.image_dir}\n"
            f"Where: the environment variable {variable}, or the key {key} in {config_path.resolve()}.\n"
            f"Expected: a folder that holds the pages to read, e.g. {variable}=/data/pages\n"
            f"How to fix: correct the folder, or create it and put the pages in it."
        )


def _start_afresh(environ: Mapping[str, str]) -> bool:
    """Whether CLEAR_PREV_OUTPUT asks for the stage's earlier output to be deleted first.

    A manifest may pass a YAML true as the text "True", so case is ignored.
    """
    given = (_set(environ, "CLEAR_PREV_OUTPUT") or "false").lower()
    if given not in ("true", "false"):
        raise ConfigError(
            f"What: CLEAR_PREV_OUTPUT is {given!r}.\n"
            f"Where: the environment variable CLEAR_PREV_OUTPUT.\n"
            f"Expected: true (delete this stage's earlier output and start again) or false (carry on from "
            f"where an earlier run stopped). Unset means false.\n"
            f"How to fix: set CLEAR_PREV_OUTPUT=true or CLEAR_PREV_OUTPUT=false, or leave it unset."
        )
    return given == "true"


def main() -> int:
    """Print the log folder, or the reason it cannot be worked out."""
    if len(sys.argv) != 2:
        print("usage: python -m common.stage_settings <settings file>")
        return 2
    try:
        print(log_dir(os.environ, Path(sys.argv[1])))
    except ConfigError as err:
        print(f"FATAL: the log folder cannot be worked out.\n{err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
