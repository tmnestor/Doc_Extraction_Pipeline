"""A task is checked whole before it writes anything, and its settings are checked whole when loaded.

Until 26 November 2026 the runner wrote the note of what made an output folder, and a stage started
afresh deleted its earlier output, before the stage had read its own settings. So:

- a task that could not start left a note in a folder that held none of its output, and the next start
  with another model was refused as "output from another run";
- a task started afresh that could not start had already deleted the output it was to replace.

And two settings were first met after that: a key the engine is started with (a ``KeyError`` as the
engine loaded), and ``resources.infrastructure.dp_join_timeout`` (only with an engine on each card). A
tracing block left out was taken for "off". The share of a card's memory a job may set was read by
``float()`` as the engine loaded: ``entrypoint.sh`` now checks it at launch.

The pages are blank images made by the bench, and every reply is invented.
"""

import importlib
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from bench import AN_ENGINE_ON_EACH_CARD, ONE_ENGINE, Bench
from common import classification, run_manifest
from common.config_schema import ConfigError
from common.settings import Settings
from common.stage_settings import TASKS, resolve
from stages import __main__ as starter

PARTS = ("What:", "Where:", "Expected:", "How to fix:")


@pytest.fixture(autouse=True)
def the_classification_file_is_read_afresh() -> Any:
    classification._settings.cache_clear()
    yield
    classification._settings.cache_clear()


def task(bench: Bench, name: str, out: Path, model: str = ONE_ENGINE, **more: str) -> Any:
    environ = {
        "model": model,
        "image_dir": str(bench.pages),
        "output": str(out),
        "DOC_EXTRACT_LOG_DIR": str(out.parent / "logs"),
        "ground_truth": str(out.parent / "ground_truth.jsonl"),
        **more,
    }
    return resolve(name, environ, bench.config)


def everything_under(folder: Path) -> list[str]:
    return sorted(str(path.relative_to(folder)) for path in folder.rglob("*")) if folder.is_dir() else []


def a_setting_that_cannot_be_used(bench: Bench) -> None:
    bench.with_settings(lambda settings: settings["pipeline"]["batching"].update(classify=0, default=0))


def mended(bench: Bench) -> None:
    bench.with_settings(lambda settings: settings["pipeline"]["batching"].update(classify=2, default=2))


# --- nothing is written by a task that cannot start -----------------------------------------------------


@pytest.mark.parametrize("name", ["classify", "extract"])
def test_a_task_that_cannot_start_leaves_nothing_in_its_output_folder(
    bench: Bench, tmp_path: Path, name: str
) -> None:
    out = tmp_path / "out"
    if name == "extract":
        starter.run_task(task(bench, "classify", out))
    before = everything_under(out)
    noted = run_manifest.read(out)
    a_setting_that_cannot_be_used(bench)

    with pytest.raises(ConfigError, match="pipeline.batching"):
        starter.run_task(task(bench, name, out))

    assert everything_under(out) == before
    assert run_manifest.read(out) == noted


def test_another_model_may_start_in_the_folder_of_a_task_that_never_started(
    bench: Bench, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    a_setting_that_cannot_be_used(bench)
    with pytest.raises(ConfigError):
        starter.run_task(task(bench, "classify", out, model=AN_ENGINE_ON_EACH_CARD))
    mended(bench)

    starter.run_task(task(bench, "classify", out, model=ONE_ENGINE))

    assert run_manifest.read(out)["classify"].model == ONE_ENGINE
    assert len((out / "classifications.jsonl").read_text().splitlines()) == len(bench.names)


def test_a_task_started_afresh_that_cannot_start_keeps_the_output_it_was_to_replace(
    bench: Bench, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    starter.run_task(task(bench, "classify", out))
    kept = (out / "classifications.jsonl").read_bytes()
    a_setting_that_cannot_be_used(bench)

    with pytest.raises(ConfigError):
        starter.run_task(task(bench, "classify", out, CLEAR_PREV_OUTPUT="true"))

    assert (out / "classifications.jsonl").read_bytes() == kept


def test_the_runner_checks_a_task_before_it_notes_it_and_starts_it(
    bench: Bench, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    stage = importlib.import_module("stages.classify")
    noting = starter._check_what_made_the_folder

    def checked(task: Any) -> None:
        order.append("check")

    def noted(task: Any) -> None:
        order.append("note")
        noting(task)

    def started(task: Any, checked: Any) -> str:
        order.append("start")
        return "done"

    monkeypatch.setattr(stage, "check", checked)
    monkeypatch.setattr(stage, "start", started)
    monkeypatch.setattr(starter, "_check_what_made_the_folder", noted)

    starter.run_task(task(bench, "classify", tmp_path / "out"))

    assert order == ["check", "note", "start"]


@pytest.mark.parametrize("name", TASKS)
def test_checking_a_task_writes_nothing(bench: Bench, tmp_path: Path, name: str) -> None:
    out = tmp_path / "out"
    for earlier in TASKS[1 : max(TASKS.index(name), 1)]:
        starter.run_task(task(bench, earlier, out))
    bench.answer_key(tmp_path / "ground_truth.jsonl")
    before = {path: path.read_bytes() for path in out.rglob("*") if path.is_file()} if out.is_dir() else {}
    engines = bench.engines_started

    importlib.import_module(f"stages.{name}").check(task(bench, name, out))

    after = {path: path.read_bytes() for path in out.rglob("*") if path.is_file()} if out.is_dir() else {}
    assert after == before
    assert bench.engines_started == engines


# --- the settings are checked whole when they are loaded ------------------------------------------------


def loaded_with(bench: Bench, change: Any, model: str = ONE_ENGINE) -> Settings:
    bench.with_settings(change)
    return Settings.load({"model_type": model}, config_path=bench.config)


@pytest.mark.parametrize(
    "key",
    [
        "gpu_memory_utilization",
        "max_model_len",
        "max_num_seqs",
        "limit_mm_per_prompt",
        "enable_prefix_caching",
    ],
)
def test_a_key_the_engine_is_started_with_is_missed_when_the_settings_are_loaded(
    bench: Bench, key: str
) -> None:
    def left_out(settings: dict[str, Any]) -> None:
        del settings["inference"]["vllm"]["defaults"][key]

    with pytest.raises(ConfigError) as refused:
        loaded_with(bench, left_out)

    message = str(refused.value)
    assert all(part in message for part in PARTS)
    assert f"inference.vllm.defaults.{key}" in message and str(bench.config) in message


@pytest.mark.parametrize("model", [ONE_ENGINE, AN_ENGINE_ON_EACH_CARD])
def test_every_key_the_engine_loader_reads_is_there_for_every_model(bench: Bench, model: str) -> None:
    loader = (Path(starter.REPOSITORY) / "models" / "model_loader.py").read_text()
    read = set(re.findall(r'tuning\["([a-z_]+)"\]', loader))
    settings = Settings.load({"model_type": model}, config_path=bench.config)

    assert read and read <= set(settings.get_vllm_config(model))
    assert read <= set(settings.get_vllm_config("a model with no block of its own"))


@pytest.mark.parametrize("model", [ONE_ENGINE, AN_ENGINE_ON_EACH_CARD])
def test_the_wait_for_a_worker_is_missed_when_the_settings_are_loaded_whichever_the_model(
    bench: Bench, model: str
) -> None:
    def left_out(settings: dict[str, Any]) -> None:
        del settings["resources"]

    with pytest.raises(ConfigError) as refused:
        loaded_with(bench, left_out, model)

    message = str(refused.value)
    assert all(part in message for part in PARTS)
    assert "resources.infrastructure.dp_join_timeout" in message


@pytest.mark.parametrize("value", [True, "sixty", -1])
def test_a_wait_for_a_worker_that_is_no_number_of_seconds_is_refused(bench: Bench, value: Any) -> None:
    def spoilt(settings: dict[str, Any]) -> None:
        settings["resources"]["infrastructure"]["dp_join_timeout"] = value

    with pytest.raises(ConfigError, match="dp_join_timeout"):
        loaded_with(bench, spoilt)


def test_a_tracing_block_left_out_is_not_taken_for_off(bench: Bench) -> None:
    def left_out(settings: dict[str, Any]) -> None:
        del settings["inference"]["tracing"]

    with pytest.raises(ConfigError) as refused:
        loaded_with(bench, left_out)

    message = str(refused.value)
    assert all(part in message for part in PARTS) and "inference.tracing" in message


# --- the settings file is read once to work a task out --------------------------------------------------


def test_working_a_task_out_reads_the_settings_file_once(
    bench: Bench, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    read: list[Path] = []
    real = yaml.safe_load

    def counted(text: str) -> Any:
        parsed = real(text)
        # The schema file is YAML too, and is read the first time it is asked for: only the settings
        # file has a bootstrap block.
        if isinstance(parsed, dict) and "bootstrap" in parsed:
            read.append(bench.config)
        return parsed

    monkeypatch.setattr("common.settings.yaml.safe_load", counted)

    task(bench, "clean", tmp_path / "out")

    assert len(read) == 1


# --- a task is worked out once: what its check built is what it runs with ------------------------------


@pytest.mark.parametrize("name", ["split", "classify", "extract"])
def test_one_start_of_a_task_reads_its_settings_file_once_and_loads_its_settings_once(
    bench: Bench, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """Until 26 November 2026 the file was read to work the task out, and again by the stage; and for
    a day the stage's settings were built by its check and built again by its run."""
    out = tmp_path / "out"
    if name == "extract":
        starter.run_task(task(bench, "classify", out))
    read: list[str] = []
    real_read = yaml.safe_load

    def counted(text: str) -> Any:
        parsed = real_read(text)
        if isinstance(parsed, dict) and "bootstrap" in parsed:
            read.append("read")
        return parsed

    loaded: list[str] = []
    real_load = Settings.load.__func__  # type: ignore[attr-defined]

    def counted_load(cls: Any, *args: Any, **kwargs: Any) -> Settings:
        loaded.append("loaded")
        return real_load(cls, *args, **kwargs)

    monkeypatch.setattr("common.settings.yaml.safe_load", counted)
    monkeypatch.setattr(Settings, "load", classmethod(counted_load))

    starter.run_task(task(bench, name, out))

    assert read == ["read"]
    assert loaded == ["loaded"]


@pytest.mark.parametrize("name", ["split", "classify", "extract"])
def test_what_a_stages_check_gives_is_what_its_run_is_handed(
    bench: Bench, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    out = tmp_path / "out"
    if name == "extract":
        starter.run_task(task(bench, "classify", out))
    stage = importlib.import_module(f"stages.{name}")
    handed: list[Any] = []
    given: list[Any] = []
    real_check = stage.check

    def check(the_task: Any) -> Any:
        given.append(real_check(the_task))
        return given[-1]

    def run(*args: Any, **kwargs: Any) -> Path:
        handed.extend(args)
        return out

    monkeypatch.setattr(stage, "check", check)
    monkeypatch.setattr(stage, "run", run)

    starter.run_task(task(bench, name, out))

    assert any(argument is given[0] for argument in handed)


def test_the_share_of_memory_a_job_sets_reaches_the_engine_with_the_settings(
    bench: Bench, tmp_path: Path
) -> None:
    from common.stage_settings import run_settings

    with_a_share = task(bench, "classify", tmp_path / "out", DOC_EXTRACT_GPU_MEMORY_UTILIZATION="0.85")
    without = task(bench, "classify", tmp_path / "out")

    assert run_settings(with_a_share).gpu_memory_utilization == 0.85
    assert run_settings(without).gpu_memory_utilization is None
    # A copy made for a worker's process, or for a stage, carries it.
    assert run_settings(with_a_share).on_one_card().for_stage("classify").gpu_memory_utilization == 0.85


def test_the_engine_loader_reads_no_variable_of_the_job_itself() -> None:
    loader = (Path(starter.REPOSITORY) / "models" / "model_loader.py").read_text()

    assert "os.environ.get" not in loader and "os.getenv" not in loader
    assert "settings.gpu_memory_utilization" in loader
