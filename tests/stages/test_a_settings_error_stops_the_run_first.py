"""A setting that cannot be used stops a run before any engine starts, and is never taken for a bad page.

Until 25 November 2026 three things were first read after an engine had started: the pages to an engine
call, the classification file (its questions, its column roles and its rules), and the schema's block for
a page's document type. The last two were read inside the engine call, where every failure is taken for
a failed call and tried again one page at a time. So one rule that could not be used gave every page the
fallback type with an error, asked the model about every page twice, and ended with exit code 0.

Now each stage checks what it reads before an engine is started, and a setting that cannot be used, met
anywhere, ends the run with its reason.

The pages are blank images made by the bench, and every reply is invented.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from bench import AN_ENGINE_ON_EACH_CARD, ONE_ENGINE, Bench
from common import classification
from common.config_schema import ConfigError
from common.pipeline_ops import check_prompts
from common.prompt_catalog import PromptCatalog
from common.stage_settings import resolve
from stages import __main__ as starter
from stages.io import read_jsonl

PARTS = ("What:", "Where:", "Expected:", "How to fix:")


@pytest.fixture(autouse=True)
def the_classification_file_is_read_afresh() -> Any:
    """The questions and rules are read once and kept: a test that spoils them must not leave them so."""
    classification._settings.cache_clear()
    yield
    classification._settings.cache_clear()


def a_rule_that_cannot_be_used(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(self: PromptCatalog) -> dict[str, Any]:
        raise ConfigError(
            "What: the rule's type 'RECIEPT' is not a document type.\nWhere: the classification file.\n"
            "Expected: a document type.\nHow to fix: correct the spelling."
        )

    monkeypatch.setattr(PromptCatalog, "get_classification_evidence", refuse)


def classified(bench: Bench, folder: Path, types: dict[str, str] | None = None) -> Path:
    """A classifications file for the bench's pages, each given its own type unless ``types`` says."""
    records = []
    for name in bench.names:
        kind = next(k for k in ("bank_statement", "invoice", "receipt") if k in name).upper()
        records.append(
            {
                "image_name": name,
                "image_path": str(bench.pages / name),
                "document_type": (types or {}).get(name, kind),
            }
        )
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "classifications.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


# --- before any engine starts ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", [ONE_ENGINE, AN_ENGINE_ON_EACH_CARD])
def test_a_rule_that_cannot_be_used_stops_classify_before_an_engine_starts(
    bench: Bench, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    a_rule_that_cannot_be_used(monkeypatch)
    output = tmp_path / "out" / "classifications.jsonl"

    with pytest.raises(ConfigError, match="RECIEPT"):
        bench.classify(output, model)

    assert bench.engines_started == 0
    assert bench.processor.attempts == []
    assert not output.exists()


def test_the_shipped_fallback_type_is_a_document_type(bench: Bench, tmp_path: Path) -> None:
    bench.classify(tmp_path / "out" / "classifications.jsonl")

    assert bench.engines_started == 1


def test_a_page_given_a_type_with_no_block_stops_extract_before_an_engine_starts(
    bench: Bench, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    classifications = classified(bench, out, {bench.names[0]: "PAYSLIP", bench.names[3]: "PAYSLIP"})

    with pytest.raises(ConfigError) as refused:
        bench.extract(classifications, out / "raw_extractions.jsonl")

    message = str(refused.value)
    assert all(part in message for part in PARTS)
    assert "PAYSLIP" in message and "config/field_definitions.yaml" in message
    assert bench.engines_started == 0
    assert not (out / "raw_extractions.jsonl").exists()


def test_a_prompt_file_that_cannot_be_read_is_a_settings_error_and_not_an_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(self: PromptCatalog, model_type: str) -> None:
        raise FileNotFoundError("Prompt file not found: prompts/extraction_prompts.yaml")

    monkeypatch.setattr(PromptCatalog, "check_extraction_prompts", missing)

    with pytest.raises(ConfigError) as refused:
        check_prompts(ONE_ENGINE)

    assert all(part in str(refused.value) for part in PARTS)
    assert "extraction_prompts.yaml" in str(refused.value)


def test_a_missing_prompt_stops_a_stage_before_an_engine_starts(
    bench: Bench, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(self: PromptCatalog, model_type: str) -> None:
        raise ConfigError("What: document type(s) with no prompt: receipt.")

    monkeypatch.setattr(PromptCatalog, "check_extraction_prompts", refuse)

    with pytest.raises(ConfigError, match="no prompt"):
        bench.classify(tmp_path / "out" / "classifications.jsonl", AN_ENGINE_ON_EACH_CARD)

    assert bench.engines_started == 0


# --- met inside an engine call --------------------------------------------------------------------------


@pytest.mark.parametrize("model", [ONE_ENGINE, AN_ENGINE_ON_EACH_CARD])
def test_a_settings_error_met_while_reading_ends_the_run_and_is_not_taken_for_a_failed_page(
    bench: Bench, tmp_path: Path, model: str
) -> None:
    """Whatever the checks before the start miss must still end the run, with its own reason."""

    def refuse(image_paths: list[str], document_type: str) -> list[str]:
        bench.processor.attempts.append([Path(p).name for p in image_paths])
        raise ConfigError("What: the document type has no block in the schema file.")

    bench.processor.read_pages = refuse  # type: ignore[method-assign]
    out = tmp_path / "out"

    with pytest.raises(ConfigError, match="no block in the schema file"):
        bench.extract(classified(bench, out), out / "raw_extractions.jsonl", model)

    # Each engine made one call and stopped: no page was tried again on its own.
    asked = [name for call in bench.processor.attempts for name in call]
    assert len(asked) == len(set(asked))
    assert len(bench.processor.attempts) <= 2
    assert not (out / "raw_extractions.jsonl").exists()


def test_a_reply_that_cannot_be_read_is_its_own_pages_and_is_not_asked_for_again(
    bench: Bench, tmp_path: Path
) -> None:
    spoilt = bench.names[2]
    read = bench.processor.classified

    def classified_unless_spoilt(reply: str) -> dict[str, Any]:
        if spoilt in reply:
            raise RuntimeError("a pattern met a reply it was not written for")
        return read(reply)

    bench.processor.classified = classified_unless_spoilt  # type: ignore[method-assign]
    output = tmp_path / "out" / "classifications.jsonl"

    bench.classify(output)

    records = {record["image_name"]: record for record in read_jsonl(output)}
    assert len(records) == len(bench.names)
    assert records[spoilt]["document_type"] == "UNIVERSAL"
    assert "a pattern met a reply" in records[spoilt]["error"]
    # The model's reply is kept, and the model was asked about each page once.
    assert records[spoilt]["raw_response"] == f"a reply about {spoilt}"
    assert sorted(bench.processor.asked_about) == sorted(bench.names)
    assert all("error" not in record for name, record in records.items() if name != spoilt)


def test_a_settings_error_met_while_a_reply_is_read_ends_the_run(bench: Bench, tmp_path: Path) -> None:
    def refuse(reply: str) -> dict[str, Any]:
        raise ConfigError("What: a rule names a column role that does not exist.")

    bench.processor.classified = refuse  # type: ignore[method-assign]

    with pytest.raises(ConfigError, match="column role"):
        bench.classify(tmp_path / "out" / "classifications.jsonl")

    # One call was made, and it was not made again.
    assert len(bench.processor.attempts) == 1


# --- as a job sees it -----------------------------------------------------------------------------------


def test_a_job_whose_rule_cannot_be_used_ends_with_its_reason_and_exit_code_one(
    bench: Bench, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    a_rule_that_cannot_be_used(monkeypatch)
    environ = {
        "model": AN_ENGINE_ON_EACH_CARD,
        "image_dir": str(bench.pages),
        "output": str(tmp_path / "out"),
        "DOC_EXTRACT_LOG_DIR": str(tmp_path / "logs"),
    }
    monkeypatch.setattr(starter, "CONFIG_PATH", bench.config)
    monkeypatch.setattr("common.settings.DEFAULT_CONFIG", bench.config)
    monkeypatch.setattr(starter.os, "environ", {**environ, "KFP_TASK": "classify"})
    assert resolve("classify", environ, bench.config).model == AN_ENGINE_ON_EACH_CARD

    code = starter.main()

    printed = capsys.readouterr().out
    assert code == 1
    assert "FATAL: the task cannot be started." in printed and "RECIEPT" in printed
    assert bench.engines_started == 0
    assert not (tmp_path / "out" / "classifications.jsonl").exists()
