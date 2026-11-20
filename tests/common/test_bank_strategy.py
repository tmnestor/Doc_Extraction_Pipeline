"""`pipeline.processing.bank_strategy` chooses how many model calls a bank statement gets.

``two_turn`` is the path every run to date has used: header detection, then a
layout-specific extraction. ``single_pass`` sends the schema prompt alone and
hands the reply to the clean stage as the model wrote it, so that a model read
in one call can be compared with another model read in one call with the same
prompt.

The statement content below is invented for these tests and appears in no
evaluation set.
"""

import json
import sys
import types
from pathlib import Path

import pytest
import yaml
from PIL import Image

from common.bank_types import BANK_STRATEGIES, validate_bank_strategy
from common.pipeline_config import load_yaml_config
from common.unified_bank_extractor import UnifiedBankExtractor, _default_catalog, serialise_bank_response

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_CONFIG = REPO_ROOT / "config" / "run_config.yml"

# Three dates, two descriptions, three amounts, and a statement period wider
# than the debit dates: everything the two-turn fallback would "repair".
UNEVEN_REPLY = (
    "DOCUMENT_TYPE: BANK_STATEMENT\n"
    "STATEMENT_DATE_RANGE: 01/02/2031 - 28/02/2031\n"
    "TRANSACTION_DATES: 04/02/2031 | 09/02/2031 | 17/02/2031\n"
    "LINE_ITEM_DESCRIPTIONS: EFTPOS QUILLFEATHER BOOKS | DIRECT DEBIT MARROWBY GYM\n"
    "TRANSACTION_AMOUNTS_PAID: $41.07 | $63.00 | $12.55\n"
)


def assert_diagnostic_error(message: str, *, where: str, example: str) -> None:
    """A config error must say what is wrong, where, what is valid, and the fix."""
    assert "What:" in message
    assert "Where:" in message and where in message
    assert "Expected:" in message and example in message
    assert "How to fix:" in message


class FakeModel:
    """Stands in for ``processor.generate``: records each call, replays replies."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, image, prompt, max_tokens=None, extra=None) -> str:
        self.prompts.append(prompt)
        return self.replies[len(self.prompts) - 1]


@pytest.fixture
def statement_png(tmp_path: Path) -> Path:
    path = tmp_path / "statement.png"
    Image.new("RGB", (8, 8), "white").save(path)
    return path


@pytest.fixture
def no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The two-turn path frees GPU memory through torch; give it a stand-in."""
    stub = types.ModuleType("torch")
    stub.cuda = types.SimpleNamespace(empty_cache=lambda: None)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", stub)


# --- the value ----------------------------------------------------------------


@pytest.mark.parametrize("value", BANK_STRATEGIES)
def test_each_named_strategy_is_accepted(value):
    assert validate_bank_strategy(value) == value


@pytest.mark.parametrize("value", ["one_shot", "", None, True, 2])
def test_anything_else_fails_with_a_full_diagnostic(value, tmp_path):
    config_path = tmp_path / "run_config.yml"
    with pytest.raises(ValueError) as err:
        validate_bank_strategy(value, config_path)
    message = str(err.value)
    assert_diagnostic_error(
        message,
        where=f"{config_path} -> pipeline.processing.bank_strategy",
        example="bank_strategy: two_turn",
    )
    assert "two_turn, single_pass" in message


# --- the run config -----------------------------------------------------------


def _config_with_processing(tmp_path: Path, **changes) -> Path:
    """A copy of the shipped run config with its processing block edited."""
    raw = yaml.safe_load(RUN_CONFIG.read_text())
    processing = raw["pipeline"]["processing"]
    for key, value in changes.items():
        if value is ...:
            processing.pop(key)
        else:
            processing[key] = value
    path = tmp_path / "run_config.yml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_shipped_config_states_single_pass():
    flat, raw = load_yaml_config(RUN_CONFIG)
    assert raw["pipeline"]["processing"]["bank_strategy"] == "single_pass"
    assert flat["bank_strategy"] == "single_pass"


def test_single_pass_is_read_from_the_config(tmp_path):
    flat, _ = load_yaml_config(_config_with_processing(tmp_path, bank_strategy="single_pass"))
    assert flat["bank_strategy"] == "single_pass"


def test_a_config_without_the_key_fails_before_any_work(tmp_path):
    path = _config_with_processing(tmp_path, bank_strategy=...)
    with pytest.raises(ValueError) as err:
        load_yaml_config(path)
    assert_diagnostic_error(
        str(err.value),
        where=f"{path} -> pipeline.processing.bank_strategy",
        example="bank_strategy:",
    )


def test_a_config_with_an_unknown_strategy_fails_before_any_work(tmp_path):
    path = _config_with_processing(tmp_path, bank_strategy="three_turn")
    with pytest.raises(ValueError) as err:
        load_yaml_config(path)
    message = str(err.value)
    assert "'three_turn'" in message
    assert_diagnostic_error(
        message,
        where=f"{path.resolve()} -> pipeline.processing.bank_strategy",
        example="bank_strategy: two_turn",
    )


# --- the extractor ------------------------------------------------------------


def test_the_extractor_cannot_be_built_without_a_strategy():
    with pytest.raises(TypeError, match="bank_strategy"):
        UnifiedBankExtractor(generate_fn=FakeModel(), verbose=False)  # type: ignore[call-arg]


def test_the_extractor_rejects_an_unknown_strategy():
    with pytest.raises(ValueError, match="pipeline.processing.bank_strategy"):
        UnifiedBankExtractor(generate_fn=FakeModel(), verbose=False, bank_strategy="one_shot")


def test_single_pass_makes_one_call_with_the_schema_prompt(statement_png):
    model = FakeModel(UNEVEN_REPLY)
    extractor = UnifiedBankExtractor(generate_fn=model, verbose=False, bank_strategy="single_pass")

    _, metadata = extractor.extract_bank_statement(statement_png)

    assert model.prompts == [_default_catalog().get_prompt("bank", "schema_fallback_extraction")]
    assert metadata["strategy_used"] == "single_pass"
    assert metadata["turns_executed"] == 1


def test_single_pass_keeps_the_reply_as_written(statement_png):
    extractor = UnifiedBankExtractor(
        generate_fn=FakeModel(UNEVEN_REPLY), verbose=False, bank_strategy="single_pass"
    )

    schema_fields, metadata = extractor.extract_bank_statement(statement_png)

    # Not cut to the shortest list, statement period not recomputed, no balance column.
    assert serialise_bank_response(schema_fields, metadata) == UNEVEN_REPLY
    assert schema_fields["STATEMENT_DATE_RANGE"] == "01/02/2031 - 28/02/2031"
    assert schema_fields["TRANSACTION_AMOUNTS_PAID"] == "$41.07 | $63.00 | $12.55"


def test_two_turn_still_detects_headers_first_and_repairs_the_fallback(statement_png, no_torch):
    # Headers the column matcher cannot place send the two-turn path to its
    # schema fallback, which receives the SAME reply as the single-pass test
    # above and alters it. This pins the difference between the two strategies.
    model = FakeModel("Monday\nTuesday\nWednesday", UNEVEN_REPLY)
    extractor = UnifiedBankExtractor(generate_fn=model, verbose=False, bank_strategy="two_turn")
    catalog = _default_catalog()

    schema_fields, metadata = extractor.extract_bank_statement(statement_png)

    assert model.prompts == [
        catalog.get_prompt("bank", "turn0_header_detection"),
        catalog.get_prompt("bank", "schema_fallback_extraction"),
    ]
    assert "verbatim_response" not in metadata
    assert schema_fields["TRANSACTION_DATES"] == "04/02/2031 | 09/02/2031"
    assert schema_fields["STATEMENT_DATE_RANGE"] == "04/02/2031 - 09/02/2031"
    written = serialise_bank_response(schema_fields, metadata)
    assert written != UNEVEN_REPLY
    assert "ACCOUNT_BALANCE: " in written


# --- through to the clean stage -----------------------------------------------


def test_the_clean_stage_reads_a_single_pass_reply(statement_png, tmp_path):
    from stages import clean

    extractor = UnifiedBankExtractor(
        generate_fn=FakeModel(UNEVEN_REPLY), verbose=False, bank_strategy="single_pass"
    )
    schema_fields, metadata = extractor.extract_bank_statement(statement_png)
    raw_path = tmp_path / "raw_extractions.jsonl"
    record = {
        "image_name": statement_png.name,
        "image_path": str(statement_png),
        "document_type": "BANK_STATEMENT",
        "raw_response": serialise_bank_response(schema_fields, metadata),
        "processing_time": 0.1,
        "prompt_used": f"unified_bank_{metadata['strategy_used']}",
        "error": None,
    }
    raw_path.write_text(json.dumps(record) + "\n")

    cleaned_path = clean.run(raw_path, tmp_path / "cleaned_extractions.jsonl")

    cleaned = json.loads(cleaned_path.read_text().splitlines()[0])["extracted_data"]
    assert cleaned["DOCUMENT_TYPE"] == "BANK_STATEMENT"
    # The cleaner's own normalisation drops the spaces; the period is the model's.
    assert cleaned["STATEMENT_DATE_RANGE"] == "01/02/2031-28/02/2031"
    assert cleaned["TRANSACTION_DATES"].count("|") == 2
    assert cleaned["LINE_ITEM_DESCRIPTIONS"].count("|") == 1
    assert cleaned["TRANSACTION_AMOUNTS_PAID"].count("|") == 2


# --- the call sites -----------------------------------------------------------


@pytest.mark.parametrize("module_path", ["stages/extract.py", "common/vllm_dp_workers.py"])
def test_every_call_site_passes_the_configured_strategy(module_path):
    source = (REPO_ROOT / module_path).read_text()
    assert source.count("UnifiedBankExtractor(") == 1
    assert "bank_strategy=config.bank_strategy" in source


@pytest.mark.parametrize("module_path", ["stages/extract.py", "common/vllm_dp_workers.py"])
def test_both_extract_paths_write_the_record_through_the_shared_serialiser(module_path):
    source = (REPO_ROOT / module_path).read_text()
    assert "serialise_bank_response(schema_fields, metadata)" in source
