"""A document type is a block in the schema file and a prompt of the same name, and nothing in Python.

These tests add a type that the pipeline has never heard of, by writing a schema file and a prompt file,
and check that it is read with its own prompt, given its own budget and scored by its own table rules.
The type, its fields and its values are invented.
"""

import re
import subprocess
from pathlib import Path

import pytest
import yaml
from PIL import Image

from common import field_schema, scoring
from common.field_schema import FieldSchema, FieldSchemaError
from common.prompt_catalog import PromptCatalog
from models.orchestrator import EXTRACT_BASE_TOKENS, EXTRACT_TOKENS_PER_FIELD, DocumentOrchestrator

REPO = Path(__file__).resolve().parents[2]
MODEL = "gemma4-31b-w4a16-vllm"
PROMPT = "Read this parking permit.\n\nDOCUMENT_TYPE: PARKING_PERMIT\nBAY_DATES: NOT_FOUND\nBAY_FEES: NOT_FOUND\n"


def with_a_new_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **block) -> PromptCatalog:
    """The shipped schema and prompts, with a parking permit added to each, put in place of the shipped."""
    schema = yaml.safe_load((REPO / "config" / "field_definitions.yaml").read_text())
    schema["document_types"]["parking_permit"] = {
        "fields": ["DOCUMENT_TYPE", "BAY_DATES", "BAY_FEES"],
        **block,
    }
    schema["fields"]["BAY_DATES"] = {
        "cleaned_as": "day_month_year",
        "list": True,
        "scored_as": "date",
        "scored_row_by_row": True,
    }
    schema["fields"]["BAY_FEES"] = {
        "cleaned_as": "money",
        "list": True,
        "scored_as": "money",
        "scored_row_by_row": True,
    }
    schema_path = tmp_path / "field_definitions.yaml"
    schema_path.write_text(yaml.safe_dump(schema, sort_keys=False))
    made = FieldSchema.from_yaml(schema_path)
    monkeypatch.setattr(field_schema, "get_field_schema", lambda: made)
    monkeypatch.setattr(scoring, "get_field_schema", lambda: made)
    monkeypatch.setattr("models.orchestrator.get_field_schema", lambda: made)

    prompts = tmp_path / "prompts"
    prompts.mkdir()
    for source in (REPO / "prompts").glob("*.yaml"):
        (prompts / source.name).write_text(source.read_text())
    extraction = yaml.safe_load((prompts / "extraction_prompts.yaml").read_text())
    extraction["prompts"]["parking_permit"] = {"prompt": PROMPT}
    (prompts / "extraction_prompts.yaml").write_text(yaml.safe_dump(extraction, sort_keys=False))
    catalog = PromptCatalog(prompts_dir=prompts)
    monkeypatch.setattr("models.orchestrator.PromptCatalog", lambda: catalog)
    return catalog


class Backend:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def chat(self, images, prompts, max_tokens) -> list[str]:
        self.calls.append({"prompts": prompts, "max_tokens": max_tokens})
        return ["a reply"] * len(images)


def processor(backend: Backend) -> DocumentOrchestrator:
    orch = object.__new__(DocumentOrchestrator)
    orch._backend = backend  # type: ignore[assignment]
    orch._model_type_key = MODEL
    return orch


@pytest.fixture
def page(tmp_path: Path) -> str:
    path = tmp_path / "page.png"
    Image.new("RGB", (40, 60), "white").save(path)
    return str(path)


def test_a_new_type_is_read_with_its_own_prompt_and_a_budget_by_its_fields(tmp_path, monkeypatch, page):
    with_a_new_type(tmp_path, monkeypatch)
    backend = Backend()

    processor(backend).read_pages([page], "PARKING_PERMIT")

    assert backend.calls == [
        {"prompts": [PROMPT], "max_tokens": EXTRACT_BASE_TOKENS + 3 * EXTRACT_TOKENS_PER_FIELD}
    ]


def test_a_new_type_may_fix_its_own_budget(tmp_path, monkeypatch, page):
    with_a_new_type(tmp_path, monkeypatch, completion_tokens=2048)
    backend = Backend()

    processor(backend).read_pages([page], "parking_permit")

    assert backend.calls[0]["max_tokens"] == 2048


def test_a_new_type_may_have_a_table_whose_rows_are_lined_up_before_scoring(tmp_path, monkeypatch):
    table = {
        "columns": ["BAY_DATES", "BAY_FEES"],
        "rows_absent_where_not_found_in": "BAY_FEES",
        "rows_in_date_order_of": "BAY_DATES",
    }
    with_a_new_type(tmp_path, monkeypatch, table=table)
    # The answer holds three rows, one of them marked absent; the reply gives the other two, latest first.
    answer = {
        "filename": "p.png",
        "BAY_DATES": "03/02/2031 | 04/02/2031 | 09/02/2031",
        "BAY_FEES": "$4.10 | NOT_FOUND | $6.30",
    }
    read = {"BAY_DATES": "09/02/2031 | 03/02/2031", "BAY_FEES": "$6.30 | $4.10"}

    lined_up = scoring.score_page(read, answer, "PARKING_PERMIT")
    as_they_lie = scoring.score_page(read, answer, "RECEIPT")

    assert lined_up["overall_accuracy"] == 1.0
    assert as_they_lie["overall_accuracy"] < 1.0


def test_a_type_without_a_block_is_refused_by_name(tmp_path, monkeypatch, page):
    with_a_new_type(tmp_path, monkeypatch)

    with pytest.raises(ValueError) as refused:
        processor(Backend()).read_pages([page], "FISHING_LICENCE")

    message = str(refused.value)
    assert all(part in message for part in ("What:", "Where:", "Expected:", "How to fix:"))
    assert "FISHING_LICENCE" in message and "document_types.fishing_licence" in message


def test_the_schema_and_the_prompt_file_must_agree_on_the_types(tmp_path, monkeypatch):
    catalog = with_a_new_type(tmp_path, monkeypatch)
    catalog.check_extraction_prompts(MODEL)

    path = catalog._dir / "extraction_prompts.yaml"
    prompts = yaml.safe_load(path.read_text())
    del prompts["prompts"]["parking_permit"]
    prompts["prompts"]["parking_permti"] = {"prompt": PROMPT}
    path.write_text(yaml.safe_dump(prompts, sort_keys=False))

    with pytest.raises(ValueError) as refused:
        PromptCatalog(prompts_dir=catalog._dir).check_extraction_prompts(MODEL)
    message = str(refused.value)
    assert all(part in message for part in ("What:", "Where:", "Expected:", "How to fix:"))
    assert "prompt(s) for no document type: parking_permti" in message
    assert "document type(s) with no prompt: parking_permit" in message


def test_every_shipped_type_has_a_prompt_and_every_prompt_a_type():
    PromptCatalog().check_extraction_prompts(MODEL)


@pytest.mark.parametrize(
    ("block", "said"),
    [
        ({"fields": []}, "no list of fields"),
        ({"fields": ["A"], "prompt_file": "x.yaml"}, "does not know: prompt_file"),
        ({"fields": ["A"], "completion_tokens": "many"}, "completion budget"),
        ({"fields": ["A"], "also_called": "permit"}, "not a list"),
        ({"fields": ["A"], "table": {"columns": []}}, "table of 'permit' cannot be read"),
        (
            {"fields": ["A"], "table": {"columns": ["A"], "rows_in_date_order_of": "B"}},
            "not one of its columns",
        ),
    ],
)
def test_a_block_that_cannot_be_used_is_refused_with_a_full_diagnostic(tmp_path, block, said):
    path = tmp_path / "field_definitions.yaml"
    entry = {"cleaned_as": "text", "list": False, "scored_as": "text"}
    path.write_text(yaml.safe_dump({"document_types": {"permit": block}, "fields": {"A": entry}}))

    with pytest.raises(FieldSchemaError) as refused:
        FieldSchema.from_yaml(path)

    message = str(refused.value)
    assert said in message
    assert all(part in message for part in ("What:", "Where:", "Expected:", "How to fix:"))
    assert str(path.absolute()) in message
    assert "document_types.permit" in message or "fields.B" in message


def test_no_python_names_a_document_type():
    """The guard: a type's name in code is a second place to edit when a type is added."""
    names = "|".join(
        yaml.safe_load((REPO / "config" / "field_definitions.yaml").read_text())["document_types"]
    )
    quoted = re.compile(rf"""["'.]({names})["']""", re.IGNORECASE)
    files = subprocess.run(
        ["git", "ls-files", "stages/*.py", "common/*.py", "models/*.py", "models/backends/*.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    found = []
    for name in files:
        for number, line in enumerate((REPO / name).read_text().splitlines(), start=1):
            code = line.split("#", 1)[0]
            if quoted.search(code):
                found.append(f"{name}:{number}: {line.strip()}")
    assert found == []
