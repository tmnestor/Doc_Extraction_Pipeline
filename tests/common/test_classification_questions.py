"""The classification questions are data: a new question and a rule recognise a new document type.

Nothing in Python names a question. These tests add a question the pipeline has never heard of to a copy
of the detection file, with a rule that uses it, and check that the prompt asks it and that a reply's
answer to it, in each of the three forms a reply may take, gives the new type. All content is invented.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from common import classification, field_schema
from common.field_schema import FieldSchema
from common.prompt_catalog import PromptCatalog

REPO = Path(__file__).resolve().parents[2]
SHIPPED_PROMPT = """Look at this document image and answer each question:

1. COLUMNS: If there is a transaction table with rows of dates,
   descriptions, and amounts, list the exact column headers
   separated by " | ".
   If there is no transaction table, write NONE.

2. PAID: Is there evidence that payment was completed?
   Look for: payment method, "PAID" stamp, amount tendered,
   change given, receipt number, EFTPOS/card details.
   Answer YES or NO.

3. ROWS: How many line items or transaction rows are visible?
   Answer with a number.

4. TRAVEL: Is this a flight ticket, e-ticket, boarding pass, or
   travel itinerary (showing passenger names, flights, or travel
   legs)? Answer YES or NO.
"""


def test_the_prompt_built_from_the_shipped_questions_is_the_prompt_the_runs_were_made_with() -> None:
    assert PromptCatalog().classification_prompt() == SHIPPED_PROMPT


@pytest.fixture
def with_a_permit_question(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[PromptCatalog]:
    """The shipped files with a fifth question, a rule that uses it, and the type the rule gives."""
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    for source in (REPO / "prompts").glob("*.yaml"):
        (prompts / source.name).write_text(source.read_text())
    detection = yaml.safe_load((prompts / "document_type_detection.yaml").read_text())
    detection["classification"]["questions"].append(
        {
            "label": "PERMIT",
            "answer": "yes_no",
            "ask": "Is this a parking permit\nwith a bay number? Answer YES or NO.\n",
        }
    )
    detection["classification_evidence"]["rules"].insert(
        0, {"type": "PARKING_PERMIT", "when": {"permit": True}}
    )
    (prompts / "document_type_detection.yaml").write_text(yaml.safe_dump(detection, sort_keys=False))

    schema = yaml.safe_load((REPO / "config" / "field_definitions.yaml").read_text())
    schema["document_types"]["parking_permit"] = {"fields": ["DOCUMENT_TYPE", "BAY_NUMBER"]}
    schema_path = tmp_path / "field_definitions.yaml"
    schema_path.write_text(yaml.safe_dump(schema, sort_keys=False))
    made = FieldSchema.from_yaml(schema_path)
    monkeypatch.setattr(field_schema, "get_field_schema", lambda: made)

    catalog = PromptCatalog(prompts_dir=prompts)
    monkeypatch.setattr(classification, "PromptCatalog", lambda: catalog)
    classification._settings.cache_clear()
    yield catalog
    classification._settings.cache_clear()


def test_a_new_question_is_asked_last_with_its_number_and_label(
    with_a_permit_question: PromptCatalog,
) -> None:
    prompt = with_a_permit_question.classification_prompt()

    assert prompt.startswith(SHIPPED_PROMPT)
    assert prompt.endswith(
        "\n\n5. PERMIT: Is this a parking permit\n   with a bay number? Answer YES or NO.\n"
    )


@pytest.mark.parametrize(
    "reply",
    [
        "1. COLUMNS: NONE\n2. PAID: NO\n3. ROWS: 0\n4. TRAVEL: NO\n5. PERMIT: YES",
        "1. NONE\n2. NO\n3. 0\n4. NO\n5. YES",
        "### 1. COLUMNS\nNONE\n### 2. PAID\nNO\n### 3. ROWS\n0\n### 4. TRAVEL\nNO\n### 5. PERMIT\nYES",
    ],
    ids=["labelled", "by position", "under headings"],
)
def test_a_reply_that_answers_the_new_question_yes_gives_the_new_type(
    with_a_permit_question, reply: str
) -> None:
    assert classification.document_type(reply, "UNIVERSAL") == "PARKING_PERMIT"
    assert classification.document_type(reply.replace("YES", "NO"), "UNIVERSAL") == "INVOICE"


def test_a_reply_with_only_the_four_old_answers_is_not_read_by_position(with_a_permit_question) -> None:
    """By position every question must be answered, or prose shaped like a list would be read as answers."""
    assert classification.document_type("1. NONE\n2. YES\n3. 4\n4. NO", "UNIVERSAL") == "UNIVERSAL"


def test_each_yes_or_no_question_is_read_under_its_own_heading() -> None:
    reply = "### 1. COLUMNS\nNONE\n### 2. PAID\nAnswer YES or NO.\nYES\n### 3. ROWS\n3\n### 4. TRAVEL\nNO"

    assert classification.document_type(reply, "UNIVERSAL") == "RECEIPT"
    assert classification.document_type(reply.replace("\nYES\n", "\nNO\n"), "UNIVERSAL") == "INVOICE"


def broken(tmp_path: Path, change) -> PromptCatalog:
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    detection = yaml.safe_load((REPO / "prompts" / "document_type_detection.yaml").read_text())
    change(detection)
    (prompts / "document_type_detection.yaml").write_text(yaml.safe_dump(detection, sort_keys=False))
    return PromptCatalog(prompts_dir=prompts)


@pytest.mark.parametrize(
    ("change", "said"),
    [
        (lambda d: d.pop("classification"), "no opening line or no questions"),
        (lambda d: d["classification"]["questions"][1].update(label="paid"), "no label in capitals"),
        (lambda d: d["classification"]["questions"][1].update(answer="maybe"), "answer kind 'maybe'"),
        (lambda d: d["classification"]["questions"][1].update(ask=" "), "has no text"),
        (
            lambda d: d["classification"]["questions"][3].update(label="PAID"),
            "two questions have the label PAID",
        ),
        (
            lambda d: d["classification"]["questions"].pop(0),
            "not exactly one question of the kind `columns`",
        ),
    ],
)
def test_questions_that_cannot_be_used_are_refused_with_a_full_diagnostic(tmp_path, change, said) -> None:
    catalog = broken(tmp_path, change)

    with pytest.raises(ValueError) as refused:
        catalog.get_classification_questions()

    message = str(refused.value)
    assert said in message
    assert all(part in message for part in ("What:", "Where:", "Expected:", "How to fix:"))
    assert str((tmp_path / "prompts" / "document_type_detection.yaml").absolute()) in message


def test_a_rule_may_only_use_a_question_that_is_asked(tmp_path) -> None:
    def change(detection) -> None:
        detection["classification_evidence"]["rules"].insert(
            0, {"type": "RECEIPT", "when": {"permit": True}}
        )

    with pytest.raises(ValueError, match="Unknown 'when:' key") as refused:
        broken(tmp_path, change).get_classification_evidence()
    assert "classification.questions" in str(refused.value)
