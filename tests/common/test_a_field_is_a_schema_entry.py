"""A field is an entry in the schema file: how its values are tidied, whether it is a list, how it is scored.

Nothing in Python reads a field's name to decide any of that. These tests give invented fields entries in
a copy of the schema file and check that a reply's values are parsed and tidied by the entries, and they
check that no field's name appears in the code. All field names and values are invented.
"""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

from common import field_schema, reply_fields
from common.field_schema import CLEANED_AS, SCORED_AS, FieldSchema, FieldSchemaError

REPO = Path(__file__).resolve().parents[2]
SHIPPED = REPO / "config" / "field_definitions.yaml"
FIELDS = [
    "PERMIT_HOLDER",
    "BAY_FEES",
    "BAY_DATES",
    "BAY_COUNTS",
    "DEPOT_ADDRESS",
    "PERMIT_REF",
    "ISSUED",
    "NOTES",
]


@pytest.fixture
def invented(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Eight fields whose names say nothing of what they are, each with an entry that does."""
    schema = yaml.safe_load(SHIPPED.read_text())
    schema["fields"].update(
        {
            "PERMIT_HOLDER": {"cleaned_as": "text", "list": False, "scored_as": "text"},
            "BAY_FEES": {"cleaned_as": "money", "list": True, "scored_as": "money"},
            "BAY_DATES": {"cleaned_as": "day_month_year", "list": True, "scored_as": "date"},
            "BAY_COUNTS": {"cleaned_as": "quantity", "list": True, "scored_as": "text"},
            "DEPOT_ADDRESS": {"cleaned_as": "address", "list": False, "scored_as": "text"},
            "PERMIT_REF": {"cleaned_as": "identifier", "list": False, "scored_as": "text"},
            "ISSUED": {"cleaned_as": "day_month_year", "list": False, "scored_as": "date"},
            "NOTES": {"cleaned_as": "text", "list": True, "scored_as": "text", "also_written": ["NOTE"]},
        }
    )
    schema["document_types"]["parking_permit"] = {"fields": FIELDS}
    path = tmp_path / "field_definitions.yaml"
    path.write_text(yaml.safe_dump(schema, sort_keys=False))
    made = FieldSchema.from_yaml(path)
    monkeypatch.setattr(field_schema, "get_field_schema", lambda: made)
    monkeypatch.setattr(reply_fields, "get_field_schema", lambda: made)
    return FIELDS


def test_a_reply_is_tidied_by_what_each_fields_entry_says(invented: list[str]) -> None:
    reply = (
        "PERMIT_HOLDER: **Marlowe Quay Cartage**,\n"
        "BAY_FEES: AUD 1,204.50 each, 6.30\n"
        "BAY_DATES: 3 Feb 2031 | 9 February 2031\n"
        "BAY_COUNTS: 2x | x4 | 1 EACH\n"
        "DEPOT_ADDRESS: 4 Tannery Row, Eastmere - Phone: (07) 5550 0142\n"
        "PERMIT_REF: NO. 77 104 2\n"
        "ISSUED: 1 March 2031\n"
        "NOTE: gate code changed, bay 4 resurfaced\n"
    )

    assert reply_fields.fields_from_reply(reply, invented) == {
        "PERMIT_HOLDER": "Marlowe Quay Cartage",
        "BAY_FEES": "$1204.50 | 6.30",
        "BAY_DATES": "03/02/2031 | 09/02/2031",
        "BAY_COUNTS": "2 | 4 | 1",
        "DEPOT_ADDRESS": "4 Tannery Row, Eastmere",
        "PERMIT_REF": "77 104 2",
        "ISSUED": "01/03/2031",
        "NOTES": "gate code changed | bay 4 resurfaced",
    }


def test_a_list_given_as_bullets_below_its_name_is_read_for_any_list_field(invented: list[str]) -> None:
    plain = "BAY_FEES:\n* $4.10\n* $6.30\nPERMIT_HOLDER: Marlowe Quay Cartage\n"
    bold = "**BAY_FEES:**\n* $4.10\n* $6.30\n\n**PERMIT_HOLDER:**\nMarlowe Quay Cartage\n"

    for reply in (plain, bold):
        read = reply_fields.fields_from_reply(reply, invented)
        assert read["BAY_FEES"] == "$4.10 | $6.30"
        assert read["PERMIT_HOLDER"] == "Marlowe Quay Cartage"


def test_a_name_with_no_entry_is_one_piece_of_text() -> None:
    assert reply_fields.clean_value("UNHEARD_OF_TOTAL_AMOUNT_DATES", " 1,204.50 , ") == "1,204.50"


def test_every_field_a_type_names_has_an_entry_and_no_entry_is_unused() -> None:
    schema = yaml.safe_load(SHIPPED.read_text())
    named = set()
    for block in schema["document_types"].values():
        named |= set(block["fields"]) | set((block.get("table") or {}).get("columns", []))

    assert set(schema["fields"]) == named


@pytest.mark.parametrize(
    ("entry", "said"),
    [
        ({"cleaned_as": "text", "list": False}, "does not say all of"),
        ({"cleaned_as": "currency", "list": False, "scored_as": "text"}, "cleaned_as: 'currency'"),
        ({"cleaned_as": "text", "list": False, "scored_as": "number"}, "scored_as: 'number'"),
        ({"cleaned_as": "text", "list": "yes", "scored_as": "text"}, "list: 'yes'"),
        ({"cleaned_as": "text", "list": False, "scored_as": "text", "units": "km"}, "does not know: units"),
        ({"cleaned_as": "text", "list": False, "scored_as": "text", "also_written": "A1"}, "not a list"),
    ],
)
def test_an_entry_that_cannot_be_used_is_refused_with_a_full_diagnostic(tmp_path, entry, said) -> None:
    path = tmp_path / "field_definitions.yaml"
    path.write_text(
        yaml.safe_dump({"document_types": {"permit": {"fields": ["A"]}}, "fields": {"A": entry}})
    )

    with pytest.raises(FieldSchemaError) as refused:
        FieldSchema.from_yaml(path)

    message = str(refused.value)
    assert said in message
    assert all(part in message for part in ("What:", "Where:", "Expected:", "How to fix:"))
    assert str(path.absolute()) in message and "fields.A" in message


def test_the_kinds_the_schema_file_describes_are_the_kinds_the_code_knows() -> None:
    head = SHIPPED.read_text()

    for kind in CLEANED_AS:
        assert re.search(rf"^#\s+{kind}\s", head, re.MULTILINE), kind
    assert all(kind in head for kind in SCORED_AS)


def test_no_python_names_a_field() -> None:
    """The guard: a field's name in code is a second place to edit when a field is added."""
    names = "|".join(yaml.safe_load(SHIPPED.read_text())["fields"])
    quoted = re.compile(rf"""["']({names})["':]""")
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
            if quoted.search(line.split("#", 1)[0]):
                found.append(f"{name}:{number}: {line.strip()}")
    assert found == []
