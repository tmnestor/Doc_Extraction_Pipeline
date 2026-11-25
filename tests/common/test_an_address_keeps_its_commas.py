"""An address is cleaned with its commas, as the page prints them and as the answer key writes them.

The answer key was once a CSV file, in which a comma inside a value could not stand, so an address had
none, and the cleaner took the commas out of every address it read to match. The key has been JSON lines
since, and its addresses have their commas. The cleaner went on taking them out. So an address read to
the letter was compared without its commas against an answer with them, and scored its similarity:
0.93 to 0.97, and never 1. On the 165 clean pages no address of any run had scored 1.

Found on 25 November 2026, when a public dataset of receipts was first scored by the pipeline's own
rules. The commas are now left where they are.

Every address here is invented.
"""

import json

import pytest

from common.field_schema import get_field_schema
from common.reply_fields import clean_value, fields_from_reply
from common.scoring import score_field

ADDRESS = "12 Rook Lane, Heron Bay QLD 4099"
ADDRESS_FIELDS = sorted(
    name for name, entry in get_field_schema().fields.items() if entry.cleaned_as == "address"
)


def test_the_schema_has_fields_cleaned_as_an_address() -> None:
    assert {"BUSINESS_ADDRESS", "PAYER_ADDRESS"} <= set(ADDRESS_FIELDS)


@pytest.mark.parametrize("name", ADDRESS_FIELDS)
def test_an_address_is_cleaned_with_its_commas(name: str) -> None:
    assert clean_value(name, ADDRESS) == ADDRESS
    assert clean_value(name, "Unit 4, 12 Rook Lane, Heron Bay, QLD 4099") == (
        "Unit 4, 12 Rook Lane, Heron Bay, QLD 4099"
    )


@pytest.mark.parametrize("name", ADDRESS_FIELDS)
def test_an_address_read_to_the_letter_scores_one(name: str) -> None:
    read = fields_from_reply(f"{name}: {ADDRESS}\n", [name])[name]

    assert read == ADDRESS
    assert score_field(read, ADDRESS, name)["f1_score"] == 1.0


def test_a_reply_written_as_json_keeps_them_too() -> None:
    reply = json.dumps({"BUSINESS_ADDRESS": ADDRESS})

    assert fields_from_reply(reply, ["BUSINESS_ADDRESS"])["BUSINESS_ADDRESS"] == ADDRESS


def test_an_address_the_model_wrote_without_its_commas_is_marked_down_for_them_and_no_more() -> None:
    """What the model wrote is what is scored: the cleaner no longer hides a comma it left out."""
    score = score_field(
        clean_value("BUSINESS_ADDRESS", "12 Rook Lane Heron Bay QLD 4099"), ADDRESS, "BUSINESS_ADDRESS"
    )

    assert 0.95 < score["f1_score"] < 1.0


@pytest.mark.parametrize(
    "written",
    [
        f"{ADDRESS} - Phone: (07) 5550 0142",
        f"{ADDRESS} (07) 5550 0142",
        f"{ADDRESS} - quill@heronbay.example",
        f"{ADDRESS};",
        f"  {ADDRESS}  ",
    ],
)
def test_what_is_not_the_address_is_still_taken_off(written: str) -> None:
    assert clean_value("BUSINESS_ADDRESS", written) == ADDRESS


def test_cleaning_a_cleaned_address_changes_nothing() -> None:
    once = clean_value("BUSINESS_ADDRESS", f"{ADDRESS} - Phone: (07) 5550 0142")

    assert clean_value("BUSINESS_ADDRESS", once) == once


def test_a_list_is_still_divided_at_its_commas() -> None:
    """Only an address is changed here: a list written with commas where bars were asked for is a list."""
    assert clean_value("LINE_ITEM_QUANTITIES", "2, 1, 4") == "2 | 1 | 4"
