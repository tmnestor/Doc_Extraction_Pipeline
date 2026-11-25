"""A reply's values are cleaned one way, whichever way the reply was written.

Until 24 November 2026 a reply was tidied in two passes that overlapped. The first ran inside the parser,
for a reply written as lines only: it put lists, dates, quantities and addresses into their forms. The
second cleaned each value, and split a list at its commas again, twice. A reply written as JSON skipped
the first pass, so the same values came out differently: a date was left as the model wrote it, prices
with wide gaps between them stayed one entry, and an address lost its commas without a space in their
place.

Every name, address and amount is invented.
"""

import json
import re
from pathlib import Path

import pytest

from common import reply_fields

FIELDS = [
    "DOCUMENT_TYPE",
    "BUSINESS_ADDRESS",
    "INVOICE_DATE",
    "LINE_ITEM_DESCRIPTIONS",
    "LINE_ITEM_QUANTITIES",
    "LINE_ITEM_PRICES",
    "TRANSACTION_DATES",
]
# Each value in a form the model was not asked for.
AS_WRITTEN = {
    "DOCUMENT_TYPE": "Tax Invoice",
    "BUSINESS_ADDRESS": "14 Quillfeather Lane,Marrowby VIC 3999",
    "INVOICE_DATE": "2031-02-09",
    "LINE_ITEM_DESCRIPTIONS": "Ink, Blotting paper, Nibs",
    "LINE_ITEM_QUANTITIES": "x2 | 1 EACH | 3x",
    "LINE_ITEM_PRICES": "$4.50   $12.00   $3.25",
    "TRANSACTION_DATES": "2031-02-03 | 2031-02-09",
}
CLEANED = {
    "DOCUMENT_TYPE": "INVOICE",
    # An address keeps its commas, as the page prints them and the answer key writes them.
    "BUSINESS_ADDRESS": "14 Quillfeather Lane,Marrowby VIC 3999",
    "INVOICE_DATE": "09/02/2031",
    "LINE_ITEM_DESCRIPTIONS": "Ink | Blotting paper | Nibs",
    "LINE_ITEM_QUANTITIES": "2 | 1 | 3",
    "LINE_ITEM_PRICES": "$4.50 | $12.00 | $3.25",
    "TRANSACTION_DATES": "03/02/2031 | 09/02/2031",
}
REPLIES = {
    "lines": "\n".join(f"{name}: {value}" for name, value in AS_WRITTEN.items()),
    "json": json.dumps(AS_WRITTEN, indent=2),
    "json in a code fence": f"```json\n{json.dumps(AS_WRITTEN, indent=2)}\n```",
    "json on one line": json.dumps(AS_WRITTEN),
}


@pytest.mark.parametrize("written", REPLIES)
def test_the_same_values_clean_the_same_however_the_reply_is_written(written: str) -> None:
    assert reply_fields.fields_from_reply(REPLIES[written], FIELDS) == CLEANED


@pytest.mark.parametrize("written", REPLIES)
def test_the_parser_gives_each_value_as_the_reply_wrote_it(written: str) -> None:
    """Putting a value into its form is the cleaner's work, so that there is one place it is done."""
    assert reply_fields.parse_reply(REPLIES[written], FIELDS) == AS_WRITTEN


@pytest.mark.parametrize("name", FIELDS)
def test_one_function_takes_a_value_from_as_written_to_cleaned(name: str) -> None:
    assert reply_fields.clean_value(name, AS_WRITTEN[name]) == CLEANED[name]


def test_cleaning_a_cleaned_value_changes_nothing() -> None:
    for name, value in CLEANED.items():
        assert reply_fields.clean_value(name, value) == value


@pytest.mark.parametrize("nothing", ["NOT_FOUND", "", None, "N/A", "null", "None"])
def test_nothing_is_not_found_for_every_kind_of_field(nothing: object) -> None:
    for name in FIELDS:
        assert reply_fields.clean_value(name, nothing) == "NOT_FOUND"


def test_a_list_of_nothing_left_after_its_commas_is_not_found() -> None:
    """The list is put into its form before it is asked whether it is nothing: ", N/A" is one N/A."""
    assert reply_fields.clean_value("LINE_ITEM_DESCRIPTIONS", ", N/A") == "NOT_FOUND"
    assert reply_fields.clean_value("LINE_ITEM_DESCRIPTIONS", "N/A, N/A") == "N/A | N/A"


def test_a_comma_inside_a_number_does_not_divide_a_list() -> None:
    assert reply_fields.clean_value("LINE_ITEM_PRICES", "$8,026.87, $14.00") == "$8026.87 | $14.00"
    assert reply_fields.clean_value("LINE_ITEM_PRICES", "$8,026.87") == "$8026.87"


def test_a_list_is_divided_at_its_commas_in_one_place() -> None:
    source = Path(reply_fields.__file__).read_text()

    assert len(re.findall(r're\.split\(r",\(\?!\\d\)"', source)) == 1
    # The mark for a value that is not there, and the label before an identifier, are each defined once.
    from common import field_schema, scoring

    assert reply_fields.NOT_FOUND is scoring.NOT_FOUND is field_schema.NOT_FOUND
    assert scoring.IDENTIFIER_LABEL is reply_fields.IDENTIFIER_LABEL
    assert "NOT_FOUND =" not in Path(scoring.__file__).read_text()
