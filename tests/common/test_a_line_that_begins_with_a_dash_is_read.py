"""A reply whose lines begin with a dash is read, as one whose lines begin with a star is.

A model asked for ``NAME: value`` lines sometimes writes them as a list: ``- NAME: value``. The parser
took a star from the front of a line and not a dash, so the name it then looked for was "- NAME", which
is no field's: every field of such a reply came out as not found, and the page scored nothing though the
model had read it.

Found on 25 November 2026, when the receipts of a public dataset were first read by the pipeline's own
parser: the task that had read them until then did read such a line.

Every reply here is invented.
"""

import pytest

from common.reply_fields import fields_from_reply

FIELDS = ["SUPPLIER_NAME", "INVOICE_DATE", "LINE_ITEM_DESCRIPTIONS", "TOTAL_AMOUNT"]
LINES = [
    "SUPPLIER_NAME: Marrowby Grocer",
    "INVOICE_DATE: 09/02/2031",
    "LINE_ITEM_DESCRIPTIONS: Ink | Blotting paper",
    "TOTAL_AMOUNT: $23.15",
]
READ = {
    "SUPPLIER_NAME": "Marrowby Grocer",
    "INVOICE_DATE": "09/02/2031",
    "LINE_ITEM_DESCRIPTIONS": "Ink | Blotting paper",
    "TOTAL_AMOUNT": "$23.15",
}


def with_each_line_begun_by(mark: str) -> str:
    return "".join(f"{mark}{line}\n" for line in LINES)


@pytest.mark.parametrize("mark", ["", "- ", "-  ", "  - ", "• ", "* ", "*", "– "])
def test_a_reply_is_read_whatever_mark_begins_its_lines(mark: str) -> None:
    assert fields_from_reply(with_each_line_begun_by(mark), FIELDS) == READ


def test_names_in_bold_after_a_dash_are_read() -> None:
    reply = "".join(f"- **{line.split(':', 1)[0]}:**{line.split(':', 1)[1]}\n" for line in LINES)

    assert fields_from_reply(reply, FIELDS) == READ


def test_a_list_written_as_dashes_below_its_name_is_still_one_list() -> None:
    reply = "SUPPLIER_NAME: Marrowby Grocer\nLINE_ITEM_DESCRIPTIONS:\n- Ink\n- Blotting paper\nTOTAL_AMOUNT: $23.15\n"

    read = fields_from_reply(reply, FIELDS)

    assert read["LINE_ITEM_DESCRIPTIONS"] == "Ink | Blotting paper"
    assert read["TOTAL_AMOUNT"] == "$23.15"


def test_a_dash_that_is_part_of_a_value_is_left_alone() -> None:
    """Only a dash before a field's name is a mark. An amount below nothing is still below nothing."""
    read = fields_from_reply("SUPPLIER_NAME: Marrowby-on-Quill Grocer\nTOTAL_AMOUNT: -23.15\n", FIELDS)

    assert read["SUPPLIER_NAME"] == "Marrowby-on-Quill Grocer"
    assert read["TOTAL_AMOUNT"].replace("$", "") == "-23.15"


def test_a_line_that_begins_with_a_dash_and_names_no_field_is_passed_over() -> None:
    reply = "- NOTE: the receipt is torn\n" + with_each_line_begun_by("- ")

    assert fields_from_reply(reply, FIELDS) == READ
