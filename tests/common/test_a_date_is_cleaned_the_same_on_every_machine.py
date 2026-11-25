"""A date is put in one form by the pipeline itself, and so in the same form on every machine.

The cleaner wrote a date as DD/MM/YYYY with the system's own formatting of a date. That pads a year
below 1000 with noughts on a Mac and does not on Linux: a date the model wrote with a digit of its year
missing, "13/01/201", was cleaned to "13/01/0201" on one and "13/01/201" on the other. The same reply
cleaned to two different files, by where it was cleaned.

Found on 25 November 2026, when a run cleaned on the sandbox was set beside the same values cleaned on
the Mac and one date of 347 differed. The year is now written with four digits by the cleaner.

Every date here is invented.
"""

import datetime
from typing import Any

import pytest

from common import reply_fields
from common.reply_fields import clean_value


class AsLinuxWritesIt(datetime.datetime):
    """A date whose own formatting leaves a short year short, as the C library of Linux does."""

    def strftime(self, format: str) -> str:
        return (
            format.replace("%d", f"{self.day:02d}")
            .replace("%m", f"{self.month:02d}")
            .replace("%Y", str(self.year))
        )


class AsAMacWritesIt(datetime.datetime):
    """A date whose own formatting pads a short year, as the C library of a Mac does."""

    def strftime(self, format: str) -> str:
        return (
            format.replace("%d", f"{self.day:02d}")
            .replace("%m", f"{self.month:02d}")
            .replace("%Y", f"{self.year:04d}")
        )


@pytest.mark.parametrize("machine", [AsLinuxWritesIt, AsAMacWritesIt])
@pytest.mark.parametrize(
    ("written", "cleaned"),
    [("13/01/201", "13/01/0201"), ("9 Feb 2031", "09/02/2031"), ("2031-02-09", "09/02/2031")],
)
def test_a_date_is_cleaned_the_same_whatever_machine_cleans_it(
    monkeypatch: pytest.MonkeyPatch, machine: type[datetime.datetime], written: str, cleaned: str
) -> None:
    read = reply_fields.date_parser.parse

    def parse(text: str, **how: Any) -> datetime.datetime:
        found = read(text, **how)
        return machine(found.year, found.month, found.day)

    monkeypatch.setattr(reply_fields.date_parser, "parse", parse)

    assert clean_value("INVOICE_DATE", written) == cleaned


def test_each_date_of_a_list_is_cleaned_so() -> None:
    assert clean_value("TRANSACTION_DATES", "13/01/201 | 2031-02-09") == "13/01/0201 | 09/02/2031"
