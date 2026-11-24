"""A date written year first (2031-02-09) is the ninth of February, to the cleaner and to the scorer.

Found 24 November 2026. The cleaner read every date with the day first, so a year-first date came out with
its day and month changed over whenever the day was 12 or less: 2031-02-09 became 02/09/2031, and nothing
said so. The scorer took the first three numbers of a date as day, month and year, so a year-first date
never matched the same day written the usual way. No recorded reply or answer key held such a date, so no
result was ever affected; a model or a document type that writes dates this way would have been.

All dates are invented.
"""

import pytest

from common import reply_fields, scoring

# INVOICE_DATE is a single date and TRANSACTION_DATES a list of them, both tidied as day_month_year.
ONE_DATE = "INVOICE_DATE"
LIST_OF_DATES = "TRANSACTION_DATES"


@pytest.mark.parametrize(
    ("written", "meant"),
    [
        ("2031-02-09", "09/02/2031"),
        ("2031-02-19", "19/02/2031"),
        ("2031/02/09", "09/02/2031"),
        ("2031.02.09", "09/02/2031"),
        ("2031-2-9", "09/02/2031"),
        ("20310209", "09/02/2031"),
        ("2031-02-09T10:15:00", "09/02/2031"),
        ("2031-02-09 10:15", "09/02/2031"),
        ("2031-12-01", "01/12/2031"),
    ],
)
def test_the_cleaner_reads_a_year_first_date_as_year_month_day(written: str, meant: str) -> None:
    assert reply_fields._as_day_month_year(written) == meant


@pytest.mark.parametrize(
    ("written", "meant"),
    [
        ("09/02/2031", "09/02/2031"),
        ("9 Feb 2031", "09/02/2031"),
        ("Feb 9, 2031", "09/02/2031"),
        ("2031 Feb 9", "09/02/2031"),
        ("02/09/31", "02/09/2031"),
        ("9-2-2031", "09/02/2031"),
        ("09.02.2031", "09/02/2031"),
        ("09022031", "09022031"),
        ("NOT_FOUND", "NOT_FOUND"),
        ("next Tuesday", "next Tuesday"),
    ],
)
def test_the_cleaner_reads_every_other_date_as_it_did(written: str, meant: str) -> None:
    assert reply_fields._as_day_month_year(written) == meant


def test_a_year_first_date_that_is_no_day_of_the_year_is_not_guessed_at() -> None:
    """The 13th month and the 31st of February: read as before, never as year, month, day."""
    assert reply_fields._as_day_month_year("2031-13-01") == "13/01/2031"
    assert reply_fields._as_day_month_year("2031-02-31") == "2031-02-31"


def test_a_reply_with_year_first_dates_is_cleaned_to_the_right_days() -> None:
    reply = f"{ONE_DATE}: 2031-02-09\n{LIST_OF_DATES}: 2031-02-03 | 2031-02-09 | 2031-02-19\n"

    assert reply_fields.fields_from_reply(reply, [ONE_DATE, LIST_OF_DATES]) == {
        ONE_DATE: "09/02/2031",
        LIST_OF_DATES: "03/02/2031 | 09/02/2031 | 19/02/2031",
    }


@pytest.mark.parametrize(
    ("written", "day_month_year"),
    [
        ("2031-02-09", (9, 2, 2031)),
        ("2031/2/9", (9, 2, 2031)),
        ("2031-02-09T10:15:00", (9, 2, 2031)),
        ("2031 Feb 9", (9, 2, 2031)),
        ("Thu 2031-02-06", (6, 2, 2031)),
        ("09/02/2031", (9, 2, 2031)),
        ("9 Feb 2031", (9, 2, 2031)),
        ("02/09/31", (2, 9, 2031)),
        ("9 Feb 31", (9, 2, 2031)),
    ],
)
def test_the_scorer_reads_the_day_the_month_and_the_year(written: str, day_month_year) -> None:
    assert scoring.parse_date(written) == day_month_year


def test_the_scorer_matches_a_year_first_date_with_the_same_day_written_day_first() -> None:
    assert scoring.score_field("2031-02-09", "09/02/2031", ONE_DATE)["f1_score"] == 1.0
    assert scoring.score_field("09/02/2031", "2031-02-09", ONE_DATE)["f1_score"] == 1.0
    assert scoring.score_field("2031-02-09", "02/09/2031", ONE_DATE)["f1_score"] < 1.0
    assert scoring.list_item_matches("2031-02-09", "09/02/2031", LIST_OF_DATES) is True
    assert scoring.list_item_matches("2031-02-09", "02/09/2031", LIST_OF_DATES) is False


@pytest.mark.parametrize(
    "period",
    [
        "2031-02-01 - 2031-02-28",
        "2031-02-01 to 2031-02-28",
        "2031-02-01-2031-02-28",
        "2031/02/01-2031/02/28",
    ],
)
def test_the_scorer_reads_both_ends_of_a_year_first_period(period: str) -> None:
    assert scoring._dates_in(period) == [(1, 2, 2031), (28, 2, 2031)]
    assert scoring._dates_in("01/02/2031 - 28/02/2031") == [(1, 2, 2031), (28, 2, 2031)]
    assert scoring._dates_in("01/02/2031-28/02/2031") == [(1, 2, 2031), (28, 2, 2031)]


def test_a_year_first_date_in_a_list_is_the_same_day_as_its_day_first_form() -> None:
    assert scoring._dates_match("2031-02-09", "9 Feb 2031") is True
    assert scoring._dates_match("2031 Feb 9", "09/02/2031") is True
    assert scoring._dates_match("2031-02-09", "2031-02-10") is False
    assert scoring._dates_match("09/02/2031", "9 Feb 2031") is True
