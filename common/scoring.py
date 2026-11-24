"""Scoring: what a page's fields were read as, against what they are.

One module holds the whole of it:

- the answer key: loading it, and finding a page's answer;
- comparing one value with another, by the kind of field (money, date, identifier, text, list);
- a field's score: precision, recall and F1, with lists compared position by position;
- a page's score: its fields' scores and their summary.

The evaluate stage calls ``load_answer_key``, ``find_answer`` and ``score_page``. Nothing here calls a
model.
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .field_schema import get_field_schema

logger = logging.getLogger(__name__)

NOT_FOUND = "NOT_FOUND"

# ---------------------------------------------------------------------------------------------------
# The answer key
# ---------------------------------------------------------------------------------------------------

_NAME_COLUMNS = ("image_file", "filename", "image_name", "file")


def load_answer_key(path: Path) -> dict[str, dict[str, Any]]:
    """The answer key as a map from a page's file name to its answer.

    A ``.jsonl`` file holds one answer a line, each with only its own document type's fields. Any other
    file is read as CSV, with every column for every row.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If it holds no answers, or a CSV has no column naming the page.
    """
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")
    answers = _load_jsonl(path) if path.suffix == ".jsonl" else _load_csv(path)
    if not answers:
        raise ValueError(
            f"Ground truth loaded from {path} but contains 0 records. Check file format and contents."
        )
    kinds = Counter(answer.get("DOCUMENT_TYPE", "UNKNOWN") for answer in answers.values())
    logger.info("Answer key: %d pages from %s (%s)", len(answers), path, dict(sorted(kinds.items())))
    return answers


def _load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    answers: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        name = record.get("filename") or record.get("image_name")
        if name:
            answers[str(name)] = record
    return answers


def _load_csv(path: Path) -> dict[str, dict[str, Any]]:
    try:
        # Every cell as text, or pandas turns the text "False" into the boolean False.
        table = pd.read_csv(path, dtype=str)
    except Exception as err:
        raise ValueError(f"Error reading CSV file: {err}") from err
    column = next((name for name in _NAME_COLUMNS if name in table.columns), None)
    if column is None:
        raise ValueError(f"No image identifier column found. Expected one of: {list(_NAME_COLUMNS)}")
    return {str(row[column]): row.to_dict() for _, row in table.iterrows() if not pd.isna(row[column])}


def find_answer(answers: dict[str, dict[str, Any]], image_path: str) -> dict[str, Any]:
    """A page's answer: by its file name, else by its name without the extension. Empty if none."""
    name, stem = Path(image_path).name, Path(image_path).stem
    if answers.get(name):
        return answers[name]
    for key, answer in answers.items():
        if Path(key).stem == stem or key == stem:
            return answer
    return {}


# ---------------------------------------------------------------------------------------------------
# Comparing one value with another
# ---------------------------------------------------------------------------------------------------

_MONTHS: dict[str, int] = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_DAY_NAMES = (
    "mon",
    "tue",
    "wed",
    "thu",
    "fri",
    "sat",
    "sun",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)


def _month_named(text: str) -> int | None:
    """The first month whose name appears in lower-case text."""
    return next((number for name, number in _MONTHS.items() if name in text), None)


def parse_date(text: str) -> tuple[int, int, int] | None:
    """A date as ``(day, month, year)``, or None if it cannot be read.

    Reads a leading day name ("Thu 04 Sep 2025"), month names, any separator, and two-digit years.
    """
    text = text.strip().lower()
    for day_name in _DAY_NAMES:
        if text.startswith(day_name):
            text = text[len(day_name) :].strip()
            break
    numbers = re.findall(r"\d+", text)
    month = _month_named(text)
    if len(numbers) >= 3:
        day, month, year = int(numbers[0]), int(numbers[1]), int(numbers[2])
    elif len(numbers) == 2 and month:
        day, year = int(numbers[0]), int(numbers[1])
    elif len(numbers) == 2:
        day, month, year = int(numbers[0]), int(numbers[1]), 0
    else:
        return None
    if year < 100:
        year = 2000 + year if year <= 50 else 1900 + year
    return (day, month, year)


def _dates_in(text: str) -> list[tuple[int, int, int]]:
    """The dates of a single date or of a range ("1 February 2024 - 28 June 2024")."""
    dates: list[tuple[int, int, int]] = []
    for separator in (" - ", " to ", " \u2013 ", " \u2014 ", "-"):  # the last two: en and em dash
        if separator in text.lower():
            parts = text.split(separator) if separator != "-" else re.split(r"\s*-\s*", text)
            parts = [part.strip() for part in parts if part.strip() and len(part.strip()) > 4]
            if len(parts) >= 2:
                dates.extend(parsed for part in parts[:2] if (parsed := parse_date(part)))
                break
    if not dates and (parsed := parse_date(text)):
        dates.append(parsed)
    return dates


def _date_field_score(read: str, truth: str) -> float:
    """How well a date or date range agrees: 1 the same, part for part of a range, 0.8 for day and month."""
    read_dates, true_dates = _dates_in(read), _dates_in(truth)
    if not read_dates or not true_dates:
        return 0.0
    if set(read_dates) == set(true_dates):
        return 1.0
    common = set(read_dates) & set(true_dates)
    if common:
        return len(common) / max(len(set(read_dates)), len(set(true_dates)))
    if read_dates[0][:2] == true_dates[0][:2]:
        return 0.8
    return 0.0


def _date_parts(text: str) -> tuple[str | None, str | None, str | None] | None:
    """A date in a list as zero-padded (day, month, year) text, any of which may be unknown."""
    numbers = re.findall(r"\d+", text)
    named = _month_named(text.lower())
    month_text = f"{named:02d}" if named else None
    day: str | None
    month: str | None
    year: str | None
    if len(numbers) == 3:
        day, month, year = numbers[0], numbers[1], numbers[2]
    elif len(numbers) == 2 and month_text:
        day, month, year = numbers[0], month_text, numbers[1]
    elif len(numbers) == 2:
        day, month, year = numbers[0], None, numbers[1]
    elif len(numbers) == 1 and month_text:
        day, month, year = numbers[0], month_text, None
    else:
        return None
    if year and len(year) == 2:
        year = str(2000 + int(year)) if int(year) <= 50 else str(1900 + int(year))
    return (day.zfill(2), month.zfill(2) if month else None, year)


def _dates_match(read: str, truth: str) -> bool:
    """Whether two dates in a list are the same day, whatever their formats."""
    if read.strip() == truth.strip():
        return True
    read_parts, true_parts = _date_parts(read), _date_parts(truth)
    if read_parts is None or true_parts is None:
        return False
    pairs = list(zip(read_parts, true_parts, strict=True))
    if any(a is not None and b is not None and a != b for a, b in pairs):
        return False
    return sum(1 for a, b in pairs if a == b and a is not None) >= 2


def _money_matches(read: str, truth: str) -> bool:
    """Whether two amounts are the same number, within the tolerance the field settings give."""
    tolerance = get_field_schema().get_threshold("monetary_tolerance")
    try:
        read_number = float(re.sub(r"[^\d.-]", "", read))
        true_number = float(re.sub(r"[^\d.-]", "", truth))
    except (ValueError, TypeError):
        return False
    allowed = abs(true_number * tolerance) if true_number != 0 else tolerance
    return abs(read_number - true_number) <= allowed


def _edit_distance(first: str, second: str) -> int:
    """Levenshtein distance, in two rows."""
    if first == second:
        return 0
    if not first:
        return len(second)
    if not second:
        return len(first)
    previous = list(range(len(second) + 1))
    for i, first_char in enumerate(first, start=1):
        current = [i]
        for j, second_char in enumerate(second, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (first_char != second_char))
            )
        previous = current
    return previous[-1]


def _text_similarity(read: str, truth: str) -> float:
    """Similarity in [0, 1]: one less the edit distance over the longer length, ignoring case."""
    left, right = read.lower(), truth.lower()
    longest = max(len(left), len(right))
    if longest == 0:
        return 1.0
    try:  # the C implementation when present: the same distance, faster
        from Levenshtein import distance as fast_distance

        distance = fast_distance(left, right)
    except ImportError:
        distance = _edit_distance(left, right)
    return 1.0 - (distance / longest)


def _words_match(read: str, truth: str) -> bool:
    """Whether two texts share enough of their words (the share the field settings give)."""
    threshold = get_field_schema().get_threshold("fuzzy_text_match")
    read_words, true_words = set(read.lower().strip().split()), set(truth.lower().strip().split())
    if not read_words or not true_words:
        return read.lower().strip() == truth.lower().strip()
    return len(read_words & true_words) / len(read_words | true_words) >= threshold


def _statement_item_matches(read: str, truth: str, field_name: str) -> bool:
    """Whether two entries of a bank statement's columns are the same.

    Amounts and balances are compared as money and dates as dates. A description matches if EITHER its
    letters or its words agree closely: the two measures fail on opposite faults. A quantity put before
    a description ("2x USB Hub" for "USB Hub") passes on words and fails on letters; one misread letter
    ("DO COASTAL" for "DD COASTAL") passes on letters and fails on words. The letter threshold is high
    because this asks "is it the same row?", and two different payments to one payee share most of
    their letters.
    """
    if "AMOUNT" in field_name:
        return _money_matches(read, truth)
    if "DATE" in field_name:
        return _dates_match(read, truth)
    if "BALANCE" in field_name:
        return _money_matches(read, truth)
    read, truth = read.strip(), truth.strip()
    threshold = get_field_schema().get_threshold("transaction_text_similarity")
    return _text_similarity(read, truth) >= threshold or _words_match(read, truth)


def list_item_matches(read: str, truth: str, field_name: str) -> bool:
    """Whether one entry of a list field matches its answer, by the kind of field.

    A ``NOT_FOUND`` entry is decided first: two of them match, and one against a value never does. It is
    not a price or a date, and must not reach a comparison that would try to read it as one.
    """
    read_absent, truth_absent = read.strip().upper() == NOT_FOUND, truth.strip().upper() == NOT_FOUND
    if read_absent or truth_absent:
        return read_absent and truth_absent
    fields = get_field_schema()
    if field_name in fields.transaction_list_fields:
        return _statement_item_matches(read, truth, field_name)
    if field_name in fields.monetary_fields:
        return _money_matches(read, truth)
    if field_name in fields.date_fields:
        return _dates_match(read, truth)
    return _words_match(read, truth)


# ---------------------------------------------------------------------------------------------------
# A field's score
# ---------------------------------------------------------------------------------------------------

_SINGLE_AMOUNTS = ("GST_AMOUNT", "TOTAL_AMOUNT", "INVOICE_TOTAL", "SUBTOTAL")
_IDENTIFIER_WORDS = ("ABN", "NUMBER", "ID", "REFERENCE", "BSB")
_IDENTIFIER_LABEL = r"^(ABN|BSB|ACN|GST|TAX|ID|NUMBER|NO\.?|#)\s*:?\s*"


def _scores(f1: float, precision: float, recall: float, tp: int, fp: int, fn: int) -> dict[str, Any]:
    return {"f1_score": f1, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def _right_or_wrong(match: bool) -> dict[str, Any]:
    return _scores(1.0, 1.0, 1.0, 1, 0, 0) if match else _scores(0.0, 0.0, 0.0, 0, 1, 1)


def _in_part(score: float, counts_from: float) -> dict[str, Any]:
    """A single value given part credit; it counts as found when its score is above ``counts_from``."""
    found = score > counts_from
    return _scores(score, score, score, int(found), int(not found), int(not found))


def _items(value: str) -> list[str]:
    return [item.strip() for item in value.split("|") if item.strip()]


def score_field(read_value: Any, true_value: Any, field_name: str) -> dict[str, Any]:
    """A field's precision, recall and F1.

    A list (values joined by "|") is compared position by position: an entry in the wrong place is a
    miss, an entry too many is a false find. A single value is compared by the kind of field it is.

    Returns:
        ``f1_score``, ``precision`` and ``recall`` between 0 and 1, and the counts ``tp``, ``fp``, ``fn``.
    """
    fields = get_field_schema()
    read = str(read_value).strip() if read_value else NOT_FOUND
    truth = str(true_value).strip() if true_value else NOT_FOUND

    if truth.upper() == NOT_FOUND:
        right = read.upper() == NOT_FOUND
        return _scores(float(right), float(right), 1.0, 0, int(not right), 0)
    if read.upper() == NOT_FOUND:
        return _scores(0.0, 0.0, 0.0, 0, 0, len(_items(truth)) or 1)

    if "|" in read or "|" in truth:
        return _score_list(_items(read), _items(truth), field_name)

    if field_name in fields.transaction_list_fields:
        return _right_or_wrong(_statement_item_matches(read, truth, field_name))

    read, truth = " ".join(read.split()), " ".join(truth.split())
    if field_name in fields.boolean_fields:
        read_flag, true_flag = _as_flag(read), _as_flag(truth)
        return _right_or_wrong(read_flag is not None and true_flag is not None and read_flag == true_flag)
    if "DATE" in field_name.upper():
        return _in_part(_date_field_score(read, truth), counts_from=0)
    if field_name in _SINGLE_AMOUNTS:
        return _right_or_wrong(_money_matches(read, truth))
    if any(word in field_name.upper() for word in _IDENTIFIER_WORDS):
        # An identifier is right or wrong: its label, spaces, dashes and case are set aside first.
        read_id = re.sub(r"[\s\-]", "", re.sub(_IDENTIFIER_LABEL, "", read, flags=re.IGNORECASE))
        true_id = re.sub(r"[\s\-]", "", re.sub(_IDENTIFIER_LABEL, "", truth, flags=re.IGNORECASE))
        return _in_part(float(read_id.lower() == true_id.lower()), counts_from=0.5)
    # Text earns its similarity, and nothing below a half: that is a wrong answer, not a near miss.
    similarity = _text_similarity(read, truth)
    return _in_part(similarity if similarity >= 0.5 else 0.0, counts_from=0.5)


def _as_flag(value: str) -> bool | None:
    lowered = value.lower().strip()
    if lowered in ("true", "1", "yes", "y"):
        return True
    if lowered in ("false", "0", "no", "n"):
        return False
    return None


def _score_list(read: list[str], truth: list[str], field_name: str) -> dict[str, Any]:
    tp = sum(list_item_matches(a, b, field_name) for a, b in zip(read, truth, strict=False))
    # A wrong entry where one was expected is one miss, not a miss and a false find as well.
    fn = len(truth) - tp
    fp = max(0, len(read) - len(truth))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return _scores(f1, precision, recall, tp, fp, fn)


# ---------------------------------------------------------------------------------------------------
# A page's score
# ---------------------------------------------------------------------------------------------------

_PAID = "TRANSACTION_AMOUNTS_PAID"
_STATEMENT_DATES = "TRANSACTION_DATES"
# The columns of a bank statement, which are lists of equal length, one entry a row.
_STATEMENT_COLUMNS = (
    "TRANSACTION_DATES",
    "TRANSACTION_AMOUNTS_PAID",
    "TRANSACTION_AMOUNTS_RECEIVED",
    "LINE_ITEM_DESCRIPTIONS",
    "ACCOUNT_BALANCE",
)


def _column(data: dict[str, Any], field: str) -> list[str] | None:
    """A statement column's entries, or None if it is absent or is not a list."""
    value = str(data.get(field, NOT_FOUND)).strip()
    if value.upper() == NOT_FOUND or "|" not in value:
        return None
    return _items(value)


def _keep_rows(data: dict[str, Any], rows: list[int], of: int) -> dict[str, Any]:
    """The statement columns with only the given rows, in the given order.

    A column whose length is not ``of`` cannot be lined up with the others and is left whole.
    """
    result = dict(data)
    for field in _STATEMENT_COLUMNS:
        entries = _column(data, field)
        if entries is not None and len(entries) == of:
            result[field] = " | ".join(entries[row] for row in rows)
    return result


def _without_credit_rows(data: dict[str, Any], document_type: str) -> dict[str, Any]:
    """A bank statement's columns without its credit rows.

    The answer key holds every row and marks a credit with ``NOT_FOUND`` in the paid column. The prompt
    asks for the paid column only, so the model gives the debit rows alone. Left as they are, the two
    lists are out of step by the number of credits and every position is marked wrong. Each side has
    its marked rows dropped; a side with none is returned as it is.
    """
    paid = _column(data, _PAID) if document_type.lower() == "bank_statement" else None
    if paid is None:
        return data
    keep = [row for row, value in enumerate(paid) if value.upper() != NOT_FOUND]
    # Nothing to drop, or everything: an empty list would score nothing against a full one.
    if len(keep) == len(paid) or not keep:
        return data
    return _keep_rows(data, keep, of=len(paid))


def _in_date_order(data: dict[str, Any], document_type: str) -> dict[str, Any]:
    """A bank statement's columns with their rows in date order.

    One side may list its rows as printed and the other by date. Putting both in date order before
    comparing position by position takes that difference away. A date that cannot be read sorts last.
    """
    dates = _column(data, _STATEMENT_DATES) if document_type.lower() == "bank_statement" else None
    if dates is None:
        return data
    parsed = [parse_date(date) or (9999, 99, 99) for date in dates]
    order = sorted(range(len(parsed)), key=lambda row: (parsed[row][2], parsed[row][1], parsed[row][0]))
    if order == list(range(len(order))):
        return data
    return _keep_rows(data, order, of=len(order))


def _is_absent(value: Any) -> bool:
    """Whether a value is nothing: empty, or the ``NOT_FOUND`` mark."""
    return not value or str(value).strip().upper() == NOT_FOUND or not str(value).strip()


def _mean(values: list[float]) -> float:
    """The mean, added up one value at a time.

    Not ``sum()``: from Python 3.12 that corrects for rounding as it goes, and gives a last digit that
    differs from the scores every earlier run recorded.
    """
    total = 0.0
    for value in values:
        total += value
    return total / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    return (ordered[middle - 1] + ordered[middle]) / 2 if len(ordered) % 2 == 0 else ordered[middle]


def score_page(
    read: dict[str, Any],
    answer: dict[str, Any],
    document_type: str,
    *,
    fields_of_type: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """A page's fields scored against its answer.

    Args:
        read: The page's fields as the clean stage left them.
        answer: The page's answer from the key.
        document_type: The page's document type.
        fields_of_type: For a CSV key only: the fields of each document type, by lower-case type. A CSV
            row has every type's columns, so the fields to score come from here. With None, the fields
            scored are the answer's own.

    Returns:
        The record the evaluate stage writes for the page, without its name and type.
    """
    if fields_of_type is None:
        scored_fields = [
            name for name in answer if name not in ("filename", "image_name", "image_file", "file")
        ]
    else:
        scored_fields = fields_of_type.get(document_type.lower(), fields_of_type.get("invoice", []))
    truth = {field: answer[field] for field in scored_fields if field in answer}

    # Credit rows go before the date ordering, so that the ordering sees lists already in step.
    read = _in_date_order(_without_credit_rows(read, document_type), document_type)
    truth = _in_date_order(_without_credit_rows(truth, document_type), document_type)

    field_scores = {field: score_field(read.get(field, NOT_FOUND), truth[field], field) for field in truth}
    count = len(field_scores)
    mean_f1 = _mean([score["f1_score"] for score in field_scores.values()])
    precision = _mean([score["precision"] for score in field_scores.values()])
    recall = _mean([score["recall"] for score in field_scores.values()])
    median_f1 = _median([score["f1_score"] for score in field_scores.values()])
    perfect = sum(1 for score in field_scores.values() if score["f1_score"] == 1.0)
    # A field is missing when it has an answer and nothing was read for it, and incorrect when it
    # was read and scores less than 1. With the perfect fields, the three make up every field scored.
    missing = [
        field
        for field, score in field_scores.items()
        if _is_absent(read.get(field)) and not _is_absent(truth[field])
    ]
    incorrect = [
        field for field, score in field_scores.items() if score["f1_score"] < 1.0 and field not in missing
    ]

    return {
        "overall_accuracy": mean_f1,
        "median_f1": median_f1,
        "overall_precision": precision,
        "overall_recall": recall,
        "total_fields": count,
        "correct_fields": perfect,
        "missing_fields": missing,
        "incorrect_fields": incorrect,
        "fields_extracted": len([name for name, value in read.items() if value != NOT_FOUND]),
        "fields_matched": perfect,
        "field_scores": field_scores,
        "overall_metrics": {
            "overall_accuracy": mean_f1,
            "median_f1": median_f1,
            "overall_precision": precision,
            "overall_recall": recall,
            "meets_threshold": median_f1 >= 0.8,
            "document_type_threshold": 0.8,
        },
    }
