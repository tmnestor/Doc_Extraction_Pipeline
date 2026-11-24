"""Every number the scorer judges by is in the schema file, and the file holds no threshold nothing reads.

Until 25 November 2026 two were written in the code: the score of a date right in its day and month and
wrong in its year, and the median F1 at which a page's record says it meets the threshold. Both were
0.8, and neither could be seen or changed in the file that is said to hold the scoring thresholds. That
file also carried four thresholds nothing read, left from scoring that had been removed: each looked
like a setting and changed nothing. A fifth, the floor below which a text value earns nothing, was in the
file and not read either: the scorer had its own copy of the number.

The dates and amounts are invented.
"""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from common import scoring
from common.field_schema import THRESHOLDS, FieldSchema, FieldSchemaError

REPO = Path(__file__).resolve().parents[2]
SCHEMA = REPO / "config" / "field_definitions.yaml"
PARTS = ("What:", "Where:", "Expected:", "How to fix:")


def schema_with(tmp_path: Path, change: Any) -> Path:
    raw = yaml.safe_load(SCHEMA.read_text())
    change(raw["evaluation"]["thresholds"])
    path = tmp_path / "field_definitions.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def scoring_by(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **thresholds: float) -> None:
    made = FieldSchema.from_yaml(schema_with(tmp_path, lambda given: given.update(thresholds)))
    monkeypatch.setattr(scoring, "get_field_schema", lambda: made)


# --- the names --------------------------------------------------------------------------------------------


def test_the_thresholds_declared_are_the_ones_the_scorer_reads() -> None:
    read = set(re.findall(r'get_threshold\("([a-z_]+)"\)', (REPO / "common" / "scoring.py").read_text()))

    assert read == set(THRESHOLDS)


def test_the_shipped_file_sets_each_of_them_and_no_other() -> None:
    shipped = yaml.safe_load(SCHEMA.read_text())["evaluation"]["thresholds"]

    assert set(shipped) == set(THRESHOLDS)
    assert FieldSchema.from_yaml(SCHEMA).get_threshold("page_meets_threshold_at") == 0.8


def test_the_scorer_has_no_number_of_its_own_to_judge_by() -> None:
    """A share or a cut-off written in the scorer is one that cannot be seen in the file."""
    source = (REPO / "common" / "scoring.py").read_text()
    code = "\n".join(line.split("#")[0] for line in source.splitlines() if '"""' not in line)

    assert re.findall(r"(?<![\w.])0\.\d*[1-9]\d*", code) == []


# --- a file that cannot be used ---------------------------------------------------------------------------


def test_a_threshold_nothing_reads_is_refused(tmp_path: Path) -> None:
    path = schema_with(tmp_path, lambda given: given.update(phone_match_high=0.8, word_overlap_low=0.6))

    with pytest.raises(FieldSchemaError) as refused:
        FieldSchema.from_yaml(path)

    message = str(refused.value)
    assert all(part in message for part in PARTS)
    assert "nothing reads: phone_match_high, word_overlap_low" in message
    assert str(path.absolute()) in message and "evaluation.thresholds.phone_match_high" in message


def test_a_threshold_the_scorer_reads_and_the_file_lacks_is_refused(tmp_path: Path) -> None:
    path = schema_with(tmp_path, lambda given: given.pop("date_day_and_month_only"))

    with pytest.raises(FieldSchemaError) as refused:
        FieldSchema.from_yaml(path)

    message = str(refused.value)
    assert all(part in message for part in PARTS)
    assert "not set: date_day_and_month_only" in message
    assert "evaluation.thresholds.date_day_and_month_only" in message


@pytest.mark.parametrize("value", ["high", True, None, [0.8]])
def test_a_threshold_that_is_not_a_number_is_refused(tmp_path: Path, value: Any) -> None:
    path = schema_with(tmp_path, lambda given: given.update(monetary_tolerance=value))

    with pytest.raises(FieldSchemaError) as refused:
        FieldSchema.from_yaml(path)

    assert all(part in str(refused.value) for part in PARTS)
    assert "monetary_tolerance" in str(refused.value) and "not a number" in str(refused.value)


def test_a_file_with_no_thresholds_at_all_is_refused(tmp_path: Path) -> None:
    raw = yaml.safe_load(SCHEMA.read_text())
    del raw["evaluation"]
    path = tmp_path / "field_definitions.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))

    with pytest.raises(FieldSchemaError) as refused:
        FieldSchema.from_yaml(path)

    assert all(part in str(refused.value) for part in PARTS)
    assert "evaluation.thresholds" in str(refused.value)


# --- the scorer judges by the file ------------------------------------------------------------------------


@pytest.mark.parametrize("share", [0.8, 0.5, 0.25])
def test_a_date_right_in_day_and_month_scores_the_share_the_file_gives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, share: float
) -> None:
    scoring_by(tmp_path, monkeypatch, date_day_and_month_only=share)

    assert scoring._date_field_score("09/02/2031", "09/02/2030") == share
    # A date that is the same, and one that shares nothing, are not shares of anything.
    assert scoring._date_field_score("09/02/2031", "09/02/2031") == 1.0
    assert scoring._date_field_score("09/02/2031", "17/11/2030") == 0.0


@pytest.mark.parametrize(("floor", "scores"), [(0.5, True), (0.95, False)])
def test_a_text_value_earns_nothing_below_the_floor_the_file_gives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, floor: float, scores: bool
) -> None:
    scoring_by(tmp_path, monkeypatch, anls_threshold=floor)
    # One letter of eighteen differs: a similarity of about 0.94.
    scored = scoring.score_field("Quillfeather Bookz", "Quillfeather Books", "SUPPLIER_NAME")

    assert (scored["f1_score"] > 0.9) is scores
    assert scores or scored["f1_score"] == 0.0


def test_an_identifier_is_right_or_wrong_as_it_was() -> None:
    right = scoring.score_field("ABN 51 824 753 556", "51824753556", "BUSINESS_ABN")
    wrong = scoring.score_field("51 824 753 557", "51824753556", "BUSINESS_ABN")

    assert right == {"f1_score": 1.0, "precision": 1.0, "recall": 1.0, "tp": 1, "fp": 0, "fn": 0}
    assert wrong == {"f1_score": 0.0, "precision": 0.0, "recall": 0.0, "tp": 0, "fp": 1, "fn": 1}


@pytest.mark.parametrize(("meets_at", "meets"), [(0.8, True), (1.0, False), (0.4, True)])
def test_a_pages_record_says_it_meets_the_threshold_the_file_gives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, meets_at: float, meets: bool
) -> None:
    scoring_by(tmp_path, monkeypatch, page_meets_threshold_at=meets_at)
    # Three fields of four are right: the median F1 of the page's fields is 1, its mean 0.75.
    answer = {
        "filename": "page_1.png",
        "SUPPLIER_NAME": "Quillfeather Books",
        "PAYER_NAME": "Marigold Tansley",
        "INVOICE_DATE": "09/02/2031",
        "TOTAL_AMOUNT": "$31.07",
    }
    read = {**answer, "TOTAL_AMOUNT": "$99.99", "INVOICE_DATE": "17/11/2030"}
    read.pop("filename")

    scored = scoring.score_page(read, answer, "INVOICE")

    assert scored["overall_metrics"]["document_type_threshold"] == meets_at
    assert scored["overall_metrics"]["meets_threshold"] is (scored["median_f1"] >= meets_at)
    if meets_at == 1.0:
        assert scored["overall_metrics"]["meets_threshold"] is meets or scored["median_f1"] == 1.0
