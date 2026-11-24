"""The NOT_FOUND sentinel must be compared as a sentinel, never as a value.

Found 2026-09-22 on the Gemma 4 31B run: eleven receipts scored 0.286-0.400 on
LINE_ITEM_PRICES with predictions BYTE-IDENTICAL to the ground truth. The order-
aware scorer pairs list items by index and hands each pair to
``_list_item_matches``, which dispatches on the field's schema type. A
``NOT_FOUND`` item in a monetary field therefore reached
``_compare_monetary_values``, where the regex stripped it to ``""``,
``float("")`` raised, and the ``except`` returned 0.0. Every correctly
predicted ``NOT_FOUND`` became a false negative (the branch adds ``fn`` on a
mismatched pair but not ``fp`` -- hence precision stayed 1.0 while recall fell
to 1/n).

A placeholder is not a price, a date or a piece of text, whatever the field's
type. The fix decides sentinel-vs-sentinel BEFORE type dispatch, in the one
comparator every list field goes through. These tests pin that, and guard that
real values and the bank-register path are untouched.

All amounts below are invented. Never put evaluation-set values in a test.
"""

import pytest

from common.scoring import list_item_matches as _list_item_matches
from common.scoring import score_field as calculate_field_accuracy_f1

MONETARY_LIST = "LINE_ITEM_PRICES"  # declared under field_types.monetary
BANK_AMOUNTS = "TRANSACTION_AMOUNTS_PAID"  # a transaction_list field


def test_sentinel_pair_matches_for_a_monetary_list_field():
    assert _list_item_matches("NOT_FOUND", "NOT_FOUND", MONETARY_LIST) is True


def test_sentinel_pair_is_case_insensitive():
    # The rest of the scorer compares the sentinel with .upper(); match it.
    assert _list_item_matches("not_found", "NOT_FOUND", MONETARY_LIST) is True
    assert _list_item_matches("Not_Found", "not_found", MONETARY_LIST) is True


@pytest.mark.parametrize(
    ("extracted", "truth"),
    [("NOT_FOUND", "$1.23"), ("$1.23", "NOT_FOUND")],
)
def test_sentinel_against_a_real_value_does_not_match(extracted, truth):
    # Absent-vs-present is a genuine error in either direction.
    assert _list_item_matches(extracted, truth, MONETARY_LIST) is False


def test_real_monetary_values_still_match():
    # Regression guard: the fix sits ABOVE type dispatch and must not change
    # how genuine amounts are compared.
    assert _list_item_matches("$1.23", "$1.23", MONETARY_LIST) is True
    assert _list_item_matches("1.23", "$1.23", MONETARY_LIST) is True
    assert _list_item_matches("$1.23", "$9.99", MONETARY_LIST) is False


def test_sentinel_pair_matches_for_a_transaction_field():
    # The bank register keeps its own comparator and has no sentinel handling
    # either. The check must sit above that dispatch too, so the bank headline
    # can only hold or rise from this change, never fall.
    assert _list_item_matches("NOT_FOUND", "NOT_FOUND", BANK_AMOUNTS) is True
    assert _list_item_matches("NOT_FOUND", "$4.56", BANK_AMOUNTS) is False


def test_identical_list_with_sentinels_scores_perfect_f1():
    # The bug reproduced end to end through the live order_aware_f1 function.
    # Before the fix this returned 0.4: TP=1, FP=0, FN=3 -> P=1.0, R=0.25.
    value = "NOT_FOUND | NOT_FOUND | $1.23 | NOT_FOUND"
    result = calculate_field_accuracy_f1(value, value, MONETARY_LIST)
    assert result["f1_score"] == 1.0
    assert result["fn"] == 0
    assert result["fp"] == 0


def test_a_genuine_mismatch_still_scores_below_one():
    # Guards against over-fixing: a sentinel where the truth has a value is
    # still a miss, so an imperfect list must not become perfect.
    extracted = "NOT_FOUND | NOT_FOUND | $1.23 | NOT_FOUND"
    truth = "NOT_FOUND | $9.99 | $1.23 | NOT_FOUND"
    result = calculate_field_accuracy_f1(extracted, truth, MONETARY_LIST)
    assert result["f1_score"] < 1.0
    assert result["fn"] == 1


def test_the_lists_of_missing_and_incorrect_fields_follow_the_scores() -> None:
    from common.scoring import score_page

    answer = {
        "filename": "page.png",
        "DOCUMENT_TYPE": "RECEIPT",
        "SUPPLIER_NAME": "Harbour Lane Stationers",
        "TOTAL_AMOUNT": "$48.20",
        "INVOICE_DATE": "03/02/2025",
        "GST_AMOUNT": "NOT_FOUND",
    }
    read = {
        "DOCUMENT_TYPE": "RECEIPT",
        "SUPPLIER_NAME": "Harbour Lane Stationers",
        "TOTAL_AMOUNT": "$84.20",
        "INVOICE_DATE": "NOT_FOUND",
        "GST_AMOUNT": "$4.38",
    }

    result = score_page(read, answer, "RECEIPT")

    assert result["missing_fields"] == ["INVOICE_DATE"]
    assert result["incorrect_fields"] == ["TOTAL_AMOUNT", "GST_AMOUNT"]
    assert result["correct_fields"] == 2
    assert result["total_fields"] == 2 + 1 + 2
    for field in result["missing_fields"] + result["incorrect_fields"]:
        assert result["field_scores"][field]["f1_score"] < 1.0


def test_the_answer_key_is_read_from_json_lines_and_scored_on_the_fields_it_carries(tmp_path) -> None:
    import json

    from common.scoring import find_answer, load_answer_key, score_page

    key = tmp_path / "ground_truth.jsonl"
    answer = {"filename": "page_one.png", "DOCUMENT_TYPE": "RECEIPT", "SUPPLIER_NAME": "Wattle Lane Stores"}
    key.write_text(json.dumps(answer) + "\n\n")

    found = find_answer(load_answer_key(key), "/somewhere/page_one.png")
    scored = score_page(
        {"DOCUMENT_TYPE": "RECEIPT", "SUPPLIER_NAME": "Wattle Lane Stores"}, found, "RECEIPT"
    )

    assert sorted(scored["field_scores"]) == ["DOCUMENT_TYPE", "SUPPLIER_NAME"]
    assert scored["overall_accuracy"] == 1.0


@pytest.mark.parametrize("name", ["ground_truth.csv", "ground_truth.json"])
def test_an_answer_key_in_another_format_is_refused(tmp_path, name: str) -> None:
    from common.scoring import load_answer_key

    key = tmp_path / name
    key.write_text("filename,SUPPLIER_NAME\npage_one.png,Wattle Lane Stores\n")

    with pytest.raises(ValueError, match="not a JSON lines file") as refused:
        load_answer_key(key)
    assert str(key) in str(refused.value) and ".jsonl" in str(refused.value)
