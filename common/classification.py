"""From the model's reply to the classification prompt, a page's document type.

The prompt asks four numbered questions: the column headings of the page's table, whether the page
shows a payment, how many rows it has, and whether it is a travel document. The type is worked out
from the answers by the rules in ``prompts/document_type_detection.yaml`` (``classification_evidence``),
not taken from a type the model names.

A reply may give its answers three ways, all read here:

- with the labels: ``1. COLUMNS: Date | Debit | Balance`` ... ``4. TRAVEL: NO``;
- by position alone: ``1. Date | Debit | Balance`` / ``2. NO`` / ``3. 14`` / ``4. NO``;
- as markdown headings with each answer on a later line.

A reply in none of these forms gives the page the fallback type.
"""

import re
from functools import lru_cache
from typing import Any

from common.prompt_catalog import PromptCatalog

# A label may follow markdown, a number or a bullet: "### 1. COLUMNS", "- PAID: YES".
_LABEL_PREFIX = r"^[#>\-*\d.\)\s]*"
_COLUMNS = re.compile(rf"{_LABEL_PREFIX}COLUMNS\**\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
_PAID = re.compile(rf"{_LABEL_PREFIX}PAID\**\s*:\s*(YES|NO)\b", re.MULTILINE | re.IGNORECASE)
_TRAVEL = re.compile(rf"{_LABEL_PREFIX}TRAVEL\**\s*:\s*(YES|NO)\b", re.MULTILINE | re.IGNORECASE)
# The TRAVEL label as a heading or numbered item, for a reply whose answer is on a later line. The
# prefix is required so that the word in a sentence ("not a travel document") is not taken for it.
_TRAVEL_HEADING = re.compile(r"(?:^|\n)\s*(?:#{1,6}\s*\d*\.?\s*|\d+\s*[.\)]\s*)TRAVEL\b", re.IGNORECASE)
# The question says "Answer YES or NO"; a reply that repeats it has not thereby answered.
_YES_OR_NO_ECHO = re.compile(r"yes\s+or\s+no|no\s+or\s+yes", re.IGNORECASE)
_YES_NO_WORD = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)
_COLUMNS_WORD = re.compile(r"\bCOLUMNS\b", re.IGNORECASE)
# A reasoning model's thoughts, closed or cut off, are not part of its answer.
_THOUGHTS = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THOUGHTS_UNFINISHED = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)
_THOUGHT_TAG = re.compile(r"</?think>", re.IGNORECASE)
# Answers by position: only 1 to 4, so that a longer numbered list is not read as answers.
_NUMBERED = re.compile(r"^\s*([1-4])\s*[.)]\s*(.+?)\s*$", re.MULTILINE)
_ONLY_YES_OR_NO = re.compile(r"^(YES|NO)$", re.IGNORECASE)
_STARTS_WITH_A_NUMBER = re.compile(r"^(\d+)\b")


@lru_cache(maxsize=1)
def _settings() -> tuple[dict[str, list[str]], dict[str, Any]]:
    """The column roles and the rules, read and checked once."""
    catalog = PromptCatalog()
    return catalog.get_column_roles(), catalog.get_classification_evidence()


def document_type(reply: str, fallback_type: str) -> str:
    """The document type a classification reply stands for.

    Args:
        reply: The model's reply to the classification prompt.
        fallback_type: The type of a page whose reply says nothing that can be read.
    """
    from_answers = _type_from_answers(reply)
    return from_answers if from_answers is not None else fallback_type


def _type_from_answers(reply: str) -> str | None:
    """The type the rules give the reply's answers, or None if it has no answers or no rule fits."""
    text = _THOUGHT_TAG.sub("", _THOUGHTS_UNFINISHED.sub("", _THOUGHTS.sub("", reply))).strip()

    if _COLUMNS_WORD.search(text) is not None:
        columns_found = _COLUMNS.search(text)
        columns = _bare(columns_found.group(1)) if columns_found else ""
        paid_found, travel_found = _PAID.search(text), _TRAVEL.search(text)
        paid = paid_found.group(1).upper() == "YES" if paid_found else False
        travel = travel_found.group(1).upper() == "YES" if travel_found else _travel_from_heading(text)
    else:
        by_position = _answers_by_position(text)
        if by_position is None:
            return None
        columns, paid, travel = by_position

    if columns.upper() == "NONE":
        headings: list[str] = []
    elif columns:
        headings = _headings(columns)
    else:
        # The label is there without its answer beside it. Only a line with bars in it is taken for
        # the headings: a sentence must never be, or a receipt's prose becomes a statement's columns.
        headings = next((_headings(line.strip()) for line in text.splitlines() if line.count("|") >= 2), [])

    roles, evidence = _settings()
    present = (
        {role for role, heading in _roles_of(headings, roles).items() if heading} if headings else set()
    )
    for rule in evidence["rules"]:
        if _rule_fits(rule["when"], present, paid, travel):
            return rule["type"]
    default = evidence["default"]
    return None if isinstance(default, str) and default.lower() == "none" else default


def _bare(value: str) -> str:
    return value.strip().strip("*`").strip()


def _answers_by_position(text: str) -> tuple[str, bool, bool] | None:
    """The answers of a reply that numbers them without labels: (columns, paid, travel).

    Strict, because prose can take the shape of a numbered list: all four numbers must be there, the
    second and fourth exactly YES or NO, and the third starting with a number. Otherwise None.
    """
    answers: dict[int, str] = {}
    for found in _NUMBERED.finditer(text):
        # The first of each number counts: a reply that restates its answers cannot change them.
        answers.setdefault(int(found.group(1)), _bare(found.group(2)))
    if set(answers) != {1, 2, 3, 4}:
        return None
    if not _ONLY_YES_OR_NO.match(answers[2]) or not _ONLY_YES_OR_NO.match(answers[4]):
        return None
    if _STARTS_WITH_A_NUMBER.match(answers[3]) is None:
        return None
    return answers[1], answers[2].upper() == "YES", answers[4].upper() == "YES"


def _travel_from_heading(text: str) -> bool:
    """The travel answer of a reply that gives it below a TRAVEL heading: its last YES or NO there."""
    headings = list(_TRAVEL_HEADING.finditer(text))
    if not headings:
        return False
    words = _YES_NO_WORD.findall(_YES_OR_NO_ECHO.sub("", text[headings[-1].end() :]))
    return bool(words) and words[-1].upper() == "YES"


def _headings(text: str) -> list[str]:
    """The column headings in an answer, however they are separated."""
    headings: list[str] = []
    for line in (line.strip() for line in text.split("\n") if line.strip()):
        cleaned = line.lstrip("0123456789.-•* ").strip().replace("**", "").replace("__", "")
        if cleaned.endswith(":"):
            continue
        if len(cleaned) > 20 and (cleaned.count(",") >= 2 or cleaned.count("|") >= 2):
            separator = "|" if cleaned.count("|") >= 2 else ","
            headings.extend(part.strip() for part in cleaned.split(separator) if len(part.strip()) > 1)
        elif 2 < len(cleaned) <= 40:
            headings.append(cleaned)
    return headings


def _roles_of(headings: list[str], roles: dict[str, list[str]]) -> dict[str, str | None]:
    """For each column role (date, debit, balance and so on), the heading that plays it, if any."""
    lowered = [heading.lower() for heading in headings]

    def heading_for(keywords: list[str]) -> str | None:
        for keyword in keywords:
            if keyword in lowered:
                return headings[lowered.index(keyword)]
        for keyword in keywords:
            if len(keyword) > 2:
                for i, heading in enumerate(lowered):
                    if len(heading) <= 20 and keyword in heading:
                        return headings[i]
        return None

    return {role: heading_for(keywords) for role, keywords in roles.items()}


def _rule_fits(when: dict[str, Any], present: set[str], paid: bool, travel: bool) -> bool:
    """Whether a rule's conditions all hold: some of its roles, all of its roles, paid, travel."""
    if "any_roles" in when and not (set(when["any_roles"]) & present):
        return False
    if "all_roles" in when and not set(when["all_roles"]).issubset(present):
        return False
    if "paid" in when and bool(when["paid"]) != paid:
        return False
    return not ("travel" in when and bool(when["travel"]) != travel)
