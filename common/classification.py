"""From the model's reply to the classification prompt, a page's document type.

The prompt asks numbered questions about things that can be seen on the page: the column headings of its
table, and questions answered YES or NO or with a number. The questions are in
``prompts/document_type_detection.yaml`` (``classification.questions``), and the type is worked out from
the answers by the rules in the same file (``classification_evidence``). The type is not taken from a
type the model names, and nothing here names a question or a type.

A reply may give its answers three ways, all read here:

- with the labels: ``1. COLUMNS: Date | Debit | Balance`` ... ``4. TRAVEL: NO``;
- by position alone: ``1. Date | Debit | Balance`` / ``2. NO`` / ``3. 14`` / ``4. NO``;
- as markdown headings with each answer on a later line.

A reply in none of these forms gives the page the fallback type.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import yaml

from common.config_schema import ConfigError
from common.prompt_catalog import PromptCatalog, Question

# A label may follow markdown, a number or a bullet: "### 1. COLUMNS", "- PAID: YES".
_LABEL_PREFIX = r"^[#>\-*\d.\)\s]*"
# The question says "Answer YES or NO"; a reply that repeats it has not thereby answered.
_YES_OR_NO_ECHO = re.compile(r"yes\s+or\s+no|no\s+or\s+yes", re.IGNORECASE)
_YES_NO_WORD = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)
# A reasoning model's thoughts, closed or cut off, are not part of its answer.
_THOUGHTS = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THOUGHTS_UNFINISHED = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)
_THOUGHT_TAG = re.compile(r"</?think>", re.IGNORECASE)
_ONLY_YES_OR_NO = re.compile(r"^(YES|NO)$", re.IGNORECASE)
_STARTS_WITH_A_NUMBER = re.compile(r"^(\d+)\b")


@dataclass(frozen=True)
class _Asked:
    """The questions, with the patterns that find each one's answer in a reply."""

    questions: tuple[Question, ...]
    # The question whose answer is the column headings: its label as a word, and its labelled answer.
    columns_word: re.Pattern[str]
    columns_answer: re.Pattern[str]
    # For each YES or NO question, by its label in lower case: its labelled answer, and its label as a
    # heading or numbered item, for a reply whose answer is on a later line. The heading's prefix is
    # required so that the word in a sentence ("not a travel document") is not taken for it.
    yes_no_answer: dict[str, re.Pattern[str]]
    yes_no_heading: dict[str, re.Pattern[str]]
    # Answers by position: only the numbers of the questions, so that a longer list is not read as answers.
    numbered: re.Pattern[str]


@lru_cache(maxsize=1)
def _settings() -> tuple[_Asked, dict[str, list[str]], dict[str, Any]]:
    """The questions, the column roles and the rules, read and checked once."""
    catalog = PromptCatalog()
    questions = catalog.get_classification_questions()
    columns = next(question.label for question in questions if question.answer == "columns")
    flags = [question.label for question in questions if question.answer == "yes_no"]
    asked = _Asked(
        questions=questions,
        columns_word=re.compile(rf"\b{columns}\b", re.IGNORECASE),
        columns_answer=re.compile(
            rf"{_LABEL_PREFIX}{columns}\**\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE
        ),
        yes_no_answer={
            label.lower(): re.compile(
                rf"{_LABEL_PREFIX}{label}\**\s*:\s*(YES|NO)\b", re.MULTILINE | re.IGNORECASE
            )
            for label in flags
        },
        yes_no_heading={
            label.lower(): re.compile(
                rf"(?:^|\n)\s*(?:#{{1,6}}\s*\d*\.?\s*|\d+\s*[.\)]\s*){label}\b", re.IGNORECASE
            )
            for label in flags
        },
        numbered=re.compile(rf"^\s*([1-{len(questions)}])\s*[.)]\s*(.+?)\s*$", re.MULTILINE),
    )
    return asked, catalog.get_column_roles(), catalog.get_classification_evidence()


def check() -> None:
    """Read and check the questions, the column roles and the rules, before any page is asked about.

    They are otherwise first read when the first reply is turned into a type: after the engine has
    started and the first pages have been asked about.

    Raises:
        ConfigError: If the classification file cannot be read, or holds a block that cannot be used.
    """
    try:
        _settings()
    except (OSError, yaml.YAMLError) as err:
        raise ConfigError(
            f"What: the classification file cannot be read: {err}\n"
            f"Where: prompts/document_type_detection.yaml in the repository.\n"
            f"Expected: a YAML file with the blocks classification, column_roles and "
            f"classification_evidence.\n"
            f"How to fix: restore the file from the repository, or correct what the message names."
        ) from err


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
    asked, roles, evidence = _settings()
    text = _THOUGHT_TAG.sub("", _THOUGHTS_UNFINISHED.sub("", _THOUGHTS.sub("", reply))).strip()

    if asked.columns_word.search(text) is not None:
        columns_found = asked.columns_answer.search(text)
        columns = _bare(columns_found.group(1)) if columns_found else ""
        flags = {label: _labelled_yes_or_no(text, asked, label) for label in asked.yes_no_answer}
    else:
        by_position = _answers_by_position(text, asked)
        if by_position is None:
            return None
        columns, flags = by_position

    if columns.upper() == "NONE":
        headings: list[str] = []
    elif columns:
        headings = _headings(columns)
    else:
        # The label is there without its answer beside it. Only a line with bars in it is taken for
        # the headings: a sentence must never be, or a receipt's prose becomes a statement's columns.
        headings = next((_headings(line.strip()) for line in text.splitlines() if line.count("|") >= 2), [])

    present = (
        {role for role, heading in _roles_of(headings, roles).items() if heading} if headings else set()
    )
    for rule in evidence["rules"]:
        if _rule_fits(rule["when"], present, flags):
            return rule["type"]
    default = evidence["default"]
    return None if isinstance(default, str) and default.lower() == "none" else default


def _bare(value: str) -> str:
    return value.strip().strip("*`").strip()


def _labelled_yes_or_no(text: str, asked: _Asked, label: str) -> bool:
    """A YES or NO question's answer: beside its label, or else the last YES or NO below its heading.

    A question that is not answered either way counts as NO.
    """
    beside = asked.yes_no_answer[label].search(text)
    if beside:
        return beside.group(1).upper() == "YES"
    headings = list(asked.yes_no_heading[label].finditer(text))
    if not headings:
        return False
    below = text[headings[-1].end() :]
    # Only as far as the next question's heading, so that a later question's answer is not taken.
    later = [
        found.start()
        for other, pattern in asked.yes_no_heading.items()
        if other != label
        for found in pattern.finditer(below)
    ]
    words = _YES_NO_WORD.findall(_YES_OR_NO_ECHO.sub("", below[: min(later)] if later else below))
    return bool(words) and words[-1].upper() == "YES"


def _answers_by_position(text: str, asked: _Asked) -> tuple[str, dict[str, bool]] | None:
    """The answers of a reply that numbers them without labels: the columns, and each YES or NO.

    Strict, because prose can take the shape of a numbered list: every question's number must be there,
    a YES or NO question's answer exactly YES or NO, and a number question's starting with a number.
    Otherwise None.
    """
    answers: dict[int, str] = {}
    for found in asked.numbered.finditer(text):
        # The first of each number counts: a reply that restates its answers cannot change them.
        answers.setdefault(int(found.group(1)), _bare(found.group(2)))
    if set(answers) != set(range(1, len(asked.questions) + 1)):
        return None
    columns = ""
    flags: dict[str, bool] = {}
    for number, question in enumerate(asked.questions, start=1):
        answer = answers[number]
        if question.answer == "columns":
            columns = answer
        elif question.answer == "yes_no":
            if not _ONLY_YES_OR_NO.match(answer):
                return None
            flags[question.label.lower()] = answer.upper() == "YES"
        elif _STARTS_WITH_A_NUMBER.match(answer) is None:
            return None
    return columns, flags


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


def _rule_fits(when: dict[str, Any], present: set[str], flags: dict[str, bool]) -> bool:
    """Whether a rule's conditions all hold: some of its roles, all of its roles, each YES or NO it names."""
    if "any_roles" in when and not (set(when["any_roles"]) & present):
        return False
    if "all_roles" in when and not set(when["all_roles"]).issubset(present):
        return False
    return all(
        flags.get(label, False) == bool(wanted)
        for label, wanted in when.items()
        if label not in ("any_roles", "all_roles")
    )
