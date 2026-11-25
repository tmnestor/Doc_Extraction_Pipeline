"""Many more classification replies than the recorded runs hold, made from them.

The recorded replies to the classification prompt are all in the form asked for. Here each is also put
into the other forms a model uses (the answers without their labels, markdown headings with the answer
on a later line, a reasoning block before the answer, plain prose), its yes and no answers are turned
over, and its column headings are exchanged for another page's. The document type the pipeline reads
from every one is written down once, from code known to be good, and compared ever after.

The cases are made from evaluation data, so the record lives outside the repository, beside the recorded
runs. To write it:

    python tests/replay/wide_classification.py
"""

import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

DATA = Path.home() / "Desktop" / "evaluation_data"
REFERENCE = DATA / "refactor_reference_20261120"
RECORD = REFERENCE / "wide_classification.json"
PROSE = (
    "This is a bank statement.",
    "The document appears to be a tax invoice from a supplier.",
    "A receipt from a shop.",
    "It looks like a boarding pass for a flight.",
    "A vehicle logbook page.",
    "I cannot tell what this document is.",
    "statement",
    "INVOICE",
    "",
)


def answers(reply: str) -> list[str]:
    """The four answers of a recorded reply, without their numbers and labels."""
    found = []
    for line in reply.splitlines():
        line = re.sub(r"^\s*\d+\s*[.)]\s*", "", line.strip())
        found.append(re.sub(r"^(COLUMNS|PAID|ROWS|TRAVEL)\s*:\s*", "", line, flags=re.IGNORECASE))
    return (found + ["NONE", "NO", "0", "NO"])[:4]


def forms(columns: str, paid: str, rows: str, travel: str) -> Iterator[tuple[str, str]]:
    """Four answers written in the ways a model writes them: (the form's name, the reply)."""
    labelled = f"1. COLUMNS: {columns}\n2. PAID: {paid}\n3. ROWS: {rows}\n4. TRAVEL: {travel}"
    yield "labelled", labelled
    yield "by_position", f"1. {columns}\n2. {paid}\n3. {rows}\n4. {travel}"
    yield "by_position_brackets", f"1) {columns}\n2) {paid}\n3) {rows}\n4) {travel}"
    yield "labels_alone", f"COLUMNS: {columns}\nPAID: {paid}\nROWS: {rows}\nTRAVEL: {travel}"
    yield "lower_case", labelled.lower()
    yield "bold_labels", f"**COLUMNS**: {columns}\n**PAID**: {paid}\n**ROWS**: {rows}\n**TRAVEL**: {travel}"
    yield (
        "headings",
        f"### 1. COLUMNS\n```\n{columns}\n```\n### 2. PAID\nAnswer YES or NO.\n{paid}\n"
        f"### 3. ROWS\n{rows}\n### 4. TRAVEL\nAnswer YES or NO. The answer is {travel}",
    )
    yield (
        "thinking_first",
        f"<think>There is no table with headers like Date, Debit, Balance.</think>\n{labelled}",
    )
    yield "thinking_unfinished", f"{labelled}\n<think>Now I should check whether Date | Debit | Balance"
    yield (
        "said_twice",
        f"{labelled}\n1. COLUMNS: Date | Description | Debit | Credit | Balance\n2. PAID: NO",
    )
    yield "three_answers", f"1. {columns}\n2. {paid}\n3. {rows}"
    yield "wordy_rows", f"1. {columns}\n2. {paid}\n3. {rows} line items\n4. {travel}"
    yield "not_yes_or_no", f"1. {columns}\n2. It was paid by card\n3. {rows}\n4. {travel}"
    yield "commas", labelled.replace(" | ", ", ")
    yield "no_travel", f"1. COLUMNS: {columns}\n2. PAID: {paid}\n3. ROWS: {rows}"
    yield "cut_short", labelled[: int(len(labelled) * 0.5)]
    yield "preamble", f"Here are my answers.\n\n{labelled}\n\nThis is therefore a document."


def replies() -> list[tuple[str, str]]:
    """Every case: (its name, the reply)."""
    recorded: list[tuple[str, str]] = []
    for run in sorted(REFERENCE.iterdir()):
        file = run / "classifications.jsonl"
        if not file.is_file():
            continue
        for number, line in enumerate(file.read_text().splitlines()):
            record = json.loads(line)
            recorded.append((f"{run.name}/{record['image_name']}", record.get("raw_response", "")))
    cases: list[tuple[str, str]] = [(f"{name}/as_it_was", reply) for name, reply in recorded]
    for number, (name, reply) in enumerate(recorded):
        if number % 12:
            continue
        columns, paid, rows, travel = answers(reply)
        other_columns = answers(recorded[(number + 7) % len(recorded)][1])[0]
        for use_columns, columns_name in ((columns, "own"), (other_columns, "other"), ("NONE", "none")):
            for use_paid in ("YES", "NO"):
                for use_travel in ("YES", "NO"):
                    for form, text in forms(use_columns, use_paid, rows, use_travel):
                        if columns_name != "own" and form not in ("labelled", "by_position", "headings"):
                            continue
                        cases.append((f"{name}/{columns_name}/{use_paid}/{use_travel}/{form}", text))
    cases.extend((f"prose/{number}", text) for number, text in enumerate(PROSE))
    return cases


def read_types(limit: int | None = None) -> list[dict[str, Any]]:
    """What the pipeline reads from every case."""
    from common import classification

    fallback_type = yaml.safe_load((REPO / "config" / "field_definitions.yaml").read_text())[
        "fallback_type"
    ]
    return [
        {
            "case": name,
            "reply": reply,
            "read": {
                "document_type": classification.document_type(reply, fallback_type),
                "confidence": 1.0,
                "raw_response": reply,
                "prompt_used": "detection",
            },
        }
        for name, reply in (replies()[:limit] if limit else replies())
    ]


def main() -> None:
    if RECORD.exists():
        raise SystemExit(f"A record already exists at {RECORD}. Move it away first.")
    first, again = read_types(), read_types(limit=400)
    if first[:400] != again:
        raise SystemExit("Two makings of the cases disagree, so the record would prove nothing.")
    RECORD.write_text(json.dumps(first, indent=1, sort_keys=True) + "\n")
    kinds: dict[str, int] = {}
    for case in first:
        kinds[case["read"]["document_type"]] = kinds.get(case["read"]["document_type"], 0) + 1
    print(f"{len(first)} replies written to {RECORD}: {kinds}")


if __name__ == "__main__":
    main()
