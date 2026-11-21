"""Many more scoring cases than the recorded runs hold, made from them.

The recorded runs are mostly right answers, so they reach little of the scorer's handling of wrong ones.
Here each recorded page is also scored against the answer of a different page of its type, and against
its own answer after its values have been damaged in set ways. The scores the scorer gives are written
down once, from code known to be good, and compared ever after.

The cases are made from evaluation data, so the record lives outside the repository, beside the recorded
runs. To write it:

    python tests/replay/wide_scoring.py
"""

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

DATA = Path.home() / "Desktop" / "evaluation_data"
REFERENCE = DATA / "refactor_reference_20261120"
RECORD = REFERENCE / "wide_scoring.json"
KEYS = {
    "output_gemma31b_all165_tensor_split_20261120": DATA / "snapshot_20261021" / "plain_20261021",
    "output_gemma12b_w4a16_all165_batched_20261120": DATA / "snapshot_20261021" / "plain_20261021",
    "output_gemma31b_crumpled_split_step_20261120": DATA / "crumpled_20261021",
}
NOT_A_FIELD = {"filename", "image_name", "image_file", "file"}


def damaged(value: str) -> Iterator[tuple[str, str]]:
    """A value spoilt in the ways a model spoils one: (what was done, the result)."""
    yield "missing", "NOT_FOUND"
    yield "empty", ""
    yield "upper", value.upper()
    yield "no_currency", value.replace("$", "")
    yield "no_commas", value.replace(",", "")
    yield "spaced", f"  {value}  "
    yield "quoted", f'"{value}"'
    yield "first_half", value[: max(1, len(value) // 2)]
    items = [item.strip() for item in value.split("|")]
    if len(items) > 1:
        yield "one_short", " | ".join(items[:-1])
        yield "one_long", " | ".join([*items, items[0]])
        yield "reversed", " | ".join(reversed(items))
        yield "first_blank", " | ".join(["NOT_FOUND", *items[1:]])
        yield "swapped", " | ".join([items[1], items[0], *items[2:]])
        yield "commas", ", ".join(items)


def pages() -> Iterator[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Every recorded page with its cleaned record and its answer."""
    for run, key_folder in KEYS.items():
        answers = {}
        for line in (key_folder / "ground_truth.jsonl").read_text().splitlines():
            answer = json.loads(line)
            answers[
                next(
                    answer[name]
                    for name in ("filename", "image_name", "image_file", "file")
                    if name in answer
                )
            ] = answer
        for line in (REFERENCE / run / "cleaned_extractions.jsonl").read_text().splitlines():
            record = json.loads(line)
            answer = answers.get(record["image_name"]) or next(
                (a for name, a in answers.items() if Path(name).stem == Path(record["image_name"]).stem),
                None,
            )
            if answer is not None and not record.get("error"):
                yield run, record, answer


def cases() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The cases: single fields, and whole pages."""
    from common import scoring

    fields: dict[tuple[str, str, str], dict[str, Any]] = {}
    whole: list[dict[str, Any]] = []
    keys = {run: scoring.load_answer_key(folder / "ground_truth.jsonl") for run, folder in KEYS.items()}
    by_type: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for run, record, answer in pages():
        by_type.setdefault((run, record["document_type"]), []).append((record, answer))

    def score(field: str, read: Any, truth: Any) -> None:
        key = (field, json.dumps(read), json.dumps(truth))
        if key not in fields:
            fields[key] = {
                "field": field,
                "read": read,
                "truth": truth,
                "scores": scoring.score_field(read, truth, field),
            }

    for (run, _kind), members in sorted(by_type.items()):
        for index, (record, answer) in enumerate(members):
            other_record, other_answer = members[(index + 1) % len(members)]
            for name, truth in answer.items():
                if name in NOT_A_FIELD:
                    continue
                read = record["extracted_data"].get(name, "NOT_FOUND")
                score(name, read, truth)
                score(name, other_record["extracted_data"].get(name, "NOT_FOUND"), truth)
                if index % 5 == 0:
                    for _what, spoilt in damaged(str(truth)):
                        score(name, spoilt, truth)
                        score(name, truth, spoilt)
            # The page read as it was, scored as a whole against a different page's answer.
            against = scoring.find_answer(
                keys[run], other_answer.get("filename", other_record["image_name"])
            )
            result = scoring.score_page(dict(record["extracted_data"]), against, record["document_type"])
            for name in ("missing_fields", "incorrect_fields"):
                if name in result:
                    result[name] = sorted(result[name])
            whole.append(
                {
                    "run": run,
                    "read": record["image_name"],
                    "against": other_record["image_name"],
                    "result": result,
                }
            )
    return list(fields.values()), whole


def main() -> None:
    if RECORD.exists():
        raise SystemExit(f"A record already exists at {RECORD}. Move it away first.")
    fields, whole = cases()
    again_fields, again_whole = cases()
    if (fields, whole) != (again_fields, again_whole):
        raise SystemExit("Two makings of the cases disagree, so the record would prove nothing.")
    RECORD.write_text(json.dumps({"fields": fields, "whole": whole}, indent=1, sort_keys=True) + "\n")
    print(f"{len(fields)} single fields and {len(whole)} whole pages written to {RECORD}")


if __name__ == "__main__":
    main()
