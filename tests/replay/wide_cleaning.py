"""Many more replies for the clean stage than the recorded runs hold, made from them.

The recorded replies are nearly all well formed, so they reach little of how the clean stage copes
with one that is not. Here each third recorded reply is also spoilt in the ways a model spoils one: cut
short, wrapped in a code fence, given as JSON, with a preamble, with its field names in lower case, and
so on. Each is also cleaned as if it were a page of another document type. What the clean stage makes of
every one is written down once, from code known to be good, and compared ever after.

The cases are made from evaluation data, so the record lives outside the repository, beside the recorded
runs. To write it:

    python tests/replay/wide_cleaning.py
"""

import json
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

DATA = Path.home() / "Desktop" / "evaluation_data"
REFERENCE = DATA / "refactor_reference_20261120"
RECORD = REFERENCE / "wide_cleaning.json"
RUNS = (
    "output_gemma31b_all165_tensor_split_20261120",
    "output_gemma12b_w4a16_all165_batched_20261120",
    "output_gemma31b_crumpled_split_step_20261120",
)
TYPES = ("RECEIPT", "INVOICE", "BANK_STATEMENT")


def as_pairs(reply: str) -> list[tuple[str, str]]:
    pairs = []
    for line in reply.splitlines():
        name, colon, value = line.partition(":")
        if colon and name.strip():
            pairs.append((name.strip(), value.strip()))
    return pairs


def spoilt(reply: str) -> Iterator[tuple[str, str]]:
    """A reply spoilt in set ways: (what was done, the result)."""
    lines = reply.splitlines()
    pairs = as_pairs(reply)
    as_json = json.dumps(dict(pairs), indent=2)
    yield "as_it_was", reply
    yield "cut_short", reply[: int(len(reply) * 0.6)]
    yield "cut_mid_line", reply[: max(1, len(reply) - 7)]
    yield "every_third_line_gone", "\n".join(line for n, line in enumerate(lines) if n % 3 != 2)
    yield "lower_case_names", "\n".join(f"{name.lower()}: {value}" for name, value in pairs)
    yield "bold_names", "\n".join(f"**{name}:** {value}" for name, value in pairs)
    yield "bold_whole", "\n".join(f"**{name}**: {value}" for name, value in pairs)
    yield "bullets", "\n".join(f"- {name}: {value}" for name, value in pairs)
    yield "numbered", "\n".join(f"{n}. {name}: {value}" for n, (name, value) in enumerate(pairs, 1))
    yield "code_fence", f"```\n{reply}\n```"
    yield (
        "preamble",
        f"Here is the information from the document:\n\n{reply}\n\nLet me know if you need more.",
    )
    yield "windows_lines", reply.replace("\n", "\r\n")
    yield "blank_lines", reply.replace("\n", "\n\n")
    yield "quoted_values", "\n".join(f'{name}: "{value}"' for name, value in pairs)
    yield "no_space", "\n".join(f"{name}:{value}" for name, value in pairs)
    yield "said_twice", f"{reply}\n{reply}"
    yield "reversed_order", "\n".join(reversed(lines))
    yield "commas_for_bars", reply.replace(" | ", ", ")
    yield "bars_tight", reply.replace(" | ", "|")
    yield "values_on_next_line", "\n".join(f"{name}:\n{value}" for name, value in pairs)
    yield "json", as_json
    yield "json_fenced", f"```json\n{as_json}\n```"
    yield "json_cut_short", as_json[: int(len(as_json) * 0.7)]
    yield "json_one_line", json.dumps(dict(pairs))
    yield "empty", ""
    yield "only_not_found", "NOT_FOUND"
    yield "prose", "I am unable to read this document clearly."


def cases() -> list[dict[str, Any]]:
    """Every case with what the clean stage makes of it."""
    from stages import clean

    raw: list[dict[str, Any]] = []
    for run in RUNS:
        lines = (REFERENCE / run / "raw_extractions.jsonl").read_text().splitlines()
        for number, line in enumerate(lines):
            record = json.loads(line)
            if record.get("error") or number % 3:
                continue
            for what, reply in spoilt(record.get("raw_response", "")):
                for kind in dict.fromkeys((record["document_type"], *TYPES)):
                    if kind != record["document_type"] and what not in ("as_it_was", "cut_short", "json"):
                        continue
                    raw.append(
                        {
                            "image_name": f"{run}/{record['image_name']}/{what}/{kind}",
                            "image_path": record.get("image_path", ""),
                            "document_type": kind,
                            "raw_response": reply,
                            "processing_time": 0.0,
                            "error": None,
                        }
                    )
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "raw_extractions.jsonl"
        source.write_text("".join(json.dumps(record) + "\n" for record in raw))
        cleaned = clean.run(source, Path(folder) / "cleaned_extractions.jsonl")
        return [json.loads(line) for line in cleaned.read_text().splitlines()]


def main() -> None:
    if RECORD.exists():
        raise SystemExit(f"A record already exists at {RECORD}. Move it away first.")
    first, again = cases(), cases()
    if first != again:
        raise SystemExit("Two makings of the cases disagree, so the record would prove nothing.")
    RECORD.write_text(json.dumps(first, indent=1, sort_keys=True) + "\n")
    print(f"{len(first)} replies written to {RECORD}")


if __name__ == "__main__":
    main()
