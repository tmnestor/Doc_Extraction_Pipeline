"""No tracked file names, or describes, the organisation the pipeline is run for.

The repository is public. A standing rule keeps the organisation's name out of it, and the words that
would identify it as surely as its name does. Until 26 November 2026 the name stood in six tracked lines:
one of the names a vehicle logbook goes by in the schema file, and five lines of the logbook's prompt,
three of them in the text the model is sent.

The name is built here from its parts, so that this file does not hold it either.
"""

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
THE_NAME = "A" + "TO"
# Words that say who it is without the name. Words of the documents themselves ("tax invoice",
# "taxpayer") are what the pipeline reads, and say nothing of who reads them.
DESCRIBES_IT = (
    r"\btax(ation)?\s+(office|authority|agency|department)\b",
    r"\brevenue\s+(office|authority|agency|service)\b",
)


def tracked_text() -> dict[str, str]:
    """Every tracked file that is text, by its path."""
    listed = subprocess.run(
        ["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True, check=True
    ).stdout.split("\n")
    held: dict[str, str] = {}
    for name in filter(None, listed):
        try:
            held[name] = (REPO / name).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue  # an image, or a file deleted and not yet committed
    return held


def lines_that_match(pattern: str, text: dict[str, str]) -> list[str]:
    found = re.compile(pattern, re.IGNORECASE)
    return [
        f"{name}:{number}"
        for name, body in text.items()
        for number, line in enumerate(body.splitlines(), start=1)
        if found.search(line)
    ]


def test_no_tracked_file_holds_the_name() -> None:
    assert lines_that_match(rf"\b{THE_NAME}\b", tracked_text()) == []


def test_no_tracked_file_describes_it() -> None:
    for pattern in DESCRIBES_IT:
        assert lines_that_match(pattern, tracked_text()) == [], pattern


def test_the_search_finds_the_name_where_it_is_and_only_as_a_word() -> None:
    """A search that matched nothing would pass every time: this one is shown a leak, and finds it."""
    leak = {
        "prompts/a.yaml": f"first line\nfor {THE_NAME} claims\n",
        "config/b.yaml": f"      - {THE_NAME.lower()} logbook\n",
        "common/c.py": "an automaton, a tomato and a datum\n",
        # Built from parts, like the name: this file is read by its own search.
        "docs/d.md": "sent to the national Revenue" + " Agency\nand the Tax" + "ation Office\n",
    }

    assert lines_that_match(rf"\b{THE_NAME}\b", leak) == ["prompts/a.yaml:2", "config/b.yaml:1"]
    assert lines_that_match(DESCRIBES_IT[0], leak) == ["docs/d.md:2"]
    assert lines_that_match(DESCRIBES_IT[1], leak) == ["docs/d.md:1"]


def test_the_search_reads_the_files_the_name_was_in() -> None:
    read = tracked_text()

    assert "config/field_definitions.yaml" in read and "prompts/extraction_prompts.yaml" in read
    assert len(read) > 100
