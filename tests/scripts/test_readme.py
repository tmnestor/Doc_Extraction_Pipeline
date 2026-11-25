"""The README's lists of tasks, models and variables are the ones the code has, so that it cannot drift."""

import re
from pathlib import Path

import models.registry as registry
from common import stage_settings

REPO = Path(__file__).resolve().parents[2]
README = (REPO / "README.md").read_text()


def first_column(heading: str) -> list[str]:
    """The backticked names in the first column of the first table under a heading."""
    section = README.split(f"\n## {heading}\n", 1)[1].split("\n## ", 1)[0]
    names = [re.match(r"\| `([^`]+)` \|", line) for line in section.splitlines()]
    return [found.group(1) for found in names if found]


def test_the_tasks_listed_are_the_tasks_there_are() -> None:
    assert first_column("The six tasks") == list(stage_settings.TASKS)


def test_the_models_listed_are_the_models_registered() -> None:
    assert sorted(first_column("The two models")) == registry.list_models()


def test_every_variable_a_job_may_set_is_in_the_table() -> None:
    listed = first_column("What a job may set")
    header = (
        (REPO / "entrypoint.sh")
        .read_text()
        .split("# The variables a job may set:")[1]
        .split("# By hand")[0]
    )
    in_script = re.findall(r"^#   (\w+)", header, flags=re.MULTILINE)
    assert len(in_script) == 11, in_script
    assert listed == in_script


def test_the_settings_keys_in_the_table_are_the_ones_the_code_reads() -> None:
    sources = (
        stage_settings._MODEL,
        stage_settings._IMAGE_DIR,
        stage_settings._OUTPUT,
        stage_settings._GROUND_TRUTH,
        stage_settings._LOG_DIR,
    )
    for variable, key in sources:
        row = next(line for line in README.splitlines() if line.startswith(f"| `{variable}` |"))
        assert f"`{key}`" in row, (variable, key)


# Words the flowchart grammar keeps for itself. A node named by one of them stops the whole diagram
# being drawn: on 25 November 2026 a node named `call` left a parse error on the page in its place.
MERMAID_KEEPS = {
    "call",
    "class",
    "classdef",
    "click",
    "default",
    "direction",
    "end",
    "flowchart",
    "graph",
    "href",
    "interpolate",
    "linkstyle",
    "style",
    "subgraph",
}


def mermaid_node_names(text: str) -> set[str]:
    """The names of the nodes in the flowchart diagrams of a Markdown text."""
    names: set[str] = set()
    for diagram in re.findall(r"```mermaid\n(.*?)\n```", text, flags=re.DOTALL):
        for line in diagram.splitlines()[1:]:
            # A name is a word at the start of a line, or one that an arrow or its label leads to.
            names.update(re.findall(r"(?:^\s*|[->.]+(?:\|[^|]*\|)?\s*)([A-Za-z_][A-Za-z0-9_]*)\b", line))
    return names


def test_no_node_of_a_diagram_is_named_by_a_word_the_diagram_language_keeps() -> None:
    names = mermaid_node_names(README)

    assert len(names) >= 14, "the diagrams were not found"
    assert not {name for name in names if name.lower() in MERMAID_KEEPS}


def test_the_check_of_node_names_finds_the_name_that_broke_the_diagram() -> None:
    broken = '```mermaid\nflowchart TD\n    call["Pages"]\n    call -->|answered| record\n    record --> end\n```'

    assert mermaid_node_names(broken) == {"call", "record", "end"}
