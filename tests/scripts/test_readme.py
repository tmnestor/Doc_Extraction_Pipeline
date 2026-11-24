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
    assert len(in_script) == 10, in_script
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
