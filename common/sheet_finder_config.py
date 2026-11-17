"""Read the sheet finder's settings from ``pipeline.photo_split.finder`` of the run config.

Every setting is required and none is defaulted here: reading the YAML alone must say how the finder is set up.
A missing, misspelt or wrongly typed key stops the run before any photograph is opened, with what is wrong, where,
what a valid block looks like and how to fix it.
"""

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from common.receipt_finder import FinderParams

SECTION = "pipeline.photo_split.finder"

_EXAMPLE = """\
pipeline:
  photo_split:
    finder:
      work_long_edge: 1600
      desk_ring_frac: 0.03
      step_min: 9.0
      ransac_seed: 0
      # ... and every other key listed in config/run_config.yml under pipeline.photo_split.finder"""


class FinderConfigError(ValueError):
    """The sheet finder's settings are missing or malformed."""


def _fail(what: str, *, key: str, config_path: Path, fix: str) -> FinderConfigError:
    where = f"{config_path.resolve()} -> {SECTION}{'.' + key if key else ''}"
    return FinderConfigError(
        f"What: {what}\n"
        f"  Where: {where}\n"
        f"  Expected: every one of {', '.join(f.name for f in dataclasses.fields(FinderParams))}, each a number, "
        f"e.g.\n{_EXAMPLE}\n"
        f"  How to fix: {fix}"
    )


def finder_params_from(section: Any, *, config_path: Path) -> FinderParams:
    """Build the finder's parameters from the parsed ``finder`` mapping.

    Args:
        section: The parsed value of ``pipeline.photo_split.finder``.
        config_path: The file it came from, named in each error.

    Returns:
        The parameters.

    Raises:
        FinderConfigError: The section is not a mapping, or a key is missing, unknown, of the wrong type or out
            of range.
    """
    if not isinstance(section, dict):
        raise _fail(
            f"the {SECTION} section is missing or is not a mapping",
            key="",
            config_path=config_path,
            fix=f"add the {SECTION} block to {config_path.name}, as shipped in the repository's run_config.yml",
        )
    fields = {field.name: field.type for field in dataclasses.fields(FinderParams)}
    unknown = sorted(set(section) - set(fields))
    if unknown:
        raise _fail(
            f"unknown key(s) {unknown}: the finder would ignore them",
            key=unknown[0],
            config_path=config_path,
            fix=f"remove or correct {unknown} under {SECTION} in {config_path.name}",
        )
    values: dict[str, int | float] = {}
    for name, kind in fields.items():
        if name not in section:
            raise _fail(
                f"missing key {name}",
                key=name,
                config_path=config_path,
                fix=f"add `{name}:` under {SECTION} in {config_path.name}",
            )
        value = section[name]
        # bool is an int in Python; a true/false where a number belongs is a mistake, not a number.
        whole = isinstance(value, int) and not isinstance(value, bool)
        if kind is int and not whole:
            raise _fail(
                f"{name} is {value!r}, which is not a whole number",
                key=name,
                config_path=config_path,
                fix=f"set `{name}` under {SECTION} in {config_path.name} to a whole number",
            )
        if kind is float and not (whole or isinstance(value, float)):
            raise _fail(
                f"{name} is {value!r}, which is not a number",
                key=name,
                config_path=config_path,
                fix=f"set `{name}` under {SECTION} in {config_path.name} to a number",
            )
        if name != "ransac_seed" and value <= 0:
            raise _fail(
                f"{name} is {value!r}, which is not above zero",
                key=name,
                config_path=config_path,
                fix=f"set `{name}` under {SECTION} in {config_path.name} to a number above zero",
            )
        values[name] = int(value) if kind is int else float(value)
    if values["mark_block_px"] % 2 == 0 or values["mark_block_px"] < 3:
        raise _fail(
            f"mark_block_px is {section['mark_block_px']!r}, which is not an odd number of 3 or more",
            key="mark_block_px",
            config_path=config_path,
            fix=f"set `mark_block_px` under {SECTION} in {config_path.name} to an odd number, e.g. 31",
        )
    return FinderParams(**values)  # type: ignore[arg-type]


def load_finder_params(config_path: Path) -> FinderParams:
    """Read the finder's parameters from a run config file.

    Args:
        config_path: Path to run_config.yml.

    Returns:
        The parameters.

    Raises:
        FinderConfigError: The file does not exist or its finder section is missing or malformed.
    """
    if not config_path.is_file():
        raise _fail(
            "the config file does not exist",
            key="",
            config_path=config_path,
            fix="pass the path of the run config, e.g. config/run_config.yml",
        )
    raw = yaml.safe_load(config_path.read_text()) or {}
    section = raw
    for part in SECTION.split("."):
        section = section.get(part) if isinstance(section, dict) else None
    return finder_params_from(section, config_path=config_path)
