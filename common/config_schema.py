"""Every setting the pipeline knows, so that one it does not know is refused.

A settings file that carries a key no code reads cannot be trusted to say what a run does: the key looks
like an instruction and is ignored. So the keys are declared here, once, and a file is checked against
them before any work begins. A key that is not declared stops the run.

This is the one place the keys are named. It holds no values and no rules about values: those checks stay
with the code that reads each setting.

One mark stands for a level whose name is data and not a setting: ``*`` is one level of any name, such
as a model's name under ``bootstrap.model.default_paths``.
"""

from pathlib import Path
from typing import Any

KNOWN_KEYS: frozenset[str] = frozenset(
    {
        "bootstrap.logging.log_dir",
        "bootstrap.model.default_paths.*",
        "bootstrap.model.type",
        "inference.tracing.path",
        "inference.tracing.raw_prompts",
        "pipeline.classification.fallback_type",
        # A dataset by any name, each with its two keys.
        "pipeline.datasets.*.document_type",
        "pipeline.datasets.*.image_dir",
        "pipeline.extraction.order",
        "pipeline.extraction.skip_labels",
        "pipeline.information_extraction.input.dir",
        "pipeline.information_extraction.input.document_types",
        "pipeline.information_extraction.input.ground_truth",
        "pipeline.information_extraction.input.max_images",
        "pipeline.information_extraction.output.dir",
        "pipeline.photo_split.enabled",
    }
)


class ConfigError(ValueError):
    """The settings cannot be used. The message says what, where, what is expected and the fix."""


def leaf_paths(raw: dict[str, Any], prefix: str = "") -> list[str]:
    """The dotted path of every leaf in a settings mapping. An empty mapping is itself a leaf."""
    paths: list[str] = []
    for key, value in raw.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            paths.extend(leaf_paths(value, f"{path}."))
        else:
            paths.append(path)
    return paths


def is_known(path: str) -> bool:
    """Whether a leaf's dotted path matches a declared key."""
    parts = path.split(".")
    return any(
        _matches(parts, known.split(".")) or _is_a_block_of(parts, known.split(".")) for known in KNOWN_KEYS
    )


def _is_a_block_of(parts: list[str], known: list[str]) -> bool:
    """Whether a path is a block that a declared key lies under.

    An empty mapping is a leaf, so a block written with nothing in it (``datasets: {}``) is a path that
    stops short of every declared key. It is known as the block it is: what must be in it is for
    whatever reads the block to say.
    """
    return len(parts) < len(known) and all(k in ("*", p) for k, p in zip(known, parts, strict=False))


def _matches(parts: list[str], known: list[str]) -> bool:
    return len(parts) == len(known) and all(k in ("*", p) for k, p in zip(known, parts, strict=True))


def check_known_keys(raw: dict[str, Any], *, config_path: Path) -> None:
    """Stop the run if the settings hold a key the pipeline does not know.

    Args:
        raw: The settings file as loaded.
        config_path: Where it was loaded from, for the message.

    Raises:
        ConfigError: Naming every unknown key, where it is, what is expected and how to mend it.
    """
    unknown = [path for path in leaf_paths(raw) if not is_known(path)]
    if not unknown:
        return
    listed = "\n".join(f"    {path}" for path in unknown)
    section = unknown[0].rsplit(".", 1)[0]
    examples = sorted(key for key in KNOWN_KEYS if key.startswith(f"{section}."))[:6]
    expected = "\n".join(f"    {key}" for key in examples) or "    (no setting is known under that section)"
    raise ConfigError(
        f"What: the settings file holds {len(unknown)} key(s) the pipeline does not know:\n{listed}\n"
        f"Where: {config_path} -> {unknown[0]}\n"
        f"Expected: only keys declared in common/config_schema.py. Under '{section}' these are known:\n"
        f"{expected}\n"
        f"How to fix: remove the key(s) from {config_path}. A key from an older settings file names "
        f"something that has been removed from the pipeline, so it has no effect to keep. If the key is "
        f"misspelt, correct it to one of the known keys."
    )
