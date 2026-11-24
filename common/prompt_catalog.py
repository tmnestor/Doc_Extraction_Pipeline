"""Unified prompt loading -- single entry point for all YAML prompt access.

Consolidates SimplePromptLoader, ConfigLoader, and inline YAML parsing
behind a single class with cached file reads.

Usage:
    from common.prompt_catalog import PromptCatalog

    catalog = PromptCatalog()
    prompt = catalog.get_prompt("gemma4-31b-w4a16-vllm", document_type)
    classification_prompt = catalog.classification_prompt()
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_DETECTION_FILE = "document_type_detection.yaml"
ANSWER_KINDS = ("columns", "yes_no", "number")


@dataclass(frozen=True)
class Question:
    """One question of the classification prompt."""

    label: str
    answer: str
    ask: str


class PromptCatalog:
    """Unified prompt loading -- single entry point for all YAML prompt access.

    All YAML files live in one directory (``prompts/``).  Namespaces map to
    filenames: ``"detection"`` -> ``document_type_detection.yaml``,
    or any registered model_type ->
    its ``VllmSpec.prompt_file``.

    Files are parsed once and cached per instance.

    Args:
        prompts_dir: Directory containing all prompt YAML files.
            Default: auto-detected from this file's location.
    """

    def __init__(self, prompts_dir: Path | None = None) -> None:
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent / "prompts"
        self._dir = prompts_dir
        self._cache: dict[Path, dict] = {}

    # -- Internal helpers ------------------------------------------------------

    def _load_yaml(self, path: Path) -> dict:
        """Load and cache a YAML file."""
        if path in self._cache:
            return self._cache[path]
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}\nExpected location: {path.absolute()}")
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self._cache[path] = data
        return data

    def _resolve_filename(self, namespace: str) -> str:
        """Resolve a namespace string to a YAML filename."""
        from models.registry import get_model

        registration = get_model(namespace)
        return registration.prompt_file

    # -- Entry point 1: Load a single prompt -----------------------------------

    def get_prompt(
        self,
        namespace: str,
        key: str,
        *,
        format_args: dict[str, str] | None = None,
    ) -> str:
        """Load a prompt string by namespace and key.

        Args:
            namespace: A registered model_type, which maps to its prompt_file.
            key: Prompt key within the YAML file's ``prompts`` section.
            format_args: Optional dict for ``str.format()`` on the template.
                Used by bank prompts with placeholders like ``{balance_col}``.

        Returns:
            The prompt text string, ready to send to the model.

        Raises:
            FileNotFoundError: YAML file for namespace not found.
            KeyError: Prompt key not found (with available keys).
        """
        filename = self._resolve_filename(namespace)
        path = self._dir / filename
        data = self._load_yaml(path)

        prompts = data.get("prompts", {})
        if key not in prompts:
            available = sorted(prompts.keys())
            raise KeyError(f"Prompt '{key}' not found in {filename}. Available: {', '.join(available)}")

        entry = prompts[key]
        # Normalize: extraction YAMLs use "prompt", bank uses "template"
        text = entry.get("prompt") or entry.get("template", "")

        if format_args:
            text = text.format(**format_args)
        return text

    # -- Entry point 2: the classification questions and rules ----------------

    def get_classification_questions(self) -> tuple[Question, ...]:
        """The questions the classification prompt asks, in order, checked.

        Raises:
            ValueError: If the ``classification`` block is missing or a question cannot be used.
        """
        path = self._dir / _DETECTION_FILE
        block = self._load_yaml(path).get("classification")
        example = (
            'classification:\n  opening: "Look at this document image and answer each question:"\n'
            "  questions:\n    - label: PAID\n      answer: yes_no\n      ask: |\n"
            "        Is there evidence that payment was completed? Answer YES or NO."
        )

        def refuse(what: str, key: str, fix: str) -> ValueError:
            return ValueError(
                f"What: {what}\nWhere: {path.absolute()} (key: {key}).\n"
                f"Expected: a block like this, with one question of the kind `columns`:\n{example}\n"
                f"How to fix: {fix}"
            )

        listed = block.get("questions") if isinstance(block, dict) else None
        if not isinstance(block, dict) or not isinstance(block.get("opening"), str) or not listed:
            raise refuse(
                "the classification block has no opening line or no questions.",
                "classification",
                "give the block an `opening:` line and a `questions:` list.",
            )
        if not isinstance(listed, list) or len(listed) > 9:
            raise refuse(
                "the questions are not a list of at most nine.",
                "classification.questions",
                "write the questions as a list. A reply numbers its answers 1 to 9 at most.",
            )
        questions: list[Question] = []
        for number, item in enumerate(listed):
            key = f"classification.questions[{number}]"
            label = item.get("label") if isinstance(item, dict) else None
            if not isinstance(label, str) or not label.isidentifier() or label != label.upper():
                raise refuse(
                    f"question {number + 1} has no label in capitals: {label!r}.",
                    f"{key}.label",
                    "give it a one-word label in capitals, e.g. label: PAID",
                )
            if item.get("answer") not in ANSWER_KINDS:
                raise refuse(
                    f"the question {label} has the answer kind {item.get('answer')!r}.",
                    f"{key}.answer",
                    f"set `answer:` to one of {', '.join(ANSWER_KINDS)}.",
                )
            if not isinstance(item.get("ask"), str) or not item["ask"].strip():
                raise refuse(
                    f"the question {label} has no text.", f"{key}.ask", "write the question under `ask:`."
                )
            if label in [question.label for question in questions]:
                raise refuse(
                    f"two questions have the label {label}.",
                    f"{key}.label",
                    "give each question its own label.",
                )
            questions.append(Question(label=label, answer=item["answer"], ask=item["ask"]))
        if [question.answer for question in questions].count("columns") != 1:
            raise refuse(
                "there is not exactly one question of the kind `columns`.",
                "classification.questions",
                "have one question ask for the column headings of the page's table, with `answer: columns`.",
            )
        return tuple(questions)

    def classification_prompt(self) -> str:
        """The classification prompt: the opening line, then the questions numbered in order.

        A question's first line follows its number and label; its later lines are set under it.
        """
        opening = self._load_yaml(self._dir / _DETECTION_FILE)["classification"]["opening"]
        parts = []
        for number, question in enumerate(self.get_classification_questions(), start=1):
            first, *later = question.ask.rstrip("\n").split("\n")
            parts.append(
                "\n".join([f"{number}. {question.label}: {first}", *(f"   {line}" for line in later)])
            )
        return opening + "\n\n" + "\n\n".join(parts) + "\n"

    def get_column_roles(self) -> dict[str, list[str]]:
        """Load semantic column-role -> header-keyword lists from detection YAML.

        Read by ``common.classification``.

        Returns:
            Mapping of role name to its list of header keywords.

        Raises:
            ValueError: If the ``column_roles`` block is missing or malformed.
        """
        path = self._dir / _DETECTION_FILE
        data = self._load_yaml(path)
        roles = data.get("column_roles")
        if not isinstance(roles, dict) or not roles:
            raise ValueError(
                "Missing or empty 'column_roles' block in "
                f"{path.absolute()} (key: column_roles).\n"
                "Expected a non-empty mapping of role -> keyword list, e.g.:\n"
                "  column_roles:\n"
                "    debit:\n      - debit\n      - withdrawal\n"
                "Fix: add the column_roles: section to document_type_detection.yaml."
            )
        for role, keywords in roles.items():
            if not isinstance(keywords, list) or not keywords:
                raise ValueError(
                    f"Role '{role}' in 'column_roles' must be a non-empty list of "
                    f"keyword strings.\n"
                    f"Where: {path.absolute()} (key: column_roles.{role}).\n"
                    f"Found: {keywords!r}.\n"
                    f"Fix: give '{role}' at least one keyword, e.g. "
                    f"`{role}:\n  - <header keyword>`."
                )
        return roles

    def get_classification_evidence(self) -> dict[str, Any]:
        """Load the evidence-based classification rule set from detection YAML.

        Read by ``common.classification``: an ordered list of
        ``rules`` (first match wins) plus a ``default`` for when none match.

        Returns:
            Dict with keys ``rules`` (list) and ``default`` (str).

        Raises:
            ValueError: If the ``classification_evidence`` block is missing or
                structurally invalid.
        """
        path = self._dir / _DETECTION_FILE
        data = self._load_yaml(path)
        evidence = data.get("classification_evidence")
        if not isinstance(evidence, dict):
            raise ValueError(
                "Missing 'classification_evidence' block in "
                f"{path.absolute()} (key: classification_evidence).\n"
                "Expected a mapping with 'rules' and 'default', e.g.:\n"
                "  classification_evidence:\n"
                "    rules:\n"
                "      - type: BANK_STATEMENT\n"
                "        when: { any_roles: [debit, credit, balance] }\n"
                "    default: none\n"
                "Fix: add the classification_evidence: section to "
                "document_type_detection.yaml."
            )
        rules = evidence.get("rules")
        if not isinstance(rules, list) or not rules:
            raise ValueError(
                "'classification_evidence.rules' must be a non-empty list.\n"
                f"Where: {path.absolute()} (key: classification_evidence.rules).\n"
                f"Found: {rules!r}.\n"
                "Fix: add at least one rule with 'type' and a 'when' clause."
            )
        if "default" not in evidence:
            raise ValueError(
                "'classification_evidence.default' is required.\n"
                f"Where: {path.absolute()} (key: classification_evidence.default).\n"
                "Expected 'none' (defer to keyword fallback) or a document-type "
                "string (e.g. INVOICE).\n"
                "Fix: add `default: none` under classification_evidence."
            )
        self._validate_evidence_references(evidence, rules, path)
        return evidence

    def _validate_evidence_references(self, evidence: dict[str, Any], rules: list, path: Path) -> None:
        """Cross-reference rules against column_roles and supported types.

        Ensures every referenced role exists, every `when:` uses known keys,
        and every emitted type (and the default) is a supported document type.
        """
        from common.field_schema import get_field_schema

        known_roles = set(self.get_column_roles())
        supported = set(get_field_schema().document_types)
        asked = {q.label.lower() for q in self.get_classification_questions() if q.answer == "yes_no"}
        allowed_when_keys = {"any_roles", "all_roles"} | asked

        for i, rule in enumerate(rules):
            loc = f"classification_evidence.rules[{i}]"
            if not isinstance(rule, dict) or "type" not in rule or "when" not in rule:
                raise ValueError(
                    f"Each classification rule needs 'type' and 'when'.\n"
                    f"Where: {path.absolute()} ({loc}).\n"
                    f"Found: {rule!r}.\n"
                    "Fix: e.g. `- type: RECEIPT\n    when: { paid: true }`."
                )
            rtype = rule["type"]
            if not isinstance(rtype, str) or rtype.lower() not in supported:
                raise ValueError(
                    f"Rule type '{rtype}' is not a supported document type.\n"
                    f"Where: {path.absolute()} ({loc}.type).\n"
                    f"Allowed: {sorted(supported)}.\n"
                    "Fix: use one of the supported types, or give the new type a "
                    "block in config/field_definitions.yaml (document_types)."
                )
            when = rule["when"]
            if not isinstance(when, dict) or not when:
                raise ValueError(
                    f"Rule 'when:' must be a non-empty mapping.\n"
                    f"Where: {path.absolute()} ({loc}.when).\n"
                    f"Found: {when!r}.\n"
                    "Fix: add a condition, e.g. `when: { any_roles: [debit] }`."
                )
            unknown_keys = set(when) - allowed_when_keys
            if unknown_keys:
                raise ValueError(
                    f"Unknown 'when:' key(s) {sorted(unknown_keys)}.\n"
                    f"Where: {path.absolute()} ({loc}.when).\n"
                    f"Allowed keys: {sorted(allowed_when_keys)}.\n"
                    "Fix: remove the unknown key, correct the spelling, or add a yes_no "
                    "question with that label under classification.questions."
                )
            for role_key in ("any_roles", "all_roles"):
                for role in when.get(role_key, []):
                    if role not in known_roles:
                        raise ValueError(
                            f"Rule references unknown column role '{role}'.\n"
                            f"Where: {path.absolute()} ({loc}.when.{role_key}).\n"
                            f"Known roles: {sorted(known_roles)}.\n"
                            f"Fix: add '{role}' under column_roles, or use a "
                            "known role name."
                        )

        default = evidence["default"]
        if isinstance(default, str) and default.lower() != "none" and default.lower() not in supported:
            raise ValueError(
                f"classification_evidence.default '{default}' is not a supported type.\n"
                f"Where: {path.absolute()} (key: classification_evidence.default).\n"
                f"Allowed: 'none' or one of {sorted(supported)}.\n"
                "Fix: set `default: none` or a supported document type."
            )

    def check_extraction_prompts(self, model_type: str) -> None:
        """Check that a model's prompt file and the schema file agree on the document types.

        A document type is a block in the schema file and a prompt of the same name. A prompt under a
        name that is no type is never used, and usually a misspelling; a type without a prompt cannot be
        read.

        Raises:
            ValueError: Naming the prompts that belong to no type and the types that have no prompt.
        """
        from common.field_schema import get_field_schema

        path = self._dir / self._resolve_filename(model_type)
        prompts = set(self._load_yaml(path).get("prompts", {}))
        types = set(get_field_schema().document_types)
        strays, unread = sorted(prompts - types), sorted(types - prompts)
        if strays or unread:
            what = []
            if strays:
                what.append(f"prompt(s) for no document type: {', '.join(strays)}")
            if unread:
                what.append(f"document type(s) with no prompt: {', '.join(unread)}")
            raise ValueError(
                f"What: {'; '.join(what)}.\n"
                f"Where: {path.absolute()}, under the prompts: section, and config/field_definitions.yaml, "
                f"under document_types.\n"
                f"Expected: one prompt for each document type, under the type's own name: "
                f"{', '.join(sorted(types))}.\n"
                f"How to fix: give each type a prompt and each prompt a type. A new type needs a block "
                f"under document_types and a prompt of the same name."
            )
