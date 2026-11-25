"""The prompts, read from the YAML files of ``prompts/``: the one place they are read.

A model's extraction prompts are in the file its registration names, one prompt for each document type.
The classification prompt, with the column roles and the rules that turn its answers into a type, is in
``document_type_detection.yaml``.

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

from common.errors import ConfigError

_DETECTION_FILE = "document_type_detection.yaml"
ANSWER_KINDS = ("columns", "yes_no", "number")


@dataclass(frozen=True)
class Question:
    """One question of the classification prompt."""

    label: str
    answer: str
    ask: str


class PromptCatalog:
    """The prompts of one folder of YAML files.

    A registered model's name stands for the file its registration names (``VllmSpec.prompt_file``).
    A file is parsed once for each instance.

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

    def get_prompt(self, namespace: str, key: str) -> str:
        """Load a prompt string by namespace and key.

        Args:
            namespace: A registered model_type, which maps to its prompt_file.
            key: Prompt key within the YAML file's ``prompts`` section.

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

        return prompts[key]["prompt"]

    # -- Entry point 2: the classification questions and rules ----------------

    def get_classification_questions(self) -> tuple[Question, ...]:
        """The questions the classification prompt asks, in order, checked.

        Raises:
            ConfigError: If the ``classification`` block is missing or a question cannot be used.
        """
        path = self._dir / _DETECTION_FILE
        block = self._load_yaml(path).get("classification")
        example = (
            'classification:\n  opening: "Look at this document image and answer each question:"\n'
            "  questions:\n    - label: PAID\n      answer: yes_no\n      ask: |\n"
            "        Is there evidence that payment was completed? Answer YES or NO."
        )

        def refuse(what: str, key: str, fix: str) -> ConfigError:
            return ConfigError(
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

    def photo_split_prompt(self) -> str:
        """The question the split task puts to the model about each photograph."""
        return self._load_yaml(self._dir / "photo_split.yaml")["prompt"].strip()

    def _refuse(self, what: str, key: str, expected: str, fix: str) -> ConfigError:
        """A fault in the classification file, with what is wrong, where, what is expected and the fix."""
        path = (self._dir / _DETECTION_FILE).absolute()
        return ConfigError(
            f"What: {what}\nWhere: {path} (key: {key}).\nExpected: {expected}\nHow to fix: {fix}"
        )

    def get_column_roles(self) -> dict[str, list[str]]:
        """Load semantic column-role -> header-keyword lists from detection YAML.

        Read by ``common.classification``.

        Returns:
            Mapping of role name to its list of header keywords.

        Raises:
            ConfigError: If the ``column_roles`` block is missing or malformed.
        """
        roles = self._load_yaml(self._dir / _DETECTION_FILE).get("column_roles")
        example = "column_roles:\n  debit:\n    - debit\n    - withdrawal"
        if not isinstance(roles, dict) or not roles:
            raise self._refuse(
                "the classification file has no column_roles block, or an empty one.",
                "column_roles",
                f"a mapping of each role to the words a column heading may use for it, e.g.\n{example}",
                "add the column_roles: section to document_type_detection.yaml.",
            )
        for role, keywords in roles.items():
            if not isinstance(keywords, list) or not keywords:
                raise self._refuse(
                    f"the column role {role!r} has no list of words: {keywords!r}.",
                    f"column_roles.{role}",
                    f"a list of at least one word a heading may use, e.g.\n{example}",
                    f"give {role!r} at least one word, or remove the role.",
                )
        return roles

    def get_classification_evidence(self) -> dict[str, Any]:
        """Load the evidence-based classification rule set from detection YAML.

        Read by ``common.classification``: an ordered list of
        ``rules`` (first match wins) plus a ``default`` for when none match.

        Returns:
            Dict with keys ``rules`` (list) and ``default`` (str).

        Raises:
            ConfigError: If the ``classification_evidence`` block is missing or
                structurally invalid.
        """
        evidence = self._load_yaml(self._dir / _DETECTION_FILE).get("classification_evidence")
        example = (
            "classification_evidence:\n  rules:\n    - type: BANK_STATEMENT\n"
            "      when: { any_roles: [debit, credit, balance] }\n  default: none"
        )
        if not isinstance(evidence, dict):
            raise self._refuse(
                "the classification file has no classification_evidence block.",
                "classification_evidence",
                f"a block with the rules and a default, e.g.\n{example}",
                "add the classification_evidence: section to document_type_detection.yaml.",
            )
        rules = evidence.get("rules")
        if not isinstance(rules, list) or not rules:
            raise self._refuse(
                f"classification_evidence.rules is not a list of at least one rule: {rules!r}.",
                "classification_evidence.rules",
                f"a list of rules, each with a type and a when clause, e.g.\n{example}",
                "add at least one rule with a type and a when clause.",
            )
        if "default" not in evidence:
            raise self._refuse(
                "classification_evidence has no default, the type of a page no rule fits.",
                "classification_evidence.default",
                "none, to give such a page the fallback type, or a document type, e.g. default: INVOICE",
                "add default: none under classification_evidence.",
            )
        self._validate_evidence_references(evidence, rules)
        return evidence

    def _validate_evidence_references(self, evidence: dict[str, Any], rules: list) -> None:
        """Cross-reference rules against column_roles and supported types.

        Ensures every referenced role exists, every `when:` uses known keys,
        and every emitted type (and the default) is a supported document type.
        """
        from common.field_schema import get_field_schema

        known_roles = set(self.get_column_roles())
        supported = set(get_field_schema().document_types)
        asked = {q.label.lower() for q in self.get_classification_questions() if q.answer == "yes_no"}
        allowed_when_keys = {"any_roles", "all_roles"} | asked
        types = ", ".join(sorted(supported))

        for i, rule in enumerate(rules):
            loc = f"classification_evidence.rules[{i}]"
            if not isinstance(rule, dict) or "type" not in rule or "when" not in rule:
                raise self._refuse(
                    f"a classification rule lacks its type or its when clause: {rule!r}.",
                    loc,
                    "a rule with both, e.g.\n    - type: RECEIPT\n      when: { paid: true }",
                    "give the rule a type and a when clause.",
                )
            rtype = rule["type"]
            if not isinstance(rtype, str) or rtype.lower() not in supported:
                raise self._refuse(
                    f"the rule type {rtype!r} is not a document type.",
                    f"{loc}.type",
                    f"a document type of the schema file, in either case: {types}.",
                    "correct the type, or give a new type a block in config/field_definitions.yaml "
                    "(document_types).",
                )
            when = rule["when"]
            if not isinstance(when, dict) or not when:
                raise self._refuse(
                    f"the when clause of the rule for {rtype} is not a mapping with a condition: {when!r}.",
                    f"{loc}.when",
                    "at least one condition, e.g. when: { any_roles: [debit] }",
                    "add a condition to the rule.",
                )
            unknown_keys = set(when) - allowed_when_keys
            if unknown_keys:
                raise self._refuse(
                    f"the when clause of the rule for {rtype} has key(s) that are not known: "
                    f"{sorted(unknown_keys)}.",
                    f"{loc}.when",
                    f"only these keys: {sorted(allowed_when_keys)}.",
                    "remove the key, correct its spelling, or add a yes_no question with that label "
                    "under classification.questions.",
                )
            for role_key in ("any_roles", "all_roles"):
                for role in when.get(role_key, []):
                    if role not in known_roles:
                        raise self._refuse(
                            f"the rule for {rtype} names a column role that does not exist: {role!r}.",
                            f"{loc}.when.{role_key}",
                            f"one of the roles under column_roles: {sorted(known_roles)}.",
                            f"add {role!r} under column_roles, or name a role that is there.",
                        )

        default = evidence["default"]
        if isinstance(default, str) and default.lower() != "none" and default.lower() not in supported:
            raise self._refuse(
                f"classification_evidence.default is {default!r}, which is not a document type.",
                "classification_evidence.default",
                f"none, or a document type of the schema file: {types}.",
                "set default: none, or a document type.",
            )

    def check_extraction_prompts(self, model_type: str) -> None:
        """Check that a model's prompt file and the schema file agree on the document types.

        A document type is a block in the schema file and a prompt of the same name. A prompt under a
        name that is no type is never used, and usually a misspelling; a type without a prompt cannot be
        read.

        Raises:
            ConfigError: Naming the prompts that belong to no type and the types that have no prompt.
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
            raise ConfigError(
                f"What: {'; '.join(what)}.\n"
                f"Where: {path.absolute()}, under the prompts: section, and config/field_definitions.yaml, "
                f"under document_types.\n"
                f"Expected: one prompt for each document type, under the type's own name: "
                f"{', '.join(sorted(types))}.\n"
                f"How to fix: give each type a prompt and each prompt a type. A new type needs a block "
                f"under document_types and a prompt of the same name."
            )
