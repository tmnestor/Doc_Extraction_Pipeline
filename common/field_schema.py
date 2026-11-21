"""The fields of each document type, read once from ``config/field_definitions.yaml``.

    from common.field_schema import get_field_schema
    fields = get_field_schema()

It answers four questions: which fields a document type has, which kind a field is for scoring (money,
date, yes-or-no, a bank statement's column), which names a document type goes by, and what the scoring
thresholds are.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG = Path(__file__).parent.parent / "config" / "field_definitions.yaml"

# Two columns of a bank statement that are neither asked for nor scored.
_NOT_ASKED_FOR: frozenset[str] = frozenset({"TRANSACTION_AMOUNTS_RECEIVED", "ACCOUNT_BALANCE"})


class FieldSchemaError(Exception):
    """``field_definitions.yaml`` is missing or cannot be used. The message says what, where and the fix."""

    def __init__(self, message: str, path: Path | None = None) -> None:
        self.path = path
        super().__init__(message)


@dataclass(frozen=True)
class FieldSchema:
    """The field definitions, as the rest of the pipeline reads them."""

    monetary_fields: frozenset[str]
    date_fields: frozenset[str]
    boolean_fields: frozenset[str]
    transaction_list_fields: frozenset[str]
    supported_document_types: tuple[str, ...]
    # Every name a document type goes by, in lower case, to the type's own name.
    aliases: dict[str, str] = field(repr=False)
    _fields_of_type: dict[str, tuple[str, ...]] = field(repr=False)
    _thresholds: dict[str, float] = field(repr=False)
    _path: Path = field(repr=False)

    @classmethod
    def from_yaml(cls, path: Path = CONFIG) -> "FieldSchema":
        """Read and check the field definitions.

        Raises:
            FieldSchemaError: If the file is missing, is not valid YAML, or lacks a part that is needed.
        """
        where = path.absolute()
        example = "document_fields:\n  invoice:\n    fields: [DOCUMENT_TYPE, BUSINESS_ABN, ...]"
        if not path.exists():
            raise FieldSchemaError(
                f"What: the field definitions file does not exist.\nWhere: {where}\n"
                f"Expected: a file with this structure:\n{example}\n"
                f"How to fix: restore config/field_definitions.yaml in the repository.",
                path=path,
            )
        try:
            raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as err:
            raise FieldSchemaError(
                f"What: the field definitions file is not valid YAML: {err}\nWhere: {where}\n"
                f"Expected: a file with this structure:\n{example}\n"
                f"How to fix: correct the YAML at the place the message names.",
                path=path,
            ) from err
        document_fields = raw.get("document_fields") if isinstance(raw, dict) else None
        if not isinstance(document_fields, dict) or not document_fields:
            raise FieldSchemaError(
                f"What: the field definitions file has no document_fields section.\n"
                f"Where: {where}, key document_fields.\nExpected:\n{example}\n"
                f"How to fix: add a document_fields section with a block for each document type.",
                path=path,
            )
        fields_of_type: dict[str, tuple[str, ...]] = {}
        for document_type, block in document_fields.items():
            if not isinstance(block, dict) or not block.get("fields"):
                raise FieldSchemaError(
                    f"What: the document type {document_type!r} has no list of fields.\n"
                    f"Where: {where}, key document_fields.{document_type}.fields.\n"
                    f"Expected: a list with at least one field name, e.g. fields: [DOCUMENT_TYPE, TOTAL_AMOUNT]\n"
                    f"How to fix: give {document_type!r} a fields list, or remove its block.",
                    path=path,
                )
            fields_of_type[document_type] = tuple(
                name for name in block["fields"] if name not in _NOT_ASKED_FOR
            )

        evaluation = raw.get("evaluation", {})
        kinds = evaluation.get("field_types", {})
        aliases: dict[str, str] = {}
        for document_type, names in raw.get("document_type_aliases", {}).items():
            aliases[document_type.lower()] = document_type
            for name in names if isinstance(names, list) else []:
                aliases[name.lower()] = document_type
        return cls(
            monetary_fields=frozenset(kinds.get("monetary", [])),
            date_fields=frozenset(kinds.get("date", [])),
            boolean_fields=frozenset(kinds.get("boolean", [])),
            transaction_list_fields=frozenset(kinds.get("transaction_list", [])),
            supported_document_types=tuple(raw.get("supported_document_types", [])),
            aliases=aliases,
            _fields_of_type=fields_of_type,
            _thresholds={name: float(value) for name, value in evaluation.get("thresholds", {}).items()},
            _path=path,
        )

    def resolve_doc_type(self, raw_type: str) -> str:
        """A document type's own name in lower case, from any name it goes by; the input if unknown."""
        return self.aliases.get(raw_type.lower(), raw_type.lower())

    def get_all_doc_type_fields(self) -> dict[str, list[str]]:
        """The fields of every document type, by the type's name in lower case. A new dict each call."""
        return {document_type: list(names) for document_type, names in self._fields_of_type.items()}

    def get_threshold(self, name: str) -> float:
        """A scoring threshold from ``evaluation.thresholds``.

        Raises:
            FieldSchemaError: If the threshold is not in the file. Nothing is assumed in its place.
        """
        if name not in self._thresholds:
            raise FieldSchemaError(
                f"What: the scoring threshold {name!r} is not set.\n"
                f"Where: {self._path.absolute()}, key evaluation.thresholds.{name}.\n"
                f"Expected: a number between 0 and 1, e.g. {name}: 0.75. "
                f"Thresholds that are set: {', '.join(sorted(self._thresholds))}.\n"
                f"How to fix: add {name} under evaluation.thresholds in that file.",
                path=self._path,
            )
        return self._thresholds[name]


@lru_cache(maxsize=1)
def get_field_schema() -> FieldSchema:
    """The field definitions, read from the file the first time and kept."""
    return FieldSchema.from_yaml()
