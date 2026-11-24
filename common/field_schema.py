"""The document types and how their fields are scored, read once from ``config/field_definitions.yaml``.

    from common.field_schema import get_field_schema
    schema = get_field_schema()

A document type is a block in that file and a prompt of the same name. Nothing in Python names a type:
what differs from one type to another (its fields, the other names it goes by, a fixed completion budget,
a table whose rows are lined up before scoring) is asked of the schema.

A field is an entry in the same file: how a value of it is tidied when a reply is cleaned, whether it
is a list, and how it is compared with the answer when it is scored. Nothing in Python names a field
either. The file also holds the scoring thresholds.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from common.config_schema import ConfigError

CONFIG = Path(__file__).parent.parent / "config" / "field_definitions.yaml"

_TYPE_KEYS = frozenset({"fields", "also_called", "completion_tokens", "table"})
_TABLE_KEYS = frozenset({"columns", "rows_absent_where_not_found_in", "rows_in_date_order_of"})
_FIELD_KEYS = frozenset(
    {"cleaned_as", "list", "scored_as", "scored_row_by_row", "single_value_scored_as", "also_written"}
)
CLEANED_AS = (
    "text",
    "money",
    "amount_as_written",
    "quantity",
    "date",
    "day_month_year",
    "identifier",
    "address",
    "document_type",
)
SCORED_AS = ("text", "money", "date", "yes_no", "identifier")
# What stands for a value that is not there: in a reply, in a cleaned field and in an answer.
NOT_FOUND = "NOT_FOUND"
_EXAMPLE = (
    "document_types:\n"
    "  invoice:\n"
    "    also_called: [invoice, tax invoice]\n"
    "    fields: [DOCUMENT_TYPE, SUPPLIER_NAME, TOTAL_AMOUNT]\n"
    "fields:\n"
    "  TOTAL_AMOUNT: {cleaned_as: money, list: false, scored_as: money}"
)


class FieldSchemaError(ConfigError):
    """``field_definitions.yaml`` is missing or cannot be used. The message says what, where and the fix."""

    def __init__(self, message: str, path: Path | None = None) -> None:
        self.path = path
        super().__init__(message)


@dataclass(frozen=True)
class Table:
    """A document type's list fields that are the columns of one table, an entry a row."""

    columns: tuple[str, ...]
    # Rows marked NOT_FOUND in this column are dropped from both sides before scoring.
    rows_absent_where_not_found_in: str | None
    # The rows of both sides are put in the date order of this column before scoring.
    rows_in_date_order_of: str | None


@dataclass(frozen=True)
class Field:
    """What the schema file says of one field."""

    name: str
    # How a value is tidied when a reply is cleaned; for a list, how each entry is. One of CLEANED_AS.
    cleaned_as: str
    is_list: bool
    # How a value is compared with the answer. One of SCORED_AS.
    scored_as: str
    # A list that is a column of a table: an entry is compared as what it is, row by row.
    scored_row_by_row: bool
    # How one value, not a list, is compared with the answer. scored_as unless the entry says otherwise.
    single_value_scored_as: str
    # Other spellings of the field's name that a reply may use.
    also_written: tuple[str, ...]


@dataclass(frozen=True)
class DocumentType:
    """What the schema file says of one document type."""

    name: str
    fields: tuple[str, ...]
    also_called: tuple[str, ...]
    completion_tokens: int | None
    table: Table | None


@dataclass(frozen=True)
class FieldSchema:
    """The field definitions, as the rest of the pipeline reads them."""

    # Every document type, by its own name in lower case, in the order of the file.
    document_types: dict[str, DocumentType] = field(repr=False)
    # Every field, by its name.
    fields: dict[str, Field] = field(repr=False)
    # Every name a document type goes by, in lower case, to the type's own name.
    aliases: dict[str, str] = field(repr=False)
    _thresholds: dict[str, float] = field(repr=False)
    _path: Path = field(repr=False)

    @classmethod
    def from_yaml(cls, path: Path = CONFIG) -> "FieldSchema":
        """Read and check the field definitions.

        Raises:
            FieldSchemaError: If the file is missing, is not valid YAML, or a document type's block
                cannot be used.
        """
        where = path.absolute()

        def refuse(what: str, key: str, expected: str, fix: str) -> FieldSchemaError:
            return FieldSchemaError(
                f"What: {what}\nWhere: {where}, key {key}.\nExpected: {expected}\nHow to fix: {fix}",
                path=path,
            )

        if not path.exists():
            raise FieldSchemaError(
                f"What: the field definitions file does not exist.\nWhere: {where}\n"
                f"Expected: a file with this structure:\n{_EXAMPLE}\n"
                f"How to fix: restore config/field_definitions.yaml in the repository.",
                path=path,
            )
        try:
            raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as err:
            raise FieldSchemaError(
                f"What: the field definitions file is not valid YAML: {err}\nWhere: {where}\n"
                f"Expected: a file with this structure:\n{_EXAMPLE}\n"
                f"How to fix: correct the YAML at the place the message names.",
                path=path,
            ) from err
        blocks = raw.get("document_types") if isinstance(raw, dict) else None
        if not isinstance(blocks, dict) or not blocks:
            raise refuse(
                "the field definitions file has no document_types section.",
                "document_types",
                f"\n{_EXAMPLE}",
                "add a document_types section with a block for each document type.",
            )

        document_types: dict[str, DocumentType] = {}
        aliases: dict[str, str] = {}
        for name, block in blocks.items():
            key = f"document_types.{name}"
            if (
                not isinstance(block, dict)
                or not block.get("fields")
                or not isinstance(block["fields"], list)
            ):
                raise refuse(
                    f"the document type {name!r} has no list of fields.",
                    f"{key}.fields",
                    "a list with at least one field name, e.g. fields: [DOCUMENT_TYPE, TOTAL_AMOUNT]",
                    f"give {name!r} a fields list, or remove its block.",
                )
            unknown = sorted(set(block) - _TYPE_KEYS)
            if unknown:
                raise refuse(
                    f"the document type {name!r} has key(s) the pipeline does not know: {', '.join(unknown)}.",
                    f"{key}.{unknown[0]}",
                    f"only {', '.join(sorted(_TYPE_KEYS))}",
                    "remove the key, or correct its spelling.",
                )
            budget = block.get("completion_tokens")
            if budget is not None and (
                not isinstance(budget, int) or isinstance(budget, bool) or budget < 1
            ):
                raise refuse(
                    f"the completion budget of {name!r} is {budget!r}.",
                    f"{key}.completion_tokens",
                    "a whole number of tokens, e.g. completion_tokens: 4096, or no key at all",
                    "set it to a positive whole number, or remove it to have the budget worked out.",
                )
            names = block.get("also_called") or []
            if not isinstance(names, list):
                raise refuse(
                    f"the other names of {name!r} are not a list.",
                    f"{key}.also_called",
                    "a list of names, e.g. also_called: [invoice, tax invoice]",
                    "write the names as a list, or remove the key.",
                )
            document_types[name.lower()] = DocumentType(
                name=name.lower(),
                fields=tuple(block["fields"]),
                also_called=tuple(str(other) for other in names),
                completion_tokens=budget,
                table=cls._table(block.get("table"), name, key, refuse),
            )
            aliases[name.lower()] = name
            for other in names:
                aliases[str(other).lower()] = name

        fields = cls._fields(raw.get("fields"), document_types, refuse)
        evaluation = raw.get("evaluation", {})
        return cls(
            document_types=document_types,
            fields=fields,
            aliases=aliases,
            _thresholds={name: float(value) for name, value in evaluation.get("thresholds", {}).items()},
            _path=path,
        )

    @staticmethod
    def _fields(raw: Any, document_types: dict[str, DocumentType], refuse: Any) -> dict[str, Field]:
        """The `fields:` section, checked. Every field a type or a table names must have an entry."""
        example = "fields:\n  TOTAL_AMOUNT: {cleaned_as: money, list: false, scored_as: money}"
        if not isinstance(raw, dict) or not raw:
            raise refuse(
                "the field definitions file has no fields section.",
                "fields",
                f"an entry for each field, e.g.\n{example}",
                "add a fields section with an entry for every field a document type names.",
            )
        fields: dict[str, Field] = {}
        for name, entry in raw.items():
            key = f"fields.{name}"
            needed = ("cleaned_as", "list", "scored_as")
            if not isinstance(entry, dict) or any(part not in entry for part in needed):
                raise refuse(
                    f"the field {name!r} does not say all of: {', '.join(needed)}.",
                    key,
                    f"each of them written out, e.g.\n{example}",
                    f"give {name!r} all three. Nothing is assumed for a key left out.",
                )
            unknown = sorted(set(entry) - _FIELD_KEYS)
            if unknown:
                raise refuse(
                    f"the field {name!r} has key(s) the pipeline does not know: {', '.join(unknown)}.",
                    f"{key}.{unknown[0]}",
                    f"only {', '.join(sorted(_FIELD_KEYS))}",
                    "remove the key, or correct its spelling.",
                )
            for part, allowed in (
                ("cleaned_as", CLEANED_AS),
                ("scored_as", SCORED_AS),
                ("single_value_scored_as", SCORED_AS),
            ):
                if part in entry and entry[part] not in allowed:
                    raise refuse(
                        f"the field {name!r} has {part}: {entry[part]!r}.",
                        f"{key}.{part}",
                        f"one of {', '.join(allowed)}",
                        f"set {part} to one of those.",
                    )
            for part in ("list", "scored_row_by_row"):
                if not isinstance(entry.get(part, False), bool):
                    raise refuse(
                        f"the field {name!r} has {part}: {entry[part]!r}.",
                        f"{key}.{part}",
                        "true or false",
                        f"set {part} to true or false.",
                    )
            spellings = entry.get("also_written", [])
            if not isinstance(spellings, list):
                raise refuse(
                    f"the other spellings of {name!r} are not a list.",
                    f"{key}.also_written",
                    "a list of names, e.g. also_written: [LINE_ITEM_DESCRIPTION]",
                    "write the spellings as a list, or remove the key.",
                )
            fields[name] = Field(
                name=name,
                cleaned_as=entry["cleaned_as"],
                is_list=entry["list"],
                scored_as=entry["scored_as"],
                scored_row_by_row=entry.get("scored_row_by_row", False),
                single_value_scored_as=str(entry.get("single_value_scored_as", entry["scored_as"])),
                also_written=tuple(str(spelling) for spelling in spellings),
            )
        for kind in document_types.values():
            named = [*kind.fields, *(kind.table.columns if kind.table else ())]
            missing = [name for name in named if name not in fields]
            if missing:
                raise refuse(
                    f"the document type {kind.name!r} names field(s) with no entry: {', '.join(missing)}.",
                    f"fields.{missing[0]}",
                    f"an entry for each, e.g.\n{example}",
                    f"add an entry for each under fields, or remove it from {kind.name!r}.",
                )
        return fields

    @staticmethod
    def _table(block: Any, name: str, key: str, refuse: Any) -> Table | None:
        if block is None:
            return None
        example = (
            "table:\n      columns: [TRANSACTION_DATES, TRANSACTION_AMOUNTS_PAID]\n"
            "      rows_in_date_order_of: TRANSACTION_DATES"
        )
        if (
            not isinstance(block, dict)
            or not isinstance(block.get("columns"), list)
            or not block["columns"]
            or set(block) - _TABLE_KEYS
        ):
            raise refuse(
                f"the table of {name!r} cannot be read.",
                f"{key}.table",
                f"a list of columns, and at most the keys {', '.join(sorted(_TABLE_KEYS))}, e.g.\n    {example}",
                "give the table its columns and remove any other key, or remove the table.",
            )
        columns = tuple(block["columns"])
        for rule in ("rows_absent_where_not_found_in", "rows_in_date_order_of"):
            if block.get(rule) is not None and block[rule] not in columns:
                raise refuse(
                    f"the table of {name!r} names {block[rule]!r} in {rule}, which is not one of its columns.",
                    f"{key}.table.{rule}",
                    f"one of {', '.join(columns)}",
                    f"name one of the table's columns in {rule}, or add the field to columns.",
                )
        return Table(
            columns=columns,
            rows_absent_where_not_found_in=block.get("rows_absent_where_not_found_in"),
            rows_in_date_order_of=block.get("rows_in_date_order_of"),
        )

    def resolve_doc_type(self, raw_type: str) -> str:
        """A document type's own name in lower case, from any name it goes by; the input if unknown."""
        return self.aliases.get(raw_type.lower(), raw_type.lower())

    def document_type(self, name: str) -> DocumentType | None:
        """What the schema says of a document type, by its own name in either case; None if it has none."""
        return self.document_types.get(name.lower())

    def field_named(self, name: str) -> Field | None:
        """What the schema says of a field, by its name; None for a name that is no field."""
        return self.fields.get(name)

    @property
    def document_type_field(self) -> str | None:
        """The field in which a reply, or an answer, names its document type; None if no field does."""
        return next(
            (name for name, entry in self.fields.items() if entry.cleaned_as == "document_type"), None
        )

    def get_all_doc_type_fields(self) -> dict[str, list[str]]:
        """The fields of every document type, by the type's name in lower case. A new dict each call."""
        return {name: list(kind.fields) for name, kind in self.document_types.items()}

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
