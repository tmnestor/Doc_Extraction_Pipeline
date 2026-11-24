"""From a model's reply to a page's fields.

The clean stage calls ``fields_from_reply``. It does two things, in order:

1. **Parse.** A reply is asked for as lines of ``FIELD_NAME: value``. Most replies are that. The parser
   also reads the ways a reply strays: markdown emphasis, bullets, a value on the line below its name,
   the whole reply written as JSON, a list written with commas where bars were asked for.
2. **Clean.** Each value is tidied by the kind of field it is (money, list, date, identifier, address,
   text), so that the same answer written two ways compares as the same.

What a field is (a list or not, and how its values are tidied) is its entry in
``config/field_definitions.yaml``. Nothing here reads a field's name to decide it, and a name that has no
entry is treated as one piece of text.

Every field asked for comes back; one the reply does not give is ``NOT_FOUND``. Nothing here calls a model.
"""

import json
import re
from typing import Any

from dateutil import parser as date_parser  # type: ignore[import-untyped]

from .field_schema import Field, get_field_schema

NOT_FOUND = "NOT_FOUND"


def fields_from_reply(reply: str, expected_fields: list[str]) -> dict[str, str]:
    """The fields of a page, from the model's reply.

    Args:
        reply: The model's reply, as the extract stage recorded it.
        expected_fields: The fields of the page's document type.

    Returns:
        A value for every expected field, ``NOT_FOUND`` where the reply gives none.
    """
    return {name: clean_value(name, value) for name, value in parse_reply(reply, expected_fields).items()}


# ---------------------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------------------

_PLAIN_TEXT = Field(
    name="",
    cleaned_as="text",
    is_list=False,
    scored_as="text",
    scored_row_by_row=False,
    single_value_scored_as="text",
    also_written=(),
)


def _entry(name: str) -> Field:
    """What the schema says of a field; a name with no entry is one piece of text."""
    return get_field_schema().field_named(name) or _PLAIN_TEXT


_ADDRESS_WORDS = (
    "street",
    "road",
    "avenue",
    "drive",
    "lane",
    "court",
    "place",
    "way",
    "vic",
    "nsw",
    "qld",
    "sa",
    "wa",
    "tas",
    "nt",
    "act",
)
# A quantity written "3x", "x3" or "3 EACH" is the right count with a decoration. Both patterns need a
# digit beside the decoration, so a quantity that is a word keeps its letters: "BOX" keeps its X.
_QUANTITY_LEADING_MULTIPLIER = re.compile(r"^\s*[xX]\s*(?=\d)")
_QUANTITY_TRAILING_DECORATION = re.compile(r"(?<=\d)\s*(?:X|EACH)\s*$", re.IGNORECASE)


def parse_reply(reply: str, expected_fields: list[str]) -> dict[str, str]:
    """The reply's value for each expected field, before cleaning."""
    if not reply:
        return dict.fromkeys(expected_fields, NOT_FOUND)
    as_json = _from_json(reply.strip(), expected_fields)
    if as_json is not None:
        # A JSON reply skips the tidying a reply in lines gets, so its quantities are bared here.
        for name in as_json:
            if _entry(name).cleaned_as == "quantity":
                as_json[name] = bare_quantities(as_json[name])
        return as_json
    return _tidied(_from_lines(reply, expected_fields))


def bare_quantities(value: str) -> str:
    """A list of quantities without the "x" or "EACH" a model puts beside the count."""
    if not value or str(value).strip().upper() == NOT_FOUND:
        return value
    items = []
    for item in str(value).split("|"):
        item = _QUANTITY_LEADING_MULTIPLIER.sub("", item.strip())
        items.append(_QUANTITY_TRAILING_DECORATION.sub("", item).strip())
    return " | ".join(items)


# -- a reply written as JSON --


def _without_fence(text: str, *, strip_between: bool) -> str:
    """The text without a markdown code fence round it."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:].strip() if strip_between else text[7:]
        if not strip_between and text.startswith("```"):
            text = text[3:]
    elif text.startswith("```"):
        text = text[3:].strip() if strip_between else text[3:]
    if text.endswith("```"):
        text = text[:-3].strip() if strip_between else text[:-3]
    return text


def _looks_like_json(text: str) -> bool:
    text = _without_fence(text, strip_between=True)
    return (
        len(text) >= 2
        and text[0] == "{"
        and text.count('"') >= 4
        and (text[-1] == "}" or text.find('"') > 0)
    )


def _from_json(text: str, expected_fields: list[str]) -> dict[str, str] | None:
    """The fields of a reply written as JSON, or None if it is not JSON or cannot be read as it."""
    if not _looks_like_json(text):
        return None
    try:
        data = json.loads(_mended_json(text))
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    fields = dict.fromkeys(expected_fields, NOT_FOUND)
    for name in expected_fields:
        if name in data and data[name] is not None and data[name] != "":
            fields[name] = str(data[name])
    return fields


def _mended_json(text: str) -> str:
    """JSON with the commonest damage mended: a reply cut short, stray commas, commas left out."""
    text = _without_fence(text, strip_between=False).strip()

    if not text.endswith("}"):
        # Cut short: close the value that was being written, then the object.
        lines = text.split("\n")
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if line.count('"') % 2 == 1 and ":" in line:
                last_quote = line.rfind('"')
                if last_quote > 0 and line[last_quote - 1] != "\\":
                    lines[i] = line + '"'
                break
            if line.endswith(("|", ",", "| ")):
                lines[i] = line.rstrip("| ,") + '"'
                break
        text = re.sub(r",\s*$", "", "\n".join(lines), flags=re.MULTILINE)
        if not text.endswith("}"):
            text += "\n}"

    kept: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if line in ('",', ","):
            # A comma on a line of its own belongs to the field above it.
            if kept and '":' in kept[-1] and not kept[-1].endswith((",", "}")):
                kept[-1] += ","
            continue
        if line.endswith('",'):
            line = line[:-2] + '"'
        kept.append(line)

    joined = []
    for i, line in enumerate(kept):
        following = kept[i + 1] if i < len(kept) - 1 else ""
        if i < len(kept) - 1 and '":' in line and not line.endswith((",", "}")) and '":' in following:
            line += ","
        joined.append(line)

    text = re.sub(r',\s*"', ',\n  "', "\n".join(joined))
    return re.sub(r'",\s*,', '",', text)


# -- a reply written as lines --


def _plain_name_line(line: str) -> str:
    """A line without the word KEY that some replies put before a field's name."""
    line = re.sub(r"^KEY:\s*([A-Z_]+):", r"\1:", line)
    return re.sub(r"^KEY\s+([A-Z_]+):", r"\1:", line)


def _from_lines(reply: str, expected_fields: list[str]) -> dict[str, str]:
    """The fields of a reply written as lines of ``NAME: value``."""
    lines = reply.strip().split("\n")
    fields = dict.fromkeys(expected_fields, NOT_FOUND)
    first = _first_pass(lines, expected_fields)
    fields.update(first)
    # Where the plain reading finds under half the fields, the reply is probably markdown with values
    # below their names. A second reading fills what the first left empty.
    if sum(1 for value in first.values() if value != NOT_FOUND) < len(expected_fields) * 0.5:
        for line in _markdown_as_lines(lines):
            if not line.strip() or ":" not in line:
                continue
            name, _, value = _plain_name_line(re.sub(r"\*+([^*]+)\*+", r"\1", line)).partition(":")
            name, value = name.strip().upper(), value.strip()
            if name in fields and (fields[name] == NOT_FOUND or not fields[name]):
                fields[name] = value if value else NOT_FOUND
    return fields


def _first_pass(lines: list[str], expected_fields: list[str]) -> dict[str, str]:
    """Each ``NAME: value`` line, and a list given as bullets on the lines below its name."""
    found: dict[str, str] = {}
    # A field's name as a reply may misspell it, to the name itself.
    spellings = {other: name for name in expected_fields for other in _entry(name).also_written}
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or ":" not in line:
            i += 1
            continue
        # Markdown goes first: a bullet's star, emphasis round a word, then any star left.
        line = re.sub(r"^\s*\*+\s*", "", line)
        line = re.sub(r"\*+([^*]+)\*+", r"\1", line).replace("**", "").replace("*", "")
        name, _, value = _plain_name_line(line).partition(":")
        name = spellings.get(name, name)
        name, value = name.strip().upper(), value.strip()
        if name in expected_fields:
            if not value and _entry(name).is_list and i + 1 < len(lines):
                bullets, after = _bullets_below(lines, i + 1)
                if bullets:
                    value = " | ".join(bullet.lstrip("* -").strip() for bullet in bullets)
                    i = after - 1
            found[name] = value if value else NOT_FOUND
        i += 1
    return found


def _bullets_below(lines: list[str], start: int) -> tuple[list[str], int]:
    """The bullet lines from ``start``, and the index of the line after them."""
    bullets: list[str] = []
    j = start
    while j < len(lines) and not lines[j].strip():
        j += 1
    while j < len(lines):
        line = lines[j].strip()
        if ":" in line and not line.startswith(("*", "-")):
            before_colon = line.split(":")[0].strip()
            if before_colon.isupper() and "_" in before_colon:
                break
        if not line and bullets:
            break
        if not line:
            j += 1
            continue
        if not line.startswith(("*", "-")):
            break
        bullets.append(line)
        j += 1
    return bullets, j


def _markdown_as_lines(lines: list[str]) -> list[str]:
    """A markdown reply as ``NAME: value`` lines: a value below its bold name is joined to it."""
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        bold = re.match(r"^\*\*([A-Z_]+):\*\*\s*(.*)?$|^\*\*([A-Z\s]+):\*\*\s*(.*)?$", line)
        if not bold:
            result.append(line)
            i += 1
            continue
        name = (bold.group(1) or bold.group(3)).replace(" ", "_")
        value = (bold.group(2) or bold.group(4) or "").strip()
        if not value and i + 1 < len(lines):
            below: list[str] = []
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            while j < len(lines):
                following = lines[j].strip()
                if re.match(r"^\*\*[A-Z_]+:\*\*|^\*\*[A-Z\s]+:\*\*", following):
                    break
                # A colon usually starts another field, but an address may hold one.
                if ":" in following and not any(word in following.lower() for word in _ADDRESS_WORDS):
                    break
                if following:
                    below.append(following)
                j += 1
            if below:
                if not _entry(name).is_list:
                    value = " ".join(below)
                elif all(entry.startswith("*") for entry in below):
                    value = " | ".join(entry.lstrip("* ").strip() for entry in below)
                else:
                    value = " | ".join(below)
                i = j
            else:
                i += 1
        else:
            i += 1
        result.append(f"{name}: {value}" if value else f"{name}: {NOT_FOUND}")
    return result


def _tidied(fields: dict[str, str]) -> dict[str, str]:
    """Parsed values put into the forms the rest of the pipeline expects.

    The document type by its settled name, lists with " | " between entries, addresses without commas,
    quantities bare, dates as DD/MM/YYYY.
    """
    for name, value in fields.items():
        if value == NOT_FOUND:
            continue
        entry = _entry(name)
        if entry.cleaned_as == "document_type":
            settled = get_field_schema().aliases.get(value.lower().strip())
            if settled:
                fields[name] = settled.upper()
            continue
        if entry.is_list:
            if "," in value and " | " not in value:
                # Not the comma inside a number: $8,026.87 is one value.
                fields[name] = " | ".join(
                    item.strip() for item in re.split(r",(?!\d)", value) if item.strip()
                )
            elif "*" in value and " | " not in value:
                fields[name] = " | ".join(
                    line.strip().lstrip("* ").strip() for line in value.split("\n") if line.strip()
                )
            elif re.search(r"\s{2,}", value) and " | " not in value:
                fields[name] = " | ".join(
                    item.strip() for item in re.split(r"\s{2,}", value) if item.strip()
                )
            if entry.cleaned_as == "quantity":
                fields[name] = bare_quantities(fields[name])
        elif entry.cleaned_as == "address" and "," in value:
            fields[name] = " ".join(value.split(",")).strip()
        if entry.cleaned_as == "day_month_year":
            if entry.is_list:
                fields[name] = " | ".join(
                    _as_day_month_year(date.strip()) for date in fields[name].split(" | ")
                )
            else:
                fields[name] = _as_day_month_year(fields[name])
    return fields


# A date that begins with its year: 2031-02-09, 2031/2/9, 2031.02.09, 2031 Feb 9, with or without a time
# after it, and the same run together, 20310209.
_BEGINS_WITH_ITS_YEAR = re.compile(r"^(?:19|20)\d{2}(?:\D|\d{4}$)")


def _as_day_month_year(date: str) -> str:
    """A date as DD/MM/YYYY, or as it was if it cannot be read.

    The reading is dateutil's. It is told that the day comes before the month, except for a date that
    begins with its year, which is written year, month, day: told the day comes first, dateutil reads
    2031-02-09 as the second of September, and nothing says so.
    """
    if not date or date == NOT_FOUND:
        return date
    text = date.split("(")[0].strip()
    day_first = _BEGINS_WITH_ITS_YEAR.match(text) is None
    try:
        return date_parser.parse(text, dayfirst=day_first).strftime("%d/%m/%Y")
    except (ValueError, TypeError, date_parser.ParserError):
        return date


# ---------------------------------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------------------------------

_MEANS_NOTHING = ("NOT_FOUND", "Not Found", "not found", "None", "null", "", "N/A", "n/a")
_PRICE_SUFFIXES = (r"\s+(each|per\s+item|per\s+unit|per\s+piece)$", r"\s+(ea\.?|@|apiece)$")
_CURRENCY_NAMES = ((r"AUD?\s*", "$"), (r"USD?\s*", "$"), (r"\$\$+", "$"))
_IDENTIFIER_LABEL = r"^(ABN|BSB|ACN|GST|TAX|ID|NUMBER|NO\.?|#)\s*:?\s*"


def clean_value(field_name: str, raw_value: Any) -> str:
    """A value tidied by the kind of field it is. Nothing, in any of its spellings, is ``NOT_FOUND``."""
    if not raw_value or raw_value in _MEANS_NOTHING:
        return NOT_FOUND
    value = str(raw_value).strip()
    if not value:
        return NOT_FOUND
    entry = _entry(field_name)
    if entry.is_list:
        if "," in value and "|" not in value:
            value = " | ".join(item.strip() for item in re.split(r",(?!\d)", value))
        return _clean_list(entry, value)
    cleaned = _CLEANERS.get(entry.cleaned_as, _clean_text)(value)
    if entry.cleaned_as == "document_type" and cleaned and cleaned != NOT_FOUND:
        cleaned = get_field_schema().resolve_doc_type(cleaned).upper()
    return cleaned


def _without_leading_stars(value: str) -> str:
    return re.sub(r"^\s*\*+\s*", "", value).replace("**", "")


def _clean_money(value: str) -> str:
    """Money without "each", with "$" for a currency's name, without commas, the "$" in front."""
    if not value or value == NOT_FOUND:
        return NOT_FOUND
    cleaned = _without_leading_stars(value)
    for pattern in _PRICE_SUFFIXES:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    for pattern, replacement in _CURRENCY_NAMES:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned.replace(",", "")).strip()
    if "$" in cleaned and not cleaned.startswith("$"):
        cleaned = re.sub(r"(.*?)\$(.+)", r"$\2", cleaned)
    return cleaned if cleaned else NOT_FOUND


def _clean_list(entry: Field, value: str) -> str:
    """A list with " | " between its entries, each entry cleaned.

    An entry that is nothing stays in its place as ``NOT_FOUND``. A list is one column of several that
    are read row by row, and dropping an entry would move every later one against the other columns.
    A list of nothing but ``NOT_FOUND`` is the single ``NOT_FOUND``, which is how the answer key writes it.
    """
    if not value or value == NOT_FOUND:
        return NOT_FOUND
    value = _without_leading_stars(value)
    if entry.cleaned_as in ("money", "amount_as_written"):
        # Two amounts with only a space between them are two entries.
        value = re.sub(r"(\$\d+\.\d{2})\s+(\$\d+\.\d{2})", r"\1 | \2", value)
    items = value.split("|") if "|" in value else re.split(r",(?!\d)", value)
    # An entry of a list is tidied as money or as text; a list's dates and quantities were put in their
    # form when the reply was parsed.
    clean_item = _clean_money if entry.cleaned_as == "money" else _clean_text
    cleaned = [clean_item(item.strip()) or NOT_FOUND for item in items if item.strip()]
    if not cleaned or all(item == NOT_FOUND for item in cleaned):
        return NOT_FOUND
    return " | ".join(cleaned)


def _clean_date(value: str) -> str:
    """A date without spaces round its separators."""
    if not value or value == NOT_FOUND:
        return NOT_FOUND
    cleaned = _without_leading_stars(value.strip())
    cleaned = re.sub(r"\s*[-/\.]\s*", lambda found: found.group(0).strip(), cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else NOT_FOUND


def _clean_identifier(value: str) -> str:
    """An identifier without its label ("ABN 12 345..."), and an 11-digit one without its spaces."""
    if not value or value == NOT_FOUND:
        return NOT_FOUND
    cleaned = re.sub(_IDENTIFIER_LABEL, "", value.strip(), flags=re.IGNORECASE).strip()
    if re.match(r"^\d+(\s+\d+)+$", cleaned) and len(cleaned.replace(" ", "")) == 11:
        cleaned = cleaned.replace(" ", "")
    elif not re.match(r"^\d{3}-\d{3}$", cleaned):
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else NOT_FOUND


def _clean_text(value: str) -> str:
    """Text without markdown stars, doubled spaces or a trailing comma."""
    if not value or value == NOT_FOUND:
        return NOT_FOUND
    cleaned = re.sub(r"^\s*\*+\s*", "", value.strip())
    cleaned = re.sub(r"\s*\*+\s*$", "", cleaned).replace("**", "")
    cleaned = re.sub(r"\s*[,;]\s*$", "", re.sub(r"\s+", " ", cleaned)).strip()
    return cleaned if cleaned else NOT_FOUND


def _clean_address(value: str) -> str:
    """An address without a telephone number or email address added to it, and without commas."""
    if not value or value == NOT_FOUND:
        return NOT_FOUND
    cleaned = re.sub(
        r"\s*-\s*(?:P|Phone|Tel|T):\s*\(?[0-9\s\)\(-]+", "", value.strip(), flags=re.IGNORECASE
    )
    cleaned = re.sub(r"\s*-?\s*[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "", cleaned)
    # A telephone number at the end goes, but not a business number (XX XXX XXX XXX).
    if not re.search(r"\d{2}\s+\d{3}\s+\d{3}\s+\d{3}$", cleaned):
        cleaned = re.sub(r"\s*-?\s*\(?[0-9]{2,4}\)?\s*[0-9\s\-]{6,}$", "", cleaned)
    cleaned = re.sub(r"\s*[;\-]\s*$", "", cleaned.replace(",", ""))
    cleaned = re.sub(r"\s+", " ", re.sub(r"\s*/\s*", "/", cleaned)).strip()
    return cleaned if cleaned else NOT_FOUND


# How one value is tidied, by what its field's entry says. What is not here is tidied as text: a
# quantity and a document type have their own step elsewhere, and an amount as written is left alone.
_CLEANERS = {
    "money": _clean_money,
    "date": _clean_date,
    "day_month_year": _clean_date,
    "identifier": _clean_identifier,
    "address": _clean_address,
}
