"""The invoice and receipt prompts say what IS_GST_INCLUDED asks.

Left undefined, the field has two defensible answers on a tax invoice whose line
prices exclude GST and whose total includes it: true (the total includes GST,
which is what the answer key means) and false (the prices do not). A model was
then scored on its reading of the field's name, not on reading the page.
"""

from pathlib import Path

import pytest
import yaml

PROMPTS = Path(__file__).resolve().parents[2] / "prompts" / "extraction_prompts.yaml"


def prompt(key: str) -> str:
    return yaml.safe_load(PROMPTS.read_text())["prompts"][key]["prompt"]


def joined(text: str) -> str:
    """The prompt with its wrapped lines joined, so a phrase split across lines is found."""
    return " ".join(text.split())


@pytest.mark.parametrize("key", ["invoice", "receipt"])
def test_the_field_is_defined_as_the_total_including_gst(key):
    text = joined(prompt(key))

    assert "Find IS_GST_INCLUDED: Does TOTAL_AMOUNT include GST?" in text
    assert "Answer true if GST is part of the total" in text
    assert "added to a subtotal" in text and "stated as included in the prices" in text
    assert "Answer false only if the document says no GST was charged" in text
    assert "Answer NOT_FOUND if GST is not mentioned" in text


@pytest.mark.parametrize("key", ["invoice", "receipt"])
def test_the_definition_does_not_change_the_fields_asked_for(key):
    field_lines = [line for line in prompt(key).splitlines() if line.endswith(": NOT_FOUND")]

    assert len(field_lines) == 13
    assert field_lines.count("IS_GST_INCLUDED: NOT_FOUND") == 1


@pytest.mark.parametrize("key", ["invoice", "receipt"])
def test_the_definition_quotes_labels_not_amounts(key):
    text = joined(prompt(key))
    definition = text[text.index("Find IS_GST_INCLUDED") : text.index("- Replace NOT_FOUND")]

    assert "$" not in definition
    assert not any(character.isdigit() for character in definition.replace("GST (10%)", ""))
