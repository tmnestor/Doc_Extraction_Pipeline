"""The SROIE receipts are a document type in the YAML files, read and scored like any other.

Until 25 November 2026 they were read by a task of their own with a second copy of what the pipeline
has: a prompt in a Python string, the four fields in a Python tuple, a parser, a scorer and a report.
What makes them a dataset is now a type with four fields in the schema file, each of a kind the pipeline
already cleans and scores, a prompt of the type's name, and a size of engine call.

The prompt is the one the old task sent, to the byte, so that what the model is asked does not change.

Every reply and value here is invented, and none is a value of the dataset's answers.
"""

import hashlib

import pytest
import yaml

from common.field_schema import FieldSchema, get_field_schema
from common.prompt_catalog import PromptCatalog
from common.reply_fields import fields_from_reply
from common.settings import DEFAULT_CONFIG

TYPE = "sroie_receipt"
FIELDS = ["COMPANY", "DATE", "ADDRESS", "TOTAL"]
# Of the prompt as the old task held it in common/sroie/prompt.py, at d01b2e4.
SHA256_OF_THE_PROMPT = "4faf339e27dd20d9e278403fc5e86586ea10e8c6493fa423946b16b00b2a9bc4"


def kind_of(schema: FieldSchema, name: str) -> tuple[str, str, bool]:
    """How a field is cleaned and scored, and whether it is a list."""
    entry = schema.field_named(name)
    assert entry is not None, name
    return entry.cleaned_as, entry.scored_as, entry.is_list


def test_the_dataset_s_prompt_is_in_the_prompt_file_to_the_byte() -> None:
    prompt = PromptCatalog().get_prompt("gemma4-31b-w4a16-vllm", TYPE)

    assert hashlib.sha256(prompt.encode()).hexdigest() == SHA256_OF_THE_PROMPT
    assert prompt.startswith("Read this scanned receipt and report four fields.\n")
    assert prompt.endswith("Write NOT_FOUND as the value of any field the receipt does not show.\n")


def test_every_model_is_given_the_same_prompt() -> None:
    from models.registry import list_models

    assert len({PromptCatalog().get_prompt(model, TYPE) for model in list_models()}) == 1


def test_the_dataset_s_type_is_four_fields_of_kinds_the_pipeline_has_with_a_fixed_budget() -> None:
    schema = get_field_schema()
    described = schema.document_type(TYPE)

    assert described is not None
    assert list(described.fields) == FIELDS
    assert described.completion_tokens == 256
    # Each is the kind of the pipeline's own field for the same thing: nothing new is cleaned or scored.
    assert {name: kind_of(schema, name) for name in FIELDS} == {
        "COMPANY": kind_of(schema, "SUPPLIER_NAME"),
        "DATE": kind_of(schema, "INVOICE_DATE"),
        "ADDRESS": kind_of(schema, "BUSINESS_ADDRESS"),
        "TOTAL": kind_of(schema, "TOTAL_AMOUNT"),
    }


def test_the_receipts_have_a_size_of_engine_call_as_every_type_has() -> None:
    """The size is every type's (tests/config/test_every_engine_call_fills_the_engine.py). Until
    25 November 2026 it was 8, the old task's, which left half an engine's places empty.
    """
    sizes = yaml.safe_load(DEFAULT_CONFIG.read_text())["pipeline"]["batching"]

    assert sizes[TYPE] == sizes["bank_statement"]


# --- a reply in the dataset's form, read by the pipeline's parser and cleaners ---------------------------
#
# These are what the parser and the cleaners give at d01b2e4, written down after each was read. The
# pipeline's own were not changed for this dataset: it is read and cleaned as every other page is.

PLAIN = (
    "company: QUILL & FEATHER BOOKS SDN BHD\n"
    "date: 09/02/2031\n"
    "address: 12, JALAN ROOK 4, TAMAN HERON, 81750 MASAI, JOHOR\n"
    "total: 31.07\n"
)
READ = {
    "COMPANY": "QUILL & FEATHER BOOKS SDN BHD",
    "DATE": "09/02/2031",
    "ADDRESS": "12, JALAN ROOK 4, TAMAN HERON, 81750 MASAI, JOHOR",
    "TOTAL": "31.07",
}


def bold(reply: str) -> str:
    return "".join(
        f"**{line.split(chr(58), 1)[0]}:**{line.split(chr(58), 1)[1]}\n" for line in reply.splitlines()
    )


@pytest.mark.parametrize(
    "reply",
    [
        PLAIN,
        "```\n" + PLAIN + "```\n",
        bold(PLAIN),
        "".join(f"* {line}\n" for line in PLAIN.splitlines()),
        PLAIN.replace("company:", "COMPANY:")
        .replace("date:", "DATE:")
        .replace("address:", "ADDRESS:")
        .replace("total:", "TOTAL:"),
        "Here are the four fields.\n\n" + PLAIN + "\nThat is everything on the receipt.\n",
    ],
    ids=[
        "plain",
        "in a code fence",
        "names in bold",
        "starred bullets",
        "names in capitals",
        "prose round it",
    ],
)
def test_a_reply_in_the_dataset_s_form_is_read_into_its_four_fields(reply: str) -> None:
    assert fields_from_reply(reply, FIELDS) == READ


@pytest.mark.parametrize(
    ("written", "read"),
    [
        ("09/02/2031 6:41:02 PM", "09/02/2031"),
        ("09 FEB 2031", "09/02/2031"),
        ("2031-02-09", "09/02/2031"),
        ("09-02-31", "09/02/2031"),
    ],
)
def test_a_date_is_put_in_one_form_however_the_receipt_prints_it(written: str, read: str) -> None:
    """A time after the date is set aside: the benchmark's own rule called such a date a wrong answer."""
    assert fields_from_reply(PLAIN.replace("09/02/2031", written), FIELDS)["DATE"] == read


@pytest.mark.parametrize(
    ("written", "read"),
    [("RM 31.07", "RM 31.07"), ("31.07 RM", "31.07 RM"), ("$31.07", "$31.07"), ("1,231.07", "1231.07")],
)
def test_a_total_keeps_its_currency_mark_which_the_scorer_sets_aside(written: str, read: str) -> None:
    from common.scoring import score_field

    cleaned = fields_from_reply(PLAIN.replace("total: 31.07", f"total: {written}"), FIELDS)["TOTAL"]

    assert cleaned == read
    assert (
        score_field(cleaned, written.replace("RM", "").replace("$", "").strip(), "TOTAL")["f1_score"] == 1.0
    )


def test_a_field_left_out_or_declined_as_the_prompt_asks_is_not_found() -> None:
    assert fields_from_reply("company: QUILL & FEATHER BOOKS SDN BHD\ntotal: 31.07\n", FIELDS) == {
        **READ,
        "DATE": "NOT_FOUND",
        "ADDRESS": "NOT_FOUND",
    }
    declined = fields_from_reply("company: NOT_FOUND\ndate: n/a\naddress: N/A\ntotal: null\n", FIELDS)
    assert set(declined.values()) == {"NOT_FOUND"}


def test_lines_that_begin_with_a_dash_are_read() -> None:
    """As the old task's parser read them (tests/common/test_a_line_that_begins_with_a_dash_is_read.py)."""
    with_dashes = "".join(f"- {line}\n" for line in PLAIN.splitlines())

    assert fields_from_reply(with_dashes, FIELDS) == READ


def test_two_things_the_old_task_s_parser_read_and_the_pipeline_s_does_not() -> None:
    """Written down as they are, and not mended here: each is a choice, and the parser is every type's.

    The old parser took the first of two answers to one field, where the pipeline's takes the last, and
    read "=" for the colon. No recorded run of the dataset kept its replies, so how often a model
    writes either is not known: the first run of the dataset on the cards counts them.
    """
    assert fields_from_reply(PLAIN + "total: 99.00\n", FIELDS)["TOTAL"] == "99.00"
    assert fields_from_reply(PLAIN.replace("total: 31.07", "total = 31.07"), FIELDS)["TOTAL"] == "NOT_FOUND"


def test_an_address_read_to_the_letter_scores_one() -> None:
    """The cleaner leaves an address its commas, and so does the answer (tests/common/
    test_an_address_keeps_its_commas.py).
    """
    from common.scoring import score_field

    answer = "12, JALAN ROOK 4, TAMAN HERON, 81750 MASAI, JOHOR"

    assert score_field(fields_from_reply(PLAIN, FIELDS)["ADDRESS"], answer, "ADDRESS")["f1_score"] == 1.0
