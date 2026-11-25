"""A page's classification says how its type was reached, so a page given the fallback type can be seen.

A page whose reply the rules could make nothing of is given the fallback type. Until 25 November 2026
its record then looked like that of any other page of that type: the type, and a ``confidence`` of 1.0
that every page had. Nothing said the type was a fallback, and nothing counted such pages.

Each record now says how its type was reached (``type_from``), and the stage counts the pages by it and
warns of those given the fallback type.

The pages are blank images and every reply is invented.
"""

import logging
import types
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from PIL import Image

from common import classification
from common.engine_calls import EngineWatch
from common.prompt_catalog import PromptCatalog
from models.orchestrator import DocumentOrchestrator
from stages import classify as classify_stage

REPO = Path(__file__).resolve().parents[2]

A_STATEMENT = (
    "1. COLUMNS: Date | Description | Debit | Credit | Balance\n2. PAID: NO\n3. ROWS: 14\n4. TRAVEL: NO"
)
A_RECEIPT = "1. NONE\n2. YES\n3. 6\n4. NO"
NO_RULE_FITS = "1. COLUMNS: NONE\n2. PAID: NO\n3. ROWS: 0\n4. TRAVEL: NO"
NO_ANSWERS = "This looks like a page of somebody's accounts."


@pytest.fixture(autouse=True)
def the_classification_file_is_read_afresh() -> Iterator[None]:
    classification._settings.cache_clear()
    yield
    classification._settings.cache_clear()


@pytest.fixture
def with_no_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The shipped classification file, but for its default: a page no rule fits has no type from it."""
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    for source in (REPO / "prompts").glob("*.yaml"):
        (prompts / source.name).write_text(source.read_text())
    detection = yaml.safe_load((prompts / "document_type_detection.yaml").read_text())
    detection["classification_evidence"]["default"] = "none"
    (prompts / "document_type_detection.yaml").write_text(yaml.safe_dump(detection, sort_keys=False))
    catalog = PromptCatalog(prompts_dir=prompts)
    monkeypatch.setattr(classification, "PromptCatalog", lambda: catalog)
    classification._settings.cache_clear()


# --- how a reply's type is reached ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "kind", "how"),
    [
        (A_STATEMENT, "BANK_STATEMENT", "rule"),
        (A_RECEIPT, "RECEIPT", "rule"),
        (NO_RULE_FITS, "INVOICE", "default"),
        (NO_ANSWERS, "UNIVERSAL", "fallback_no_answers"),
        ("", "UNIVERSAL", "fallback_no_answers"),
    ],
    ids=["a rule, labelled", "a rule, by position", "the file's default", "prose", "nothing"],
)
def test_a_reply_gives_its_type_and_how_the_type_was_reached(reply: str, kind: str, how: str) -> None:
    assert classification.typed(reply, "UNIVERSAL") == (kind, how)
    # The type alone is what it always was.
    assert classification.document_type(reply, "UNIVERSAL") == kind


def test_with_no_default_a_page_no_rule_fits_is_given_the_fallback_type_and_says_so(
    with_no_default: None,
) -> None:
    assert classification.typed(NO_RULE_FITS, "UNIVERSAL") == ("UNIVERSAL", "fallback_no_rule")
    assert classification.typed(A_STATEMENT, "UNIVERSAL") == ("BANK_STATEMENT", "rule")
    assert classification.typed(NO_ANSWERS, "UNIVERSAL") == ("UNIVERSAL", "fallback_no_answers")


def test_every_way_a_type_is_reached_has_a_name() -> None:
    assert classification.REACHED_BY == (
        "rule",
        "default",
        "fallback_no_answers",
        "fallback_no_rule",
        "fallback_not_asked",
        # A page of a dataset that says what its pages are: the model is not asked.
        "dataset",
    )


# --- the record -----------------------------------------------------------------------------------------


class Backend:
    """Answers each page with the next of the replies it was given, however the pages are grouped."""

    def __init__(self, replies: list[str], refuses: bool = False) -> None:
        self.replies, self.refuses, self.served = replies, refuses, 0

    def chat(self, images: list, prompts: list[str], max_tokens: int) -> list[str]:
        if self.refuses:
            raise RuntimeError("the engine refused the page")
        given = [self.replies[(self.served + i) % len(self.replies)] for i in range(len(images))]
        self.served += len(images)
        return given


def processor(backend: Backend) -> DocumentOrchestrator:
    made = object.__new__(DocumentOrchestrator)
    made._backend = backend  # type: ignore[assignment]
    made._fallback_type = "UNIVERSAL"
    made.app_config = types.SimpleNamespace(get_token_budget=lambda name: 150)  # type: ignore[assignment]
    return made


@pytest.fixture
def pages(tmp_path: Path) -> list[str]:
    paths = []
    for number in (1, 2, 3, 4):
        path = tmp_path / f"page_{number}.png"
        Image.new("RGB", (8, 8), "white").save(path)
        paths.append(str(path))
    return paths


def classified(backend: Backend, pages: list[str]) -> list[dict]:
    return classify_stage._classify_images(
        processor(backend), pages, batch_size=4, verbose=False, watch=EngineWatch(100)
    )


def test_each_record_says_how_its_pages_type_was_reached(pages: list[str]) -> None:
    records = classified(Backend([A_STATEMENT, A_RECEIPT, NO_RULE_FITS, NO_ANSWERS]), pages)

    assert [(record["document_type"], record["type_from"]) for record in records] == [
        ("BANK_STATEMENT", "rule"),
        ("RECEIPT", "rule"),
        ("INVOICE", "default"),
        ("UNIVERSAL", "fallback_no_answers"),
    ]
    # A reply the rules could make nothing of is not an error: the model answered. It is a fallback.
    assert all("error" not in record for record in records)
    assert records[3]["raw_response"] == NO_ANSWERS


def test_a_page_the_model_could_not_be_asked_about_says_so(pages: list[str]) -> None:
    (record,) = classified(Backend([], refuses=True), pages[:1])

    assert record["document_type"] == "UNIVERSAL" and record["type_from"] == "fallback_not_asked"
    assert record["error"] == "the engine refused the page"


# --- the count ------------------------------------------------------------------------------------------


def test_the_stage_counts_the_pages_by_how_their_types_were_reached_and_names_the_fallbacks(
    pages: list[str], caplog: pytest.LogCaptureFixture
) -> None:
    records = classified(Backend([A_STATEMENT, NO_ANSWERS, NO_RULE_FITS, NO_ANSWERS]), pages)

    with caplog.at_level(logging.INFO):
        classify_stage._say_how_the_types_were_reached(records)

    assert "How the types were reached: default 1, fallback_no_answers 2, rule 1" in caplog.text
    warning = next(record for record in caplog.records if record.levelno == logging.WARNING).getMessage()
    assert "2 of 4 page(s) were given the fallback type" in warning
    assert "page_2.png, page_4.png" in warning and "type_from" in warning


def test_a_run_with_no_fallback_is_counted_and_not_warned_of(
    pages: list[str], caplog: pytest.LogCaptureFixture
) -> None:
    records = classified(Backend([A_STATEMENT, A_RECEIPT]), pages)

    with caplog.at_level(logging.INFO):
        classify_stage._say_how_the_types_were_reached(records)

    assert "How the types were reached: rule 4" in caplog.text
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


def test_many_fallbacks_are_named_in_part(caplog: pytest.LogCaptureFixture) -> None:
    records = [{"image_name": f"page_{n}.png", "type_from": "fallback_no_answers"} for n in range(14)]

    with caplog.at_level(logging.WARNING):
        classify_stage._say_how_the_types_were_reached(records)

    assert "14 of 14 page(s)" in caplog.text and "page_9.png and 4 more" in caplog.text
    assert "page_10.png" not in caplog.text
