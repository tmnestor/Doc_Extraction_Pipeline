"""Classification pages share an engine call, so the engine can run them concurrently.

Classifying a page is one independent model call. Sent one at a time, the
engine never has more than one page in flight; sent together, it runs up to
max_num_seqs of them at once. Each request must be built exactly as the
one-at-a-time call builds it and each reply parsed by the same rules, or a
batched run would not classify the way an unbatched one does.

The replies below are invented for these tests.
"""

import types
from pathlib import Path

import pytest
import yaml
from PIL import Image

from common.extraction_batching import BatchingConfigError, read_classify_batch_size
from models.orchestrator import DocumentOrchestrator
from stages import classify as classify_stage

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_CONFIG = REPO_ROOT / "config" / "run_config.yml"
DETECTION_FILE = REPO_ROOT / "prompts" / "document_type_detection.yaml"

REPLIES = [
    "COLUMNS: Date | Description | Debit | Credit | Balance\nPAID: NO\nROWS: 14\nTRAVEL: NO",
    "This is a receipt.",
    "1. Date | Details | Withdrawals | Deposits | Balance\n2. NO\n3. 9\n4. NO",
]


class FakeBackend:
    """Records every engine call; replays REPLIES in the order pages are served, however grouped."""

    def __init__(self, fail_shared: bool = False) -> None:
        self.many_calls: list[dict] = []
        self.fail_shared = fail_shared
        self.served = 0

    def chat(self, images, prompts, max_tokens) -> list[str]:
        if self.fail_shared and len(images) > 1:
            raise RuntimeError("engine refused the batch")
        self.many_calls.append({"prompts": prompts, "max_tokens": max_tokens, "n": len(images)})
        replies = [REPLIES[(self.served + i) % len(REPLIES)] for i in range(len(images))]
        self.served += len(images)
        return replies


def orchestrator(backend: FakeBackend) -> DocumentOrchestrator:
    """An orchestrator holding only what detection reads, without loading a model."""
    orch = object.__new__(DocumentOrchestrator)
    orch._backend = backend  # type: ignore[assignment]
    orch._fallback_type = "INVOICE"
    orch.debug = False
    orch.prompt_config = {"detection_file": str(DETECTION_FILE), "detection_key": "detection"}
    orch.app_config = types.SimpleNamespace(get_token_budget=lambda name: {"classify": 150}[name])  # type: ignore[assignment]
    return orch


@pytest.fixture
def pages(tmp_path: Path) -> list[str]:
    paths = []
    for number in (1, 2, 3):
        path = tmp_path / f"page_{number}.png"
        Image.new("RGB", (8, 8), "white").save(path)
        paths.append(str(path))
    return paths


# --- the orchestrator -----------------------------------------------------------


def test_detect_many_is_one_call_with_the_detection_prompt_for_every_page(pages):
    backend = FakeBackend()

    results = orchestrator(backend).detect_many(pages)

    prompt = yaml.safe_load(DETECTION_FILE.read_text())["prompts"]["detection"]["prompt"]
    assert backend.many_calls == [{"prompts": [prompt] * 3, "max_tokens": 150, "n": 3}]
    assert len(results) == 3


def test_detect_many_gives_each_page_the_result_the_single_call_gives(pages):
    one_backend, many_backend = FakeBackend(), FakeBackend()
    one = orchestrator(one_backend)

    singles = [one.detect_and_classify_document(page) for page in pages]
    many = orchestrator(many_backend).detect_many(pages)

    assert many == singles
    assert [r["raw_response"] for r in many] == REPLIES
    assert [r["document_type"] for r in many] == [
        "BANK_STATEMENT",
        "INVOICE",
        "BANK_STATEMENT",
    ]  # prose is given the fallback type
    assert [call["n"] for call in one_backend.many_calls] == [1, 1, 1]
    assert one_backend.many_calls[0]["prompts"] == many_backend.many_calls[0]["prompts"][:1]
    assert one_backend.many_calls[0]["max_tokens"] == many_backend.many_calls[0]["max_tokens"]


def test_detect_many_of_nothing_makes_no_call():
    backend = FakeBackend()

    assert orchestrator(backend).detect_many([]) == []
    assert backend.many_calls == []


# --- the stage ------------------------------------------------------------------


def test_pages_are_classified_in_shared_calls_of_the_configured_size(pages):
    backend = FakeBackend()

    records = classify_stage._classify_images(orchestrator(backend), pages, batch_size=2, verbose=False)

    assert [call["n"] for call in backend.many_calls] == [2, 1]
    assert [r["image_path"] for r in records] == pages


def test_a_shared_call_writes_the_records_the_single_calls_write(pages):
    together = classify_stage._classify_images(
        orchestrator(FakeBackend()), pages, batch_size=3, verbose=False
    )
    one_by_one = classify_stage._classify_images(
        orchestrator(FakeBackend()), pages, batch_size=1, verbose=False
    )

    assert together == one_by_one
    assert set(together[0]) == {
        "image_path",
        "image_name",
        "document_type",
        "confidence",
        "raw_response",
        "prompt_used",
    }
    assert together[0]["image_name"] == "page_1.png"
    assert together[0]["prompt_used"] == "detection"


def test_a_batch_size_of_one_keeps_one_call_a_page(pages):
    backend = FakeBackend()

    classify_stage._classify_images(orchestrator(backend), pages, batch_size=1, verbose=False)

    assert [call["n"] for call in backend.many_calls] == [1, 1, 1]


def test_a_failed_shared_call_is_retried_one_page_at_a_time(pages, caplog):
    backend = FakeBackend(fail_shared=True)

    with caplog.at_level("WARNING"):
        records = classify_stage._classify_images(orchestrator(backend), pages, batch_size=3, verbose=False)

    assert [call["n"] for call in backend.many_calls] == [1, 1, 1]
    assert [r["document_type"] for r in records] == [
        "BANK_STATEMENT",
        "INVOICE",
        "BANK_STATEMENT",
    ]  # prose is given the fallback type
    assert "retrying one at a time" in caplog.text


def test_a_page_that_fails_alone_is_given_the_fallback_type_and_carries_the_error(pages):
    class Refuses:
        def chat(self, images, prompts, max_tokens) -> list[str]:
            raise RuntimeError("engine refused the page")

    result = orchestrator(Refuses()).detect_and_classify_document(pages[0])  # type: ignore[arg-type]

    assert result["document_type"] == "INVOICE"
    assert result["error"] == "engine refused the page"
    assert result["prompt_used"] == "fallback_heuristic"


# --- the config -----------------------------------------------------------------


def test_the_shipped_classify_batch_size_is_stated():
    size = read_classify_batch_size(yaml.safe_load(RUN_CONFIG.read_text()), config_path=RUN_CONFIG)

    assert size >= 1


@pytest.mark.parametrize("block", [{"default": 2}, {"default": 2, "classify": 0}, {"classify": True}, None])
def test_a_missing_or_invalid_classify_batch_size_fails_with_a_diagnostic(block):
    raw = {"pipeline": {"batching": block}} if block is not None else {"pipeline": {}}

    with pytest.raises(BatchingConfigError) as caught:
        read_classify_batch_size(raw, config_path=RUN_CONFIG)

    message = str(caught.value)
    assert "What:" in message and "classify" in message
    assert "Where:" in message and str(RUN_CONFIG) in message and "pipeline.batching" in message
    assert "Expected:" in message and "classify: 64" in message
    assert "How to fix:" in message


def test_a_reply_that_only_names_a_type_in_words_is_given_the_fallback_type() -> None:
    """The word-based reading belonged to an older prompt and is gone: nothing is guessed from words."""
    from common import classification

    for reply in ("This is a bank statement.", "INVOICE", "A receipt from a shop.", ""):
        assert classification.document_type(reply, "UNIVERSAL") == "UNIVERSAL"


def test_the_detection_file_holds_one_prompt_and_the_two_tables_the_rules_read() -> None:
    import yaml

    detection = yaml.safe_load(DETECTION_FILE.read_text())

    assert list(detection) == ["prompts", "column_roles", "classification_evidence"]
    assert list(detection["prompts"]) == ["detection"]
