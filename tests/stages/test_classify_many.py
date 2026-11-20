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
    """Records every engine call; replays REPLIES in order for both call shapes."""

    def __init__(self, fail_many: bool = False) -> None:
        self.single_calls: list[dict] = []
        self.many_calls: list[dict] = []
        self.fail_many = fail_many

    def generate(self, image, prompt, params) -> str:
        self.single_calls.append({"prompt": prompt, "max_tokens": params.max_tokens})
        return REPLIES[(len(self.single_calls) - 1) % len(REPLIES)]

    def generate_many(self, images, prompts, params) -> list[str]:
        if self.fail_many:
            raise RuntimeError("engine refused the batch")
        self.many_calls.append({"prompts": prompts, "max_tokens": params.max_tokens, "n": len(images)})
        return [REPLIES[i % len(REPLIES)] for i in range(len(images))]


def orchestrator(backend: FakeBackend) -> DocumentOrchestrator:
    """An orchestrator holding only what detection reads, without loading a model."""
    orch = object.__new__(DocumentOrchestrator)
    orch._backend = backend  # type: ignore[assignment]
    orch._has_oom_recovery = False
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
    assert backend.single_calls == []
    assert backend.many_calls == [{"prompts": [prompt] * 3, "max_tokens": 150, "n": 3}]
    assert len(results) == 3


def test_detect_many_gives_each_page_the_result_the_single_call_gives(pages):
    one_backend, many_backend = FakeBackend(), FakeBackend()
    one = orchestrator(one_backend)

    singles = [one.detect_and_classify_document(page) for page in pages]
    many = orchestrator(many_backend).detect_many(pages)

    assert many == singles
    assert [r["raw_response"] for r in many] == REPLIES
    assert [r["document_type"] for r in many] == ["BANK_STATEMENT", "RECEIPT", "BANK_STATEMENT"]
    assert one_backend.single_calls[0] == {
        "prompt": many_backend.many_calls[0]["prompts"][0],
        "max_tokens": many_backend.many_calls[0]["max_tokens"],
    }


def test_detect_many_of_nothing_makes_no_call():
    backend = FakeBackend()

    assert orchestrator(backend).detect_many([]) == []
    assert backend.many_calls == []


# --- the stage ------------------------------------------------------------------


def test_pages_are_classified_in_shared_calls_of_the_configured_size(pages):
    backend = FakeBackend()

    records = classify_stage._classify_images(orchestrator(backend), pages, batch_size=2, verbose=False)

    assert [call["n"] for call in backend.many_calls] == [2, 1]
    assert backend.single_calls == []
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

    assert backend.many_calls == []
    assert len(backend.single_calls) == 3


def test_a_failed_shared_call_is_retried_one_page_at_a_time(pages, caplog):
    backend = FakeBackend(fail_many=True)

    with caplog.at_level("WARNING"):
        records = classify_stage._classify_images(orchestrator(backend), pages, batch_size=3, verbose=False)

    assert len(backend.single_calls) == 3
    assert [r["document_type"] for r in records] == ["BANK_STATEMENT", "RECEIPT", "BANK_STATEMENT"]
    assert "retrying one at a time" in caplog.text


def test_a_backend_without_a_shared_call_keeps_one_call_a_page(pages):
    class SingleOnly:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, image, prompt, params) -> str:
            self.calls += 1
            return REPLIES[1]

    backend = SingleOnly()

    records = classify_stage._classify_images(orchestrator(backend), pages, batch_size=3, verbose=False)

    assert backend.calls == 3
    assert len(records) == 3


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
