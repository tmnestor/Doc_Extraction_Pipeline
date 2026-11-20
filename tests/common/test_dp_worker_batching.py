"""The two-engine path groups pages into engine calls exactly as the one-engine path does.

A model small enough for one card runs as one engine on each card, each engine taking half the pages. Until
20 November 2026 that path classified one page at a time and read bank statements one at a time, even when
statements were read single pass and could share a call, as the one-engine path already did. A run on two
engines was then slower than a run on one, for no reason of the model's.

The two paths must make the same calls and write the same records, or a timing taken on one says nothing
about the other. So the two-engine helpers are checked against the one-engine stage's own functions.

The statement content below is invented for these tests and appears in no evaluation set.
"""

import ast
from pathlib import Path

import pytest
from PIL import Image

from common import vllm_dp_workers as workers
from common.extraction_batching import plan_extraction_batches, read_batch_sizes
from common.pipeline_config import load_yaml_config
from common.unified_bank_extractor import UnifiedBankExtractor
from stages import extract as extract_stage

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_CONFIG = REPO_ROOT / "config" / "run_config.yml"


def reply(number: int) -> str:
    return (
        "DOCUMENT_TYPE: BANK_STATEMENT\n"
        f"STATEMENT_DATE_RANGE: 01/0{number}/2031 - 28/0{number}/2031\n"
        f"TRANSACTION_DATES: 04/0{number}/2031\n"
        "LINE_ITEM_DESCRIPTIONS: EFTPOS QUILLFEATHER BOOKS\n"
        f"TRANSACTION_AMOUNTS_PAID: ${number}1.07\n"
    )


class FakeMany:
    """Stands in for ``processor.generate_many``."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[int] = []
        self.fail = fail

    def __call__(self, images, prompts, max_tokens=None, extra=None) -> list[str]:
        if self.fail:
            raise RuntimeError("engine refused the batch")
        self.calls.append(len(images))
        return [reply(i + 1) for i in range(len(images))]


class FakeOne:
    """Stands in for ``processor.generate``."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, image, prompt, max_tokens=None, extra=None) -> str:
        self.calls += 1
        return reply(self.calls)


class FakeClassifier:
    """Stands in for the document processor during classification."""

    supports_generate_many = True

    def __init__(self, fail_batches: bool = False) -> None:
        self.batches: list[int] = []
        self.singles = 0
        self.fail_batches = fail_batches

    def detect_many(self, image_paths, verbose=False):
        if self.fail_batches:
            raise RuntimeError("engine refused the batch")
        self.batches.append(len(image_paths))
        return [{"document_type": "RECEIPT", "raw_response": Path(p).name} for p in image_paths]

    def detect_and_classify_document(self, image_path, verbose=False):
        self.singles += 1
        return {"document_type": "RECEIPT", "raw_response": Path(image_path).name}


@pytest.fixture
def statements(tmp_path: Path) -> list[Path]:
    paths = []
    for number in range(1, 6):
        path = tmp_path / f"statement_{number}.png"
        Image.new("RGB", (8, 8), "white").save(path)
        paths.append(path)
    return paths


def bank(many=None, one=None) -> UnifiedBankExtractor:
    return UnifiedBankExtractor(
        generate_fn=one or FakeOne(), generate_many_fn=many, verbose=False, extract_tokens=4096
    )


def classified(paths: list[Path]) -> list[dict]:
    return [{"image_path": str(p), "image_name": p.name, "document_type": "BANK_STATEMENT"} for p in paths]


def plan(records: list[dict], adapter) -> list[list[dict]]:
    _, raw = load_yaml_config(RUN_CONFIG)
    return plan_extraction_batches(
        records,
        batch_sizes=read_batch_sizes(raw, config_path=RUN_CONFIG),
        unbatchable=extract_stage._unbatchable_types(adapter),
    )


# --- classification -----------------------------------------------------------


def test_a_workers_pages_are_classified_several_to_a_call() -> None:
    processor = FakeClassifier()
    pages = [f"/pages/page_{n:02d}.png" for n in range(10)]

    records = workers._classify_records(processor, pages, 4, False)

    assert processor.batches == [4, 4, 2] and processor.singles == 0
    assert [r["image_path"] for r in records] == pages
    assert [r["raw_response"] for r in records] == [Path(p).name for p in pages]


def test_each_classification_carries_an_equal_share_of_the_time() -> None:
    """The parent sums these for each worker to report inference apart from wall clock."""
    records = workers._classify_records(FakeClassifier(), [f"/pages/{n}.png" for n in range(6)], 3, False)

    times = {r["processing_time"] for r in records}
    assert len(times) == 1 and next(iter(times)) >= 0.0


def test_a_classify_batch_that_fails_is_retried_one_page_at_a_time() -> None:
    processor = FakeClassifier(fail_batches=True)

    records = workers._classify_records(processor, [f"/pages/{n}.png" for n in range(5)], 4, False)

    assert processor.singles == 5 and len(records) == 5


def test_no_pages_give_no_records() -> None:
    assert workers._classify_records(FakeClassifier(), [], 4, False) == []


# --- bank statements ----------------------------------------------------------


def test_single_pass_statements_share_a_call_on_the_two_engine_path(statements) -> None:
    many = FakeMany()
    adapter = bank(many=many)
    batches = plan(classified(statements), adapter)

    assert [len(batch) for batch in batches] == [5]
    records = workers._bank_batch_records(adapter, batches[0])

    assert many.calls == [5]
    assert [r["image_name"] for r in records] == [p.name for p in statements]
    assert all(r["error"] is None and r["prompt_used"].startswith("unified_bank_") for r in records)


def test_without_a_batched_call_statements_are_read_one_at_a_time(statements) -> None:
    batches = plan(classified(statements), bank(many=None))

    assert [len(batch) for batch in batches] == [1] * 5


def test_the_two_paths_write_the_same_record_for_a_statement(statements) -> None:
    """Apart from the time taken, which is a measurement and not a result."""
    batch = classified(statements)
    by_worker = workers._bank_batch_records(bank(many=FakeMany()), batch)
    by_stage = workers._Records()
    extract_stage._extract_bank_single_pass_batch(by_stage, bank(many=FakeMany()), batch)

    def without_time(records: list[dict]) -> list[dict]:
        return [{k: v for k, v in r.items() if k != "processing_time"} for r in records]

    assert without_time(by_worker) == without_time(by_stage.records)
    assert len(by_worker) == 5


def test_a_bank_batch_that_fails_is_retried_one_statement_at_a_time(statements) -> None:
    one = FakeOne()
    adapter = bank(many=FakeMany(fail=True), one=one)

    records = workers._bank_batch_records(adapter, classified(statements))

    assert one.calls == 5 and len(records) == 5 and all(r["error"] is None for r in records)


# --- the worker itself --------------------------------------------------------


def worker_source(name: str) -> str:
    tree = ast.parse(Path(workers.__file__).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.unparse(node)


def test_the_extract_worker_uses_the_one_engine_paths_own_rule_and_calls() -> None:
    """The fault this guards against is a rule changed in one path and missed in the other."""
    source = worker_source("classified_extract_worker")

    assert "generate_many_fn=" in source  # the bank reader is given the batched call
    assert "_unbatchable_types(bank_adapter)" in source  # not the bare list of types
    assert "unbatchable=UNBATCHABLE_TYPES" not in source
    assert "_bank_batch_records(bank_adapter, batch)" in source


def test_the_classify_worker_batches_by_the_configured_size() -> None:
    source = worker_source("classify_worker")

    assert "_classify_records(" in source and "_classify_batch_size(" in source
    assert "detect_and_classify_document" not in source  # no loop of single calls left in the worker
