"""Every document type is read by one routine, a bank statement included.

``DocumentOrchestrator.read_pages`` asks for the fields of a group of pages of one type in one engine
call. The extract stage calls it for a group, and for a page on its own as a group of one. Each reply
must reach the extraction record as the model wrote it.

The content below is invented for these tests and appears in no evaluation set.
"""

import json
import types
from pathlib import Path

import pytest
import yaml
from PIL import Image

from common.extraction_batching import plan_extraction_batches, read_batch_sizes
from common.prompt_catalog import PromptCatalog
from models.backend import GenerationParams
from models.backends.vllm_backend import VllmBackend
from models.orchestrator import DocumentOrchestrator
from stages import extract as extract_stage

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_CONFIG = REPO_ROOT / "config" / "run_config.yml"
MODEL = "gemma4-31b-w4a16-vllm"


def reply(number: int) -> str:
    return (
        "DOCUMENT_TYPE: BANK_STATEMENT\n"
        f"STATEMENT_DATE_RANGE: 01/0{number}/2031 - 28/0{number}/2031\n"
        f"TRANSACTION_DATES: 04/0{number}/2031\n"
        "LINE_ITEM_DESCRIPTIONS: EFTPOS QUILLFEATHER BOOKS\n"
        f"TRANSACTION_AMOUNTS_PAID: ${number}1.07\n"
    )


class FakeBackend:
    """Keeps every engine call, by the form it was made in, and answers with numbered replies."""

    def __init__(self, fail_groups: bool = False, fail_always: bool = False) -> None:
        self.calls: list[dict] = []
        self.fail_groups = fail_groups
        self.fail_always = fail_always
        self.served = 0

    def _answer(self, form: str, images, prompts, params) -> list[str]:
        if self.fail_always or (self.fail_groups and len(images) > 1):
            raise RuntimeError("no reply")
        self.calls.append(
            {"form": form, "prompts": prompts, "max_tokens": params.max_tokens, "n": len(images)}
        )
        replies = [reply(self.served + i + 1) for i in range(len(images))]
        self.served += len(images)
        return replies

    def generate_many(self, images, prompts, params) -> list[str]:
        return self._answer("many", images, prompts, params)

    def generate_batch(self, images, prompts, params) -> list[str]:
        return self._answer("batch", images, prompts, params)


def orchestrator(backend: FakeBackend) -> DocumentOrchestrator:
    """An orchestrator holding only what reading pages needs, without loading a model."""
    orch = object.__new__(DocumentOrchestrator)
    orch._backend = backend  # type: ignore[assignment]
    orch._model_type_key = MODEL
    orch.debug = False
    orch.field_list = ["DOCUMENT_TYPE", "TOTAL_AMOUNT"]
    orch.document_field_lists = {"receipt": ["DOCUMENT_TYPE", "SUPPLIER_NAME", "TOTAL_AMOUNT"]}
    orch.gen_config = {"max_new_tokens_base": 500, "max_new_tokens_per_field": 50}
    orch.app_config = types.SimpleNamespace(
        get_token_budget=lambda name: {
            "extract_bank": 4096,
            "fallback_base": 512,
            "fallback_per_field": 64,
        }[name]
    )  # type: ignore[assignment]
    return orch


class ListWriter:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def write(self, record: dict) -> None:
        self.records.append(record)


@pytest.fixture
def pages(tmp_path: Path) -> list[Path]:
    paths = []
    for number in (1, 2, 3):
        path = tmp_path / f"page_{number}.png"
        Image.new("RGB", (8, 8), "white").save(path)
        paths.append(path)
    return paths


def classified(paths: list[Path], kind: str = "BANK_STATEMENT") -> list[dict]:
    return [{"image_path": str(p), "image_name": p.name, "document_type": kind} for p in paths]


# --- the routine --------------------------------------------------------------


def test_statements_are_one_call_with_the_statement_prompt_and_its_fixed_budget(pages):
    backend = FakeBackend()

    replies = orchestrator(backend).read_pages([str(p) for p in pages], "BANK_STATEMENT")

    schema_prompt = PromptCatalog().get_prompt("bank", "schema_fallback_extraction")
    assert backend.calls == [{"form": "many", "prompts": [schema_prompt] * 3, "max_tokens": 4096, "n": 3}]
    assert replies == [reply(1), reply(2), reply(3)]


def test_other_types_are_one_call_with_the_prompt_of_their_name_and_a_budget_by_their_fields(pages):
    backend = FakeBackend()

    replies = orchestrator(backend).read_pages([str(p) for p in pages], "RECEIPT")

    receipt_prompt = PromptCatalog().get_prompt(MODEL, "receipt")
    assert backend.calls == [
        {"form": "batch", "prompts": [receipt_prompt] * 3, "max_tokens": 500 + 3 * 50, "n": 3}
    ]
    assert len(replies) == 3


def test_a_page_on_its_own_is_asked_as_a_group_of_one_is(pages):
    alone, together = FakeBackend(), FakeBackend()

    orchestrator(alone).read_pages([str(pages[0])], "BANK_STATEMENT")
    orchestrator(together).read_pages([str(p) for p in pages], "BANK_STATEMENT")

    assert alone.calls[0]["form"] == together.calls[0]["form"]
    assert alone.calls[0]["prompts"] == together.calls[0]["prompts"][:1]
    assert alone.calls[0]["max_tokens"] == together.calls[0]["max_tokens"]


def test_no_pages_makes_no_call():
    backend = FakeBackend()

    assert orchestrator(backend).read_pages([], "RECEIPT") == []
    assert backend.calls == []


# --- the backend: a shared request is built as the single request is -----------


class FakeEngine:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return [
            types.SimpleNamespace(
                outputs=[types.SimpleNamespace(text=f" reply {i} ", token_ids=[1])],
                prompt_token_ids=[1],
                num_cached_tokens=0,
            )
            for i, _ in enumerate(kwargs["messages"])
        ]


def gemma_like_backend(engine: FakeEngine) -> VllmBackend:
    return VllmBackend(
        engine=engine,
        model_type_key=MODEL,
        chat_template_kwargs={"enable_thinking": False},
        default_image_first=True,
    )


def test_generate_many_builds_each_request_as_generate_does(monkeypatch):
    monkeypatch.setitem(
        __import__("sys").modules,
        "vllm",
        types.SimpleNamespace(SamplingParams=lambda **kwargs: kwargs),
    )
    image = Image.new("RGB", (8, 8), "white")
    params = GenerationParams(max_tokens=77, extra={})
    single_engine, many_engine = FakeEngine(), FakeEngine()

    gemma_like_backend(single_engine).generate(image, "read this", params)
    texts = gemma_like_backend(many_engine).generate_many(
        [image, image], ["read this", "read this"], params
    )

    single, many = single_engine.calls[0], many_engine.calls[0]
    assert texts == ["reply 0", "reply 1"]
    assert many["messages"] == [single["messages"], single["messages"]]
    assert many["messages"][0][0]["content"][0]["type"] == "image_url"  # image before text
    assert many["chat_template_kwargs"] == single["chat_template_kwargs"] == {"enable_thinking": False}
    assert many["sampling_params"] == single["sampling_params"] == {"max_tokens": 77, "temperature": 0}


def test_generate_many_refuses_mismatched_lists():
    backend = gemma_like_backend(FakeEngine())
    image = Image.new("RGB", (8, 8), "white")

    with pytest.raises(ValueError) as caught:
        backend.generate_many([image, image], ["only one"], GenerationParams(max_tokens=5, extra={}))

    message = str(caught.value)
    assert (
        "What:" in message and "Where:" in message and "Expected:" in message and "How to fix:" in message
    )


# --- the stage ----------------------------------------------------------------


def test_statements_are_planned_into_shared_calls_like_any_type(pages):
    batches = plan_extraction_batches(classified(pages), batch_sizes={"bank_statement": 2, "default": 10})

    assert [len(batch) for batch in batches] == [2, 1]


def test_a_group_writes_one_record_per_page_as_pages_alone_do(pages):
    group_writer, alone_writer = ListWriter(), ListWriter()

    extract_stage._read_batch(group_writer, orchestrator(FakeBackend()), classified(pages))
    processor = orchestrator(FakeBackend())
    for one in classified(pages):
        extract_stage._read_batch(alone_writer, processor, [one])

    assert [r["image_name"] for r in group_writer.records] == [p.name for p in pages]
    assert [r["raw_response"] for r in group_writer.records] == [reply(1), reply(2), reply(3)]
    for grouped, alone in zip(group_writer.records, alone_writer.records, strict=True):
        assert set(grouped) == set(alone)
        assert grouped["raw_response"] == alone["raw_response"]
        assert grouped["prompt_used"] == alone["prompt_used"] == "unified_bank_single_pass"
        assert grouped["error"] is None
    shares = {r["processing_time"] for r in group_writer.records}
    assert len(shares) == 1 and shares.pop() >= 0


def test_a_record_names_the_prompt_of_its_type(pages):
    writer = ListWriter()

    extract_stage._read_batch(writer, orchestrator(FakeBackend()), classified(pages, "RECEIPT"))

    assert {r["prompt_used"] for r in writer.records} == {"receipt"}


def test_a_failed_group_is_read_again_one_page_at_a_time(pages, caplog):
    writer, backend = ListWriter(), FakeBackend(fail_groups=True)

    with caplog.at_level("WARNING"):
        extract_stage._read_batch(writer, orchestrator(backend), classified(pages))

    assert [call["n"] for call in backend.calls] == [1, 1, 1]
    assert [r["image_name"] for r in writer.records] == [p.name for p in pages]
    assert all(r["error"] is None for r in writer.records)
    assert "retrying one at a time" in caplog.text


def test_a_page_that_fails_alone_is_recorded_as_an_error(pages):
    writer = ListWriter()

    extract_stage._read_batch(writer, orchestrator(FakeBackend(fail_always=True)), classified(pages))

    assert [r["image_name"] for r in writer.records] == [p.name for p in pages]
    assert [r["prompt_used"] for r in writer.records] == ["error"] * 3
    assert all("no reply" in r["error"] for r in writer.records)


def test_records_round_trip_as_json(pages):
    writer = ListWriter()

    extract_stage._read_batch(writer, orchestrator(FakeBackend()), classified(pages))

    for record in writer.records:
        assert json.loads(json.dumps(record)) == record
        assert record["raw_response"].endswith("\n")


# --- the shipped config --------------------------------------------------------


def test_the_shipped_bank_batch_size_is_stated_and_explained():
    sizes = read_batch_sizes(yaml.safe_load(RUN_CONFIG.read_text()), config_path=RUN_CONFIG)
    text = RUN_CONFIG.read_text()

    assert sizes["bank_statement"] >= 1
    assert "it is NOT in effect today" not in text
    assert "one independent" in text.split("batching:")[1].split("processing:")[0]
