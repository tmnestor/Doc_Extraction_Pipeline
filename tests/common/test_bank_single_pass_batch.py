"""Bank statements share an engine call, so the engine can run them concurrently.

A statement is read in one independent call, so several can be submitted
together. Each request must be built exactly as the one-at-a-time call builds
it (same prompt, same token budget, image before text, thinking suppressed),
and each reply must reach the extraction record unaltered, or a batched run
would not be the same measurement as an unbatched one.

The statement content below is invented for these tests and appears in no
evaluation set.
"""

import json
import types
from pathlib import Path

import pytest
import yaml
from PIL import Image

from common.extraction_batching import plan_extraction_batches, read_batch_sizes
from common.unified_bank_extractor import UnifiedBankExtractor, _default_catalog
from models.backend import GenerationParams
from models.backends.vllm_backend import VllmBackend
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
    """Stands in for ``processor.generate_many``: records the call, replays one reply per image."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[dict] = []
        self.fail = fail

    def __call__(self, images, prompts, max_tokens=None, extra=None) -> list[str]:
        if self.fail:
            raise RuntimeError("engine refused the batch")
        self.calls.append({"images": images, "prompts": prompts, "max_tokens": max_tokens, "extra": extra})
        return [reply(i + 1) for i in range(len(images))]


class FakeOne:
    """Stands in for ``processor.generate``."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, image, prompt, max_tokens=None, extra=None) -> str:
        self.prompts.append(prompt)
        return reply(len(self.prompts))


class ListWriter:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def write(self, record: dict) -> None:
        self.records.append(record)


@pytest.fixture
def statements(tmp_path: Path) -> list[Path]:
    paths = []
    for number in (1, 2, 3):
        path = tmp_path / f"statement_{number}.png"
        Image.new("RGB", (8, 8), "white").save(path)
        paths.append(path)
    return paths


def extractor(many=None, one=None) -> UnifiedBankExtractor:
    return UnifiedBankExtractor(generate_fn=one or FakeOne(), generate_many_fn=many, verbose=False)


def classifications(paths: list[Path]) -> list[dict]:
    return [{"image_path": str(p), "image_name": p.name, "document_type": "BANK_STATEMENT"} for p in paths]


# --- the extractor ------------------------------------------------------------


def test_a_batch_is_one_call_with_the_schema_prompt_for_every_statement(statements):
    many = FakeMany()
    bank = extractor(many=many)

    results = bank.extract_bank_statements_single_pass(statements)

    schema_prompt = _default_catalog().get_prompt("bank", "schema_fallback_extraction")
    assert len(many.calls) == 1
    assert many.calls[0]["prompts"] == [schema_prompt] * 3
    assert many.calls[0]["max_tokens"] == bank._extract_tokens
    assert len(results) == 3


def test_each_statement_keeps_its_own_reply_as_written(statements):
    results = extractor(many=FakeMany()).extract_bank_statements_single_pass(statements)

    for number, (fields, metadata) in enumerate(results, start=1):
        assert metadata["verbatim_response"] == reply(number)
        assert metadata["strategy_used"] == "single_pass"
        assert metadata["turns_executed"] == 1
        assert fields["TRANSACTION_AMOUNTS_PAID"] == f"${number}1.07"


def test_a_batched_statement_has_the_metadata_of_an_unbatched_one(statements):
    one = extractor().extract_bank_statement(statements[0])
    many = extractor(many=FakeMany()).extract_bank_statements_single_pass(statements[:1])[0]

    assert many == one


def test_statements_cannot_be_batched_without_a_batched_call(statements):
    with pytest.raises(ValueError) as caught:
        extractor(many=None).extract_bank_statements_single_pass(statements)

    message = str(caught.value)
    assert "What:" in message and "generate_many_fn" in message
    assert "Where:" in message and "stages/extract.py" in message
    assert "Expected:" in message and "generate_many" in message
    assert "How to fix:" in message


def test_can_batch_needs_a_batch_call():
    assert extractor(many=FakeMany()).can_batch
    assert not extractor(many=None).can_batch


# --- the backend: a batched request is built as the single request is -----------


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
        model_type_key="gemma4-31b-w4a16-vllm",
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


def test_single_pass_statements_are_planned_into_shared_calls(statements):
    records = classifications(statements)
    sizes = {"bank_statement": 2, "default": 10}

    batchable = plan_extraction_batches(
        records, batch_sizes=sizes, unbatchable=extract_stage._unbatchable_types(extractor(many=FakeMany()))
    )
    no_batched_call = plan_extraction_batches(
        records, batch_sizes=sizes, unbatchable=extract_stage._unbatchable_types(extractor(many=None))
    )
    no_adapter = plan_extraction_batches(
        records, batch_sizes=sizes, unbatchable=extract_stage._unbatchable_types(None)
    )

    assert [len(b) for b in batchable] == [2, 1]
    assert [len(b) for b in no_batched_call] == [1, 1, 1]
    assert [len(b) for b in no_adapter] == [1, 1, 1]


def test_a_batch_writes_one_record_per_statement_in_the_unbatched_shape(statements):
    batch_writer, single_writer = ListWriter(), ListWriter()
    records = classifications(statements)

    extract_stage._extract_bank_single_pass_batch(batch_writer, extractor(many=FakeMany()), records)
    for record in records:
        extract_stage._extract_bank_with_adapter(
            single_writer, extractor(), record["image_path"], record["image_name"], "BANK_STATEMENT"
        )

    assert [r["image_name"] for r in batch_writer.records] == [p.name for p in statements]
    assert [r["raw_response"] for r in batch_writer.records] == [reply(1), reply(2), reply(3)]
    for batched, single in zip(batch_writer.records, single_writer.records, strict=True):
        assert set(batched) == set(single)
        assert batched["prompt_used"] == single["prompt_used"] == "unified_bank_single_pass"
        assert batched["error"] is None
    shares = {r["processing_time"] for r in batch_writer.records}
    assert len(shares) == 1 and shares.pop() >= 0


def test_a_failed_batch_is_retried_one_statement_at_a_time(statements):
    writer, one = ListWriter(), FakeOne()
    bank = extractor(many=FakeMany(fail=True), one=one)

    extract_stage._extract_bank_single_pass_batch(writer, bank, classifications(statements))

    assert len(one.prompts) == 3
    assert [r["image_name"] for r in writer.records] == [p.name for p in statements]
    assert all(r["error"] is None for r in writer.records)


def test_a_statement_that_fails_on_retry_is_recorded_as_an_error(statements):
    class Broken(FakeOne):
        def __call__(self, image, prompt, max_tokens=None, extra=None) -> str:
            raise RuntimeError("no reply")

    writer = ListWriter()
    bank = extractor(many=FakeMany(fail=True), one=Broken())

    extract_stage._extract_bank_single_pass_batch(writer, bank, classifications(statements))

    assert [r["prompt_used"] for r in writer.records] == ["error"] * 3
    assert all("no reply" in r["error"] for r in writer.records)


# --- the shipped config --------------------------------------------------------


def test_the_shipped_bank_batch_size_is_stated_and_explained():
    sizes = read_batch_sizes(yaml.safe_load(RUN_CONFIG.read_text()), config_path=RUN_CONFIG)
    text = RUN_CONFIG.read_text()

    assert sizes["bank_statement"] >= 1
    assert "it is NOT in effect today" not in text
    assert "one independent" in text.split("batching:")[1].split("processing:")[0]


def test_records_round_trip_as_json(statements):
    writer = ListWriter()
    extract_stage._extract_bank_single_pass_batch(
        writer, extractor(many=FakeMany()), classifications(statements)
    )

    assert json.loads(json.dumps(writer.records)) == writer.records
