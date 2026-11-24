"""One engine across the cards and an engine on each card read pages through the same functions.

Until 21 November 2026 each stage held its reading twice: once in the stage, for one engine, and once in the
worker module, for an engine on each card. A rule changed in one copy and missed in the other made the
quantised 12B four times slower than it needed to be, and nothing failed. Now a stage has one function that
reads pages (`classify_pages`, `extract_pages`), and `common.vllm_dp` only starts an engine on a card and
calls it.

The page content below is invented for these tests and appears in no evaluation set.
"""

import ast
import contextlib
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reference"))

from model_inputs import InProcessContext, test_config  # noqa: E402

from common import vllm_dp  # noqa: E402
from common.settings import Settings  # noqa: E402
from stages.classify import classify_pages  # noqa: E402
from stages.extract import extract_pages  # noqa: E402

test_config.__test__ = False  # type: ignore[attr-defined]

KINDS = ("bank_statement", "invoice", "receipt")
MEASUREMENTS = ("processing_time", "gpu_id")


def statement_reply(name: str) -> str:
    return (
        "DOCUMENT_TYPE: BANK_STATEMENT\n"
        "STATEMENT_DATE_RANGE: 01/03/2031 - 28/03/2031\n"
        "TRANSACTION_DATES: 04/03/2031\n"
        f"LINE_ITEM_DESCRIPTIONS: EFTPOS QUILLFEATHER BOOKS {name}\n"
        "TRANSACTION_AMOUNTS_PAID: $31.07\n"
    )


class StandIn:
    """Stands in for the document processor: answers from the file name and notes every call made."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def detect_many(self, image_paths: list[str], verbose: bool = False) -> list[dict[str, Any]]:
        self.calls.append(("classify", [Path(p).name for p in image_paths]))
        return [self._type(p) for p in image_paths]

    def read_pages(self, image_paths: list[str], document_type: str) -> list[str]:
        self.calls.append(("extract", [Path(p).name for p in image_paths]))
        if document_type.upper() == "BANK_STATEMENT":
            return [statement_reply(Path(p).name) for p in image_paths]
        return [f"TOTAL_AMOUNT: read from {Path(p).name}" for p in image_paths]

    @staticmethod
    def _type(image_path: str) -> dict[str, Any]:
        kind = next(kind for kind in KINDS if kind in Path(image_path).name)
        return {"document_type": kind.upper(), "raw_response": Path(image_path).name}


class Bench:
    """A folder of invented pages, the settings to read them with, and the two arrangements to compare."""

    def __init__(self, folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.config_path = test_config(folder)
        self.folder = folder / "pages"
        self.folder.mkdir()
        self.overrides = {"data_dir": str(self.folder), "output_dir": str(folder / "out")}
        self.processor = StandIn()
        self.opened_with: list[Any] = []
        monkeypatch.setattr(vllm_dp.mp, "get_context", lambda method=None: InProcessContext)
        monkeypatch.setattr("common.pipeline_ops.open_processor", self._open)

    @contextlib.contextmanager
    def _open(self, app_config: Any) -> Iterator[StandIn]:
        self.opened_with.append(app_config)
        yield self.processor

    def pages(self, count: int, kinds: tuple[str, ...] = KINDS) -> list[Path]:
        """`count` pages of each kind, in the order the extract stage reads them: one kind after another."""
        made: list[Path] = []
        for kind in kinds:
            for number in range(count):
                path = self.folder / f"CASE{number:03d}_{kind}.png"
                Image.new("RGB", (8 + len(made), 8), "white").save(path)
                made.append(path)
        return made

    def classified(self, pages: list[Path]) -> list[dict[str, Any]]:
        return [
            {
                "image_path": str(page),
                "image_name": page.name,
                "document_type": next(kind for kind in KINDS if kind in page.name).upper(),
            }
            for page in pages
        ]

    def records(self, stage: str, pages: list[Path], *, engines: int) -> list[dict[str, Any]]:
        """What the stage's reading gives for these pages, without the times, which are measurements."""
        self.processor.calls.clear()
        if engines == 1:
            app_config = Settings.load({**self.overrides, "num_gpus": 1}, config_path=self.config_path)
            if stage == "classify":
                found = classify_pages(self.processor, pages, app_config.for_stage("classify"))
            else:
                found = extract_pages(self.processor, self.classified(pages), app_config)
        else:
            app_config = Settings.load(dict(self.overrides), config_path=self.config_path)
            if stage == "classify":
                found = vllm_dp.run_dp(
                    num_gpus=engines,
                    items=pages,
                    read=classify_pages,
                    settings=app_config.for_stage("classify"),
                )
            else:
                found = vllm_dp.run_dp(
                    num_gpus=engines, items=self.classified(pages), read=extract_pages, settings=app_config
                )
        return [{k: v for k, v in record.items() if k not in MEASUREMENTS} for record in found]

    def calls(self, stage: str, pages: list[Path], *, engines: int) -> list[tuple[str, list[str]]]:
        self.records(stage, pages, engines=engines)
        return list(self.processor.calls)


@pytest.fixture
def bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Bench:
    return Bench(tmp_path, monkeypatch)


STAGES = ("classify", "extract")


def test_the_module_that_starts_engines_knows_nothing_about_documents_or_stages() -> None:
    source = Path(vllm_dp.__file__).read_text()
    imported_from = {
        node.module or "" for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom)
    }

    # It is handed a stage's reading function; it does not import one.
    assert not any(module.split(".")[0] == "stages" for module in imported_from)
    for word in ("bank", "receipt", "invoice", "batch_size", "prompt"):
        assert word not in source.lower(), word


def test_an_engine_on_a_card_is_started_with_the_settings_the_stage_hands_over(bench: Bench) -> None:
    """The stage's own engine settings, and one card: nothing is read again for a worker."""
    pages = bench.pages(2)
    staged = Settings.load(dict(bench.overrides), config_path=bench.config_path).for_stage("classify")

    vllm_dp.run_dp(num_gpus=2, items=pages, read=classify_pages, settings=staged)

    assert len(bench.opened_with) == 2
    assert all(opened._stage == "classify" and opened.num_gpus == 1 for opened in bench.opened_with)
    # The stage's own settings are not changed by making a card's copy.
    assert staged.num_gpus != 1 or staged is not bench.opened_with[0]


def test_what_crosses_to_a_workers_process_can_be_pickled(bench: Bench) -> None:
    """A worker's process is given the reading function by name and the rest by value.

    The tests start a card's worker in this process, which pickles nothing, so this is the only place a
    setting or a function that could not cross would be found before a run on the cards.
    """
    import pickle

    from stages.sroie import read_sroie

    settings = Settings.load(dict(bench.overrides), config_path=bench.config_path).for_stage("classify")
    crossed = pickle.loads(pickle.dumps(settings.on_one_card()))

    assert (crossed.model_type, crossed.num_gpus, crossed._stage) == (settings.model_type, 1, "classify")
    assert crossed.get_vllm_config(crossed.model_type) == settings.get_vllm_config(settings.model_type)
    assert crossed.raw == settings.raw and crossed.model_path == settings.model_path
    for read in (classify_pages, extract_pages, read_sroie):
        assert pickle.loads(pickle.dumps(read)) is read
    pages = bench.pages(1)
    assert pickle.loads(pickle.dumps((pages, bench.classified(pages)))) == (pages, bench.classified(pages))


def test_one_engine_writes_as_it_reads_and_engines_on_cards_write_when_they_finish(
    bench: Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Sink:
        def __init__(self) -> None:
            self.records: list[dict[str, Any]] = []

        def write(self, record: dict[str, Any]) -> None:
            self.records.append(record)

    pages = bench.pages(2)
    settings = Settings.load(dict(bench.overrides), config_path=bench.config_path)
    names = [page.name for page in pages]

    for engines in (None, 2):
        monkeypatch.setattr(vllm_dp, "resolve_dp_gpus", lambda config, model_type, n=engines: n)
        sink = Sink()

        read = vllm_dp.read_on_the_cards(settings, bench.classified(pages), extract_pages, sink=sink)

        assert read.engines == engines
        assert read.arrangement == ("data-parallel (2 GPUs)" if engines else "single-engine")
        assert [record["image_name"] for record in read.records] == names
        assert [record["image_name"] for record in sink.records] == names


@pytest.mark.parametrize("stage", STAGES)
def test_both_arrangements_give_the_same_records(bench: Bench, stage: str) -> None:
    pages = bench.pages(5)

    one = bench.records(stage, pages, engines=1)

    assert bench.records(stage, pages, engines=2) == one
    assert [record["image_name"] for record in one] == [page.name for page in pages]


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("engines", [1, 2])
def test_no_pages_is_not_an_error(bench: Bench, stage: str, engines: int) -> None:
    assert bench.records(stage, [], engines=engines) == []


@pytest.mark.parametrize("stage", STAGES)
def test_fewer_pages_than_engines_keeps_every_page(bench: Bench, stage: str) -> None:
    pages = bench.pages(1, kinds=("receipt",))

    assert [r["image_name"] for r in bench.records(stage, pages, engines=2)] == [pages[0].name]


@pytest.mark.parametrize("stage", STAGES)
def test_each_engine_groups_its_pages_as_one_engine_would(bench: Bench, stage: str) -> None:
    """An engine on each card takes every second page, and reads its share exactly as one engine would."""
    pages = bench.pages(7)

    alone = [*bench.calls(stage, pages[0::2], engines=1), *bench.calls(stage, pages[1::2], engines=1)]

    assert bench.calls(stage, pages, engines=2) == alone
    assert len(alone) >= 2


def test_every_record_names_the_time_it_took_and_a_worker_names_its_card(bench: Bench) -> None:
    pages = bench.pages(2)
    app_config = Settings.load({**bench.overrides, "num_gpus": 1}, config_path=bench.config_path)

    one = classify_pages(bench.processor, pages, app_config.for_stage("classify"))
    two = vllm_dp.run_dp(
        num_gpus=2, items=pages, read=classify_pages, settings=app_config.for_stage("classify")
    )

    assert all(record["processing_time"] >= 0.0 and "gpu_id" not in record for record in one)
    assert [record["gpu_id"] for record in two] == [0, 1, 0, 1, 0, 1]


def test_a_page_that_cannot_be_read_is_recorded_and_the_rest_are_read(bench: Bench) -> None:
    """In a shared call that fails, each page is tried alone; one that still fails must not end the run."""
    pages = bench.pages(3, kinds=("invoice",))

    def read(image_paths, document_type):
        if len(image_paths) > 1:
            raise RuntimeError("engine refused the batch")
        if "CASE001" in image_paths[0]:
            raise RuntimeError("page too large")
        return ["TOTAL_AMOUNT: $12.00"]

    bench.processor.read_pages = read  # type: ignore[method-assign]

    records = bench.records("extract", pages, engines=1)

    assert [record["error"] for record in records] == [None, "page too large", None]
    assert [record["prompt_used"] for record in records] == ["invoice", "error", "invoice"]
