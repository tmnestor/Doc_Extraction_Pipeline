"""What every stage keeps to, from split to evaluate: the five stand alone, and the five behave alike.

A stage stands alone:

1. it is one module of ``stages`` with a ``BANNER`` and a ``start(task)``;
2. it reads its settings and the file the stage before it wrote, writes its own file, and imports no
   other stage;
3. a record it hands on is made by ``stages/records.py``
   (``test_a_record_between_stages_is_defined_once.py`` holds that).

The five behave alike:

4. a stage's output holds one record for each item of its input, in the stage's own order;
5. that output is the same file whether the run went straight through or was stopped and started again;
6. an output file is put in place whole, through what the stages share, and by no stage itself;
7. a stage that uses the cards carries on by ``stages.carry_on``
   (``test_a_stage_carries_on_one_way.py`` holds how).

What the stages share is three modules of functions that a stage calls: ``io``, ``records`` and
``carry_on``. None of them imports a stage.

The pages are blank images made by the bench, and every reply and every answer is invented.
"""

import ast
import importlib
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from bench import AN_ENGINE_ON_EACH_CARD, ONE_ENGINE, Bench, Stopped
from common.extraction_sort import sort_for_extraction
from common.settings import Settings
from stages import classify, extract, split
from stages.io import Journal, read_jsonl

STAGES = ("split", "classify", "extract", "clean", "evaluate")
SHARED = {"io", "records", "carry_on"}
STAGES_FOLDER = Path(inspect.getfile(split)).parent
# The file each stage writes, under a run's output folder, in the order the stages run.
FILES = (
    "split/splits.jsonl",
    "classifications.jsonl",
    "raw_extractions.jsonl",
    "cleaned_extractions.jsonl",
    "evaluation/evaluation_results.jsonl",
)
BOTH = [ONE_ENGINE, AN_ENGINE_ON_EACH_CARD]


# --- a stage stands alone -------------------------------------------------------------------------------


def modules_of_stages_imported_by(tree: ast.AST) -> set[str]:
    """The modules of the ``stages`` package a module imports, by any of the ways one can be."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[1] for alias in node.names if alias.name.startswith("stages.")}
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level and module:  # from .io import x
                found.add(module.split(".")[0])
            elif node.level or module == "stages":  # from . import x; from stages import x
                found |= {alias.name for alias in node.names}
            elif module.startswith("stages."):  # from stages.io import x
                found.add(module.split(".")[1])
    return found


def test_every_way_of_importing_a_stage_is_seen() -> None:
    """The check below is of no use if it cannot see an import: each form is shown to it here."""
    source = (
        "import stages.a\nfrom stages import b\nfrom stages.c import x\n"
        "from . import d\nfrom .e import x\n"
        "def later():\n    from stages.f import x\n"
        "import common.g\nfrom common.h import x\n"
    )

    assert modules_of_stages_imported_by(ast.parse(source)) == {"a", "b", "c", "d", "e", "f"}


@pytest.mark.parametrize("name", [*STAGES, "sroie"])
def test_a_stage_is_started_one_way(name: str) -> None:
    stage = importlib.import_module(f"stages.{name}")

    assert isinstance(stage.BANNER, str) and stage.BANNER
    assert list(inspect.signature(stage.start).parameters) == ["task"]


@pytest.mark.parametrize("name", [*STAGES, "sroie"])
def test_a_stage_imports_no_other_stage(name: str) -> None:
    tree = ast.parse((STAGES_FOLDER / f"{name}.py").read_text())

    assert modules_of_stages_imported_by(tree) <= SHARED


@pytest.mark.parametrize("name", sorted(SHARED))
def test_what_the_stages_share_imports_no_stage(name: str) -> None:
    tree = ast.parse((STAGES_FOLDER / f"{name}.py").read_text())

    assert modules_of_stages_imported_by(tree) <= SHARED - {name}


@pytest.mark.parametrize("name", STAGES)
def test_a_stage_writes_no_file_of_records_itself(name: str) -> None:
    """A file of records is written by ``stages/io.py``, which puts it in place whole.

    A stage that opened its own file, or wrote one out as text, would be the one whose output a reader
    could find half written. ``Image.open`` reads a picture and is not that.
    """
    tree = ast.parse((STAGES_FOLDER / f"{name}.py").read_text())
    own_writes = [
        f"line {node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and (
            node.func.attr in ("write_text", "write_bytes")
            or (
                node.func.attr == "open"
                and not (isinstance(node.func.value, ast.Name) and node.func.value.id == "Image")
            )
        )
    ]

    assert own_writes == []


# --- the five behave alike ------------------------------------------------------------------------------


def names_in(path: Path) -> list[str]:
    return [record["image_name"] for record in read_jsonl(path)]


def comparable(path: Path, run_folder: Path) -> list[dict[str, Any]]:
    """A file's records, apart from what differs between two runs that read the same: the times, the
    card that read each page, and the run's own folder wherever a path names it.
    """
    lines = path.read_text().replace(str(run_folder), "<the run>").splitlines()
    return [
        {key: value for key, value in json.loads(line).items() if key not in ("processing_time", "gpu_id")}
        for line in lines
    ]


def left_part_done(
    bench: Bench,
    model: str,
    stage: Callable[[], Any],
    output: Path,
    card_0s_share: Callable[[Journal], Any],
) -> None:
    """Leave a stage as a run stopped part way leaves it, before it is run.

    With one engine the stage is run and stopped at its second engine call. With an engine on each
    card the bench cannot stage that: a worker runs inside the test's own process, and a run killed
    there leaves before its parent has written what the cards sent. What a stop between the cards
    leaves is put there instead: card 0's share in the stage's journal, made by the stage's own reading
    function.
    """
    if model == ONE_ENGINE:
        bench.processor.stop_at_call = len(bench.processor.calls) + 2
        with pytest.raises(Stopped):
            stage()
        bench.processor.stop_at_call = None
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with Journal(output) as journal:
        card_0s_share(journal)


def through_all_five(bench: Bench, out: Path, model: str, *, stop_each_card_stage_once: bool) -> Path:
    """Run the five stages in order, each reading what the one before it wrote."""
    settings = Settings.load({"model_type": model}, config_path=bench.config)
    photographs = [bench.pages / name for name in bench.names]
    classifications = out / "classifications.jsonl"
    raw_extractions = out / "raw_extractions.jsonl"

    def split_it() -> Path:
        return bench.split(out / "split", model)

    if stop_each_card_stage_once:
        left_part_done(
            bench,
            model,
            split_it,
            out / "split" / "splits.jsonl",
            lambda journal: split.ask_about_photographs(
                bench.processor, photographs[0::2], settings, sink=journal
            ),
        )
    images = split_it()

    def classify_it() -> Path:
        return bench.classify(classifications, model, pages=images)

    if stop_each_card_stage_once:
        pages = sorted(images.iterdir(), key=lambda page: page.name.lower())
        left_part_done(
            bench,
            model,
            classify_it,
            classifications,
            lambda journal: classify.classify_pages(
                bench.processor, pages[0::2], settings.for_stage("classify"), sink=journal
            ),
        )
    classify_it()

    def extract_it() -> Path:
        return bench.extract(classifications, raw_extractions, model)

    if stop_each_card_stage_once:
        in_order = sort_for_extraction(
            read_jsonl(classifications), settings.extraction_order, secondary_sort=settings.secondary_sort
        )
        left_part_done(
            bench,
            model,
            extract_it,
            raw_extractions,
            lambda journal: extract.extract_pages(bench.processor, in_order[0::2], settings, sink=journal),
        )
    extract_it()

    cleaned = bench.clean(raw_extractions, out / "cleaned_extractions.jsonl")
    bench.evaluate(cleaned, bench.answer_key(out / "ground_truth.jsonl"), out / "evaluation")
    return out


@pytest.mark.parametrize("model", BOTH)
def test_the_five_stages_give_the_same_files_however_often_the_run_was_stopped(
    bench: Bench, tmp_path: Path, model: str
) -> None:
    straight = through_all_five(bench, tmp_path / "straight", model, stop_each_card_stage_once=False)
    stopped = through_all_five(bench, tmp_path / "stopped", model, stop_each_card_stage_once=True)

    for file in FILES:
        assert comparable(stopped / file, stopped) == comparable(straight / file, straight), file
    assert sorted(p.name for p in (stopped / "split" / "images").iterdir()) == bench.names


@pytest.mark.parametrize("model", BOTH)
def test_every_stage_writes_one_record_for_each_item_it_was_given(
    bench: Bench, tmp_path: Path, model: str
) -> None:
    out = through_all_five(bench, tmp_path / "out", model, stop_each_card_stage_once=True)

    for file in FILES:
        names = names_in(out / file)
        assert sorted(names) == sorted(bench.names) and len(set(names)) == len(names), file


def test_a_stage_run_a_second_time_leaves_its_file_as_it_was(bench: Bench, tmp_path: Path) -> None:
    out = through_all_five(bench, tmp_path / "out", ONE_ENGINE, stop_each_card_stage_once=False)
    before = {file: (out / file).read_bytes() for file in FILES}
    started = bench.engines_started

    through_all_five(bench, out, ONE_ENGINE, stop_each_card_stage_once=False)

    assert {file: (out / file).read_bytes() for file in FILES} == before
    assert bench.engines_started == started


@pytest.mark.parametrize("model", BOTH)
def test_no_stage_leaves_a_journal_or_a_file_half_put_in_place(
    bench: Bench, tmp_path: Path, model: str
) -> None:
    out = through_all_five(bench, tmp_path / "out", model, stop_each_card_stage_once=True)

    assert not [p.name for p in out.rglob("*") if p.name.endswith((".journal", ".rewriting"))]
