"""What every stage keeps to, from split to evaluate: the five stand alone, and the five behave alike.

A stage stands alone:

1. it is one module of ``stages`` with a ``BANNER``, a ``check(task)`` and a ``start(task)``;
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
from common.engine_calls import StageStopped
from stages import split
from stages.io import read_jsonl

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


@pytest.mark.parametrize("name", STAGES)
def test_a_stage_is_started_one_way(name: str) -> None:
    stage = importlib.import_module(f"stages.{name}")

    assert isinstance(stage.BANNER, str) and stage.BANNER
    assert list(inspect.signature(stage.start).parameters) == ["task"]
    assert list(inspect.signature(stage.check).parameters) == ["task"]


@pytest.mark.parametrize("name", STAGES)
def test_a_stage_imports_no_other_stage(name: str) -> None:
    tree = ast.parse((STAGES_FOLDER / f"{name}.py").read_text())

    assert modules_of_stages_imported_by(tree) <= SHARED


@pytest.mark.parametrize("name", sorted(SHARED))
def test_what_the_stages_share_imports_no_stage(name: str) -> None:
    tree = ast.parse((STAGES_FOLDER / f"{name}.py").read_text())

    assert modules_of_stages_imported_by(tree) <= SHARED - {name}


def writes_a_file_itself(call: ast.Call) -> bool:
    """Whether a call writes a file where it stands, and not by putting it in place whole."""
    function = call.func
    if isinstance(function, ast.Name):
        return function.id == "append_jsonl"
    if not isinstance(function, ast.Attribute):
        return False
    if function.attr == "open":
        # Image.open reads a picture.
        return not (isinstance(function.value, ast.Name) and function.value.id == "Image")
    return function.attr in ("write_text", "write_bytes", "append_jsonl")


def test_every_way_of_writing_a_file_where_it_stands_is_seen() -> None:
    """The check below is of no use if it cannot see such a call: each is shown to it here."""
    source = (
        "path.write_text(text)\npath.write_bytes(data)\npath.open('a')\nappend_jsonl(path, records)\n"
        "io.append_jsonl(path, records)\nImage.open(photograph)\nwrite_jsonl(path, records)\n"
        "in_the_order(path, order, adding=records)\n"
    )
    calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)]

    assert sorted(call.lineno for call in calls if writes_a_file_itself(call)) == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("name", STAGES)
def test_a_stage_writes_no_file_of_records_itself(name: str) -> None:
    """A file of records is written by ``stages/io.py``, which puts it in place whole.

    A stage that opened its own file, wrote one out as text, or added to one where it stood, would be
    the one whose output a reader could find half written.
    """
    tree = ast.parse((STAGES_FOLDER / f"{name}.py").read_text())
    own_writes = [
        f"line {node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and writes_a_file_itself(node)
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


def left_part_done(bench: Bench, model: str, stage: Callable[[], Any]) -> None:
    """Run a stage and stop it part way, as a run that did not finish leaves it.

    With one engine the run is killed at its second engine call. With an engine on each card the bench
    cannot stage that truly: a worker runs inside the test's own process, and a run killed there
    leaves before its parent has kept what the cards sent. So there the engine stops answering after
    its first call, which is a stop the bench does stage truly: the card's worker hands it to its
    parent through the queue, as on the cards.
    """
    if model == ONE_ENGINE:
        bench.processor.stop_at_call = len(bench.processor.calls) + 2
        with pytest.raises(Stopped):
            stage()
        bench.processor.stop_at_call = None
        return
    bench.processor.dead_from_attempt = len(bench.processor.attempts) + 2
    with pytest.raises(StageStopped):
        stage()
    bench.processor.dead_from_attempt = None


def through_all_five(bench: Bench, out: Path, model: str, *, stop_each_card_stage_once: bool) -> Path:
    """Run the five stages in order, each reading what the one before it wrote."""
    classifications = out / "classifications.jsonl"
    raw_extractions = out / "raw_extractions.jsonl"
    if stop_each_card_stage_once:
        # Two requests that fail one after another stop a stage: the bench has six pages.
        bench.with_settings(
            lambda settings: settings["pipeline"]["processing"].update(stop_after_failed_requests=2)
        )

    def split_it() -> Path:
        return bench.split(out / "split", model)

    def classify_it() -> Path:
        return bench.classify(classifications, model, pages=out / "split" / "images")

    def extract_it() -> Path:
        return bench.extract(classifications, raw_extractions, model)

    for stage in (split_it, classify_it, extract_it):
        if stop_each_card_stage_once:
            left_part_done(bench, model, stage)
            assert [p.name for p in out.rglob("*.journal")], "the stop left nothing to carry on from"
        stage()

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
