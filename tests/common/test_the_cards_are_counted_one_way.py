"""The number of cards a run has is counted in one place, for both things it decides.

It decides how many engines are started when a model fits on one card, and how many cards one engine is
split across when it does not. Until 25 November 2026 each had a count of its own: the first asked torch,
the second asked nvidia-smi. On one machine the two could disagree, and a run would start a number of
engines that was not the number of cards an engine thought it had.

A stage's row of timing also said "2 GPUs" for a run that started one engine, for the one page it had
left to read: it gave the number of cards, not of engines.
"""

import subprocess
import types
from pathlib import Path
from typing import Any

import pytest

from common import cards, vllm_dp
from common.cards import cards_to_use
from common.vllm_dp import CardsRead, resolve_dp_gpus

REPO = Path(__file__).resolve().parents[2]
ON_EACH_CARD = "gemma4-12b-unified-w4a16-vllm"
ACROSS_THE_CARDS = "gemma4-31b-w4a16-vllm"


def nvidia_smi_lists(monkeypatch: pytest.MonkeyPatch, lines: int | None, returncode: int = 0) -> None:
    """Make nvidia-smi list this many cards, or be missing from the machine when ``lines`` is None."""
    monkeypatch.setattr(
        cards.shutil, "which", lambda name: None if lines is None else "/usr/bin/nvidia-smi"
    )

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        listed = "".join(f"GPU {n}: NVIDIA L4 (UUID: GPU-{n})\n" for n in range(lines or 0))
        return subprocess.CompletedProcess(command, returncode, stdout=listed, stderr="")

    monkeypatch.setattr(cards.subprocess, "run", run)


# --- the count ------------------------------------------------------------------------------------------


def test_the_count_takes_no_setting() -> None:
    """Until 26 November 2026 the settings file could give a number of cards, and a number of engines
    apart from it. No run ever set either: the cards are counted."""
    import inspect

    assert list(inspect.signature(cards_to_use).parameters) == []
    assert list(inspect.signature(resolve_dp_gpus).parameters) == ["model_type"]


@pytest.mark.parametrize(("visible", "count"), [("0", 1), ("0,1", 2), ("1, 3 ,5", 3), ("", 1)])
def test_the_cards_the_process_may_see_are_counted(
    monkeypatch: pytest.MonkeyPatch, visible: str, count: int
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    nvidia_smi_lists(monkeypatch, 8)

    assert cards_to_use() == count


@pytest.mark.parametrize("listed", [1, 2, 4])
def test_with_nothing_else_said_the_cards_nvidia_smi_lists_are_counted(
    monkeypatch: pytest.MonkeyPatch, listed: int
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    nvidia_smi_lists(monkeypatch, listed)

    assert cards_to_use() == listed


def test_a_machine_that_says_nothing_of_its_cards_is_taken_to_have_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    nvidia_smi_lists(monkeypatch, None)
    assert cards_to_use() == 1

    nvidia_smi_lists(monkeypatch, 2, returncode=9)
    assert cards_to_use() == 1


# --- one count for both uses ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("number_set", "visible", "listed"),
    [(0, "0,1", 8), (0, None, 2), (0, "0", 2), (0, None, 4), (0, None, None)],
    ids=["visible", "listed", "one visible of two", "four listed", "nothing said"],
)
def test_the_engines_started_and_the_cards_an_engine_is_split_across_are_one_number(
    monkeypatch: pytest.MonkeyPatch, number_set: int, visible: str | None, listed: int | None
) -> None:
    """Whatever says how many cards there are, both uses hear the same answer."""
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    nvidia_smi_lists(monkeypatch, listed)
    split_across = cards_to_use()

    # A model that fits on one card has an engine on each; one that does not has one across them all.
    assert resolve_dp_gpus(ON_EACH_CARD) == (split_across if split_across > 1 else None)
    assert resolve_dp_gpus(ACROSS_THE_CARDS) is None


def test_nothing_else_counts_the_cards() -> None:
    """A second count is how the two came to disagree: only common/cards.py may ask the machine."""
    asks = ("nvidia-smi", "device_count", 'environ.get("CUDA_VISIBLE_DEVICES")')
    found = {}
    for folder in ("common", "models", "stages"):
        for path in (REPO / folder).rglob("*.py"):
            text = path.read_text()
            hits = [ask for ask in asks if ask in text]
            if hits and path.name != "cards.py":
                found[str(path.relative_to(REPO))] = hits

    assert found == {}


# --- what a stage says of its engines -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("engines", "said"),
    [(None, "single-engine"), (1, "data-parallel (1 GPU)"), (2, "data-parallel (2 GPUs)")],
)
def test_the_arrangement_names_the_engines_that_were_started(engines: int | None, said: str) -> None:
    assert CardsRead([], engines).arrangement == said


@pytest.mark.parametrize(("pages", "started"), [(1, 1), (2, 2), (5, 2)])
def test_one_page_left_to_read_starts_one_engine_and_the_stage_says_so(
    monkeypatch: pytest.MonkeyPatch, pages: int, started: int
) -> None:
    """Two cards, and fewer pages than cards: the row of timing must not say two engines read them."""
    handed_out: list[int] = []

    def run_dp(*, num_gpus: int, items: list, read: Any, settings: Any, sink: Any = None) -> list[dict]:
        handed_out.append(min(num_gpus, len(items)))
        return [{"image_name": item} for item in items]

    monkeypatch.setattr(vllm_dp, "run_dp", run_dp)
    monkeypatch.setattr("common.pipeline_ops.check_processor", lambda settings: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    run: Any = types.SimpleNamespace(model=ON_EACH_CARD)

    read = vllm_dp.read_on_the_cards(run, [f"page_{n}.png" for n in range(pages)], lambda *a, **k: [])

    assert read.engines == started == handed_out[0]
    assert read.arrangement == f"data-parallel ({started} GPU{'' if started == 1 else 's'})"
