"""A folder of blank pages and a stand-in for the document processor, for tests of whole stages.

No engine is started: the stand-in answers from a page's file name, and can be told to stop the run or
to fail. The pages are blank images made here, and every reply is invented.
"""

import contextlib
import dataclasses
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference"))

from model_inputs import InProcessContext, models_folder, preprocess_checked, run_settings

from common import cards, engine_calls, extraction_batching, image_preprocessing, vllm_dp
from common.prompt_catalog import PromptCatalog
from stages import classify, clean, evaluate, extract, preprocess

ONE_ENGINE = "gemma4-31b-w4a16-vllm"
AN_ENGINE_ON_EACH_CARD = "gemma4-12b-unified-w4a16-vllm"
KINDS = ("bank_statement", "invoice", "receipt")


def asks_which_way_up(prompts: list[str]) -> bool:
    """Whether a call puts the preprocess stage's first question, of whether each photograph is the right way up.

    The stage puts two questions about a batch of photographs, in two calls. The stand-in answers the
    first YES and does not count it: a call it counts is a batch asked where its receipts are.
    """
    return bool(prompts) and prompts[0] == PromptCatalog().preprocess_prompt("upright")


class Stopped(BaseException):
    """Stands in for a run being killed: nothing in the pipeline catches it."""


class Processor:
    """Stands in for the document processor.

    It can be told to stop the run at one of its calls, to fail every call that holds a named page, and
    to fail every call from one on, as an engine that has stopped answering would.
    """

    def __init__(self) -> None:
        # The calls that were answered, and every call that was made, answered or not.
        self.calls: list[list[str]] = []
        self.attempts: list[list[str]] = []
        self.stop_at_call: int | None = None
        self.cannot_read: set[str] = set()
        self.dead_from_attempt: int | None = None

    def _called(self, names: list[str]) -> None:
        if self.stop_at_call is not None and len(self.calls) + 1 == self.stop_at_call:
            raise Stopped
        self.attempts.append(names)
        if self.dead_from_attempt is not None and len(self.attempts) >= self.dead_from_attempt:
            raise RuntimeError("the engine has stopped answering")
        for name in names:
            if name in self.cannot_read:
                raise RuntimeError(f"cannot read {name}")
        self.calls.append(names)

    def ask_document_types(self, image_paths: list[str]) -> list[str]:
        self._called([Path(p).name for p in image_paths])
        return [f"a reply about {Path(p).name}" for p in image_paths]

    def classified(self, reply: str) -> dict[str, Any]:
        return {"document_type": next(k for k in KINDS if k in reply).upper(), "raw_response": reply}

    def unclassified(self, error: str | None, reply: str = "") -> dict[str, Any]:
        return {"document_type": "UNIVERSAL", "raw_response": reply, "error": error or ""}

    def read_pages(self, image_paths: list[str], document_type: str) -> list[str]:
        self._called([Path(p).name for p in image_paths])
        return [f"TOTAL_AMOUNT: read from {Path(p).name}" for p in image_paths]

    def load_document_image(self, image_path: str) -> Image.Image:
        return Image.open(image_path)

    def ask(self, images: list[Any], prompts: list[str], max_tokens: int) -> list[str]:
        if asks_which_way_up(prompts):
            return ["YES"] * len(images)
        self._called([f"{len(images)} photographs"])
        return ["[]"] * len(images)

    @property
    def asked_about(self) -> list[str]:
        return [name for call in self.calls for name in call]


class Bench:
    """A folder of blank pages, the settings to read them two to an engine call, and a stand-in processor."""

    def __init__(self, folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = folder
        self.monkeypatch = monkeypatch
        # The folder a job would name in DOC_EXTRACT_MODELS_DIR.
        self.models = models_folder(folder)
        # Two pages to an engine call, and two cards, where the code has 64 and counts the machine's.
        self.pages_to_a_call(2)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setattr(cards, "listed_by_nvidia_smi", lambda: 2)
        monkeypatch.setattr(
            image_preprocessing,
            "CONFIG",
            dataclasses.replace(
                image_preprocessing.CONFIG,
                batch_size=2,
                straighten=False,
                rectify_pages=False,
                rectify_pieces=False,
            ),
        )
        self.pages = folder / "pages"
        self.pages.mkdir()
        self.names: list[str] = []
        for number in range(2):
            for kind in KINDS:
                name = f"CASE{number:03d}_{kind}.png"
                Image.new("RGB", (64, 64), "white").save(self.pages / name)
                self.names.append(name)
        self.names.sort(key=str.lower)
        self.processor = Processor()
        # How many times an engine was started.
        self.engines_started = 0
        monkeypatch.setattr(vllm_dp.mp, "get_context", lambda method=None: InProcessContext)
        monkeypatch.setattr("common.pipeline_ops.open_processor", self._open)

    @contextlib.contextmanager
    def _open(self, settings: Any) -> Iterator[Processor]:
        self.engines_started += 1
        yield self.processor

    def pages_to_a_call(self, pages: int) -> None:
        self.monkeypatch.setattr(extraction_batching, "PAGES_PER_CALL", pages)

    def stop_after(self, failures: int) -> None:
        """Have a stage stop when this many requests in a row have failed."""
        self.monkeypatch.setattr(engine_calls, "STOP_AFTER_FAILED_REQUESTS", failures)

    def preprocess(
        self, output: Path, model: str = ONE_ENGINE, *, preprocess_photographs: bool = True
    ) -> Path:
        """Run the preprocess stage on the bench's pages. Gives the folder the later stages read images from."""
        return preprocess.run(
            output,
            preprocess_checked(
                model, self.models, self.pages, preprocess_photographs=preprocess_photographs
            ),
        )

    def classify(
        self, output: Path, model: str = ONE_ENGINE, pages: Path | None = None, **choices: Any
    ) -> Path:
        """Run classify. ``choices`` are a job's: ``document_types``, ``max_images``."""
        settings = run_settings(model, self.models, **choices)
        return classify.run(output, classify.checked(pages or self.pages, settings))

    def extract(self, classifications: Path, output: Path, model: str = ONE_ENGINE) -> Path:
        return extract.run(output, extract.checked(classifications, run_settings(model, self.models)))

    def clean(self, raw_extractions: Path, output: Path) -> Path:
        return clean.run(clean.checked(raw_extractions), output)

    def evaluate(self, cleaned_extractions: Path, key: Path, output_dir: Path) -> Path:
        return evaluate.run(evaluate.checked(cleaned_extractions, key), output_dir)

    def answer_key(self, path: Path) -> Path:
        """Write an answer key for the bench's pages. Every value is invented."""
        path.parent.mkdir(parents=True, exist_ok=True)
        answers = [{"filename": name, "TOTAL_AMOUNT": "$31.07"} for name in self.names]
        path.write_text("".join(json.dumps(answer) + "\n" for answer in answers))
        return path
