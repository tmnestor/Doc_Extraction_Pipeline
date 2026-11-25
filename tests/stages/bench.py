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
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference"))

from model_inputs import InProcessContext, run_settings, split_checked, test_config  # noqa: E402

from common import photo_split, vllm_dp  # noqa: E402
from stages import classify, clean, evaluate, extract, split  # noqa: E402

test_config.__test__ = False  # type: ignore[attr-defined]

ONE_ENGINE = "gemma4-31b-w4a16-vllm"
AN_ENGINE_ON_EACH_CARD = "gemma4-12b-unified-w4a16-vllm"
KINDS = ("bank_statement", "invoice", "receipt")


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
        self._called([f"{len(images)} photographs"])
        return ["[]"] * len(images)

    @property
    def asked_about(self) -> list[str]:
        return [name for call in self.calls for name in call]


class Bench:
    """A folder of blank pages, the settings to read them two to an engine call, and a stand-in processor."""

    def __init__(self, folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = folder
        shipped = yaml.safe_load(test_config(folder).read_text())
        shipped["pipeline"]["batching"] = {"classify": 2, "default": 2}
        monkeypatch.setattr(
            photo_split,
            "CONFIG",
            dataclasses.replace(
                photo_split.CONFIG,
                batch_size=2,
                straighten=False,
                rectify_pages=False,
                rectify_pieces=False,
            ),
        )
        self.config = folder / "two_to_a_call.yml"
        self.config.write_text(yaml.safe_dump(shipped, sort_keys=False))
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

    def with_settings(self, change: Any) -> None:
        """Write the settings file again, with ``change`` applied to its contents."""
        settings = yaml.safe_load(self.config.read_text())
        change(settings)
        self.config.write_text(yaml.safe_dump(settings, sort_keys=False))

    def split(self, output: Path, model: str = ONE_ENGINE) -> Path:
        """Run the split stage on the bench's pages. Gives the folder the later stages read images from."""
        return split.run(output, split_checked(model, self.config, self.pages))

    def classify(self, output: Path, model: str = ONE_ENGINE, pages: Path | None = None) -> Path:
        return classify.run(output, classify.checked(pages or self.pages, run_settings(model, self.config)))

    def extract(self, classifications: Path, output: Path, model: str = ONE_ENGINE) -> Path:
        return extract.run(output, extract.checked(classifications, run_settings(model, self.config)))

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
