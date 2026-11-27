"""Record everything the stages hand to the model, without a model.

The stages are run for real, from their own `run` functions, on a folder of small invented pages. In place
of vLLM stands a module whose engine keeps what it is given: the arguments it is built with, and for each
call the requests, the sampling settings and the chat arguments. It answers with fixed invented replies.

So the record is taken at the one place where the pipeline meets the model. However the code between a
stage and the engine is arranged, if the same things reach the engine in the same order, the record is the
same; and if a prompt, a setting, the grouping of pages into calls or the number of engines changes, the
record changes.

Nothing here reads an evaluation set. The pages are blank rectangles, and the replies are made up.
"""

import base64
import contextlib
import hashlib
import io
import os
import queue
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]

CARDS = 2
KINDS = ("bank_statement", "invoice", "receipt")
PAGES_OF_EACH_KIND = 55
"""The size of the clean evaluation set, so that the grouping into calls is the one real runs see."""

# A page is told apart by its size alone: the width says which page, the height says which kind.
BASE_WIDTH = 200
BASE_HEIGHT = 300

CLASSIFY_REPLIES = {
    "bank_statement": "COLUMNS: Date | Description | Debit | Credit | Balance\nPAID: NO\nROWS: 14\nTRAVEL: NO",
    "invoice": "COLUMNS: Description | Quantity | Unit Price | Amount\nPAID: NO\nROWS: 3\nTRAVEL: NO",
    "receipt": "COLUMNS: Item | Price\nPAID: YES\nROWS: 4\nTRAVEL: NO",
}
EXTRACT_REPLIES = {
    "bank_statement": (
        "DOCUMENT_TYPE: BANK_STATEMENT\n"
        "STATEMENT_DATE_RANGE: 01/02/2031 - 28/02/2031\n"
        "TRANSACTION_DATES: 04/02/2031 | 09/02/2031\n"
        "LINE_ITEM_DESCRIPTIONS: EFTPOS QUILLFEATHER BOOKS | DIRECT DEBIT MARROWBY GYM\n"
        "TRANSACTION_AMOUNTS_PAID: $41.07 | $63.00\n"
    ),
    "invoice": "DOCUMENT_TYPE: INVOICE\nSUPPLIER_NAME: Thistledown Joinery\nTOTAL_AMOUNT: $118.40\n",
    "receipt": "DOCUMENT_TYPE: RECEIPT\nSUPPLIER_NAME: Marrowby Grocer\nTOTAL_AMOUNT: $23.15\n",
}


SPLIT_REPLY = '[{"box_2d": [0, 0, 1000, 1000], "label": "receipt"}]'
SROIE_REPLY = (
    '{"company": "MARROWBY GROCER", "date": "04/02/2031", "address": "1 QUILL LANE", "total": "23.15"}'
)
SROIE_RECEIPTS = 12


def page_names() -> list[str]:
    """Invented page names, in the order a folder listing gives them."""
    return sorted(
        f"PAGE{number:03d}_{kind}.png" for number in range(1, PAGES_OF_EACH_KIND + 1) for kind in KINDS
    )


def write_pages(folder: Path) -> None:
    """Write one small blank page per name, its size saying which page and kind it is."""
    folder.mkdir(parents=True)
    for index, name in enumerate(page_names()):
        kind = next(k for k in KINDS if name.endswith(f"_{k}.png"))
        size = (BASE_WIDTH + index, BASE_HEIGHT + KINDS.index(kind))
        Image.new("RGB", size, "white").save(folder / name)


def write_receipts(folder: Path) -> None:
    """Write a small folder of blank receipts, each told apart by its width, for the dataset of them."""
    folder.mkdir(parents=True)
    for number in range(SROIE_RECEIPTS):
        Image.new("RGB", (100 + number, 150), "white").save(folder / f"R{number:03d}.jpg")


def page_of(image: Image.Image) -> tuple[str, str]:
    """The name and kind of the page an image holds, read back from its size."""
    width, height = image.size
    index, kind = width - BASE_WIDTH, height - BASE_HEIGHT
    names = page_names()
    if 0 <= index < len(names) and 0 <= kind < len(KINDS):
        return names[index], KINDS[kind]
    return f"image {width}x{height}", "receipt"


def image_of(part: dict[str, Any]) -> Image.Image:
    """The image a content part holds, whichever way the part gives it."""
    if part["type"] == "image_pil":
        return part["image_pil"]
    encoded = part["image_url"]["url"].split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


class Recorder:
    """What the stand-in engine was given, in the order it was given."""

    def __init__(self) -> None:
        self.stage = ""
        self.engines: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.prompts: dict[str, str] = {}

    def prompt_key(self, text: str) -> str:
        key = hashlib.sha256(text.encode()).hexdigest()[:12]
        self.prompts[key] = text
        return key

    def as_record(self) -> dict[str, Any]:
        return {"engines": self.engines, "calls": self.calls, "prompts": dict(sorted(self.prompts.items()))}


def stand_in_vllm(recorder: Recorder, model_folder: Path) -> types.ModuleType:
    """A module to import in place of vLLM, whose engine records and does no work."""

    class SamplingParams:
        def __init__(self, **settings: Any) -> None:
            self.settings = settings

    class LLM:
        def __init__(self, **arguments: Any) -> None:
            shown = {
                key: ("<model>" if value == str(model_folder) else value)
                for key, value in arguments.items()
            }
            self.number = len(recorder.engines)
            recorder.engines.append(
                {
                    "stage": recorder.stage,
                    "cards": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
                    "arguments": shown,
                }
            )

        def chat(self, messages: Any, sampling_params: Any = None, **others: Any) -> list[Any]:
            conversations = messages if messages and isinstance(messages[0], list) else [messages]
            requests, replies = [], []
            for conversation in conversations:
                parts, kind = [], "receipt"
                for message in conversation:
                    for part in message["content"]:
                        if part["type"] == "text":
                            parts.append([message["role"], "text", recorder.prompt_key(part["text"])])
                        else:
                            name, kind = page_of(image_of(part))
                            # How the image is given is part of what the engine is sent.
                            parts.append([message["role"], part["type"], f"{name} {image_of(part).mode}"])
                requests.append(parts)
                replies.append(self.reply(kind))
            recorder.calls.append(
                {
                    "stage": recorder.stage,
                    "engine": self.number,
                    "sampling": dict(sorted(sampling_params.settings.items())),
                    "others": dict(sorted(others.items())),
                    "requests": requests,
                }
            )
            return [
                types.SimpleNamespace(
                    outputs=[types.SimpleNamespace(text=reply, token_ids=[])],
                    prompt_token_ids=[],
                    num_cached_tokens=0,
                )
                for reply in replies
            ]

        def reply(self, kind: str) -> str:
            if recorder.stage == "preprocess":
                return SPLIT_REPLY
            if recorder.stage == "dataset_sroie":
                return SROIE_REPLY
            return (CLASSIFY_REPLIES if recorder.stage == "classify" else EXTRACT_REPLIES)[kind]

    module = types.ModuleType("vllm")
    module.LLM = LLM  # type: ignore[attr-defined]
    module.SamplingParams = SamplingParams  # type: ignore[attr-defined]
    return module


class InProcess:
    """Stands in for a worker process: runs its target at once, in this process."""

    def __init__(self, target: Any, args: tuple[Any, ...]) -> None:
        self.target, self.args, self.exitcode = target, args, 0

    def start(self) -> None:
        before = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            self.target(*self.args)
        finally:
            if before is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = before

    def is_alive(self) -> bool:
        return False

    def join(self, timeout: float | None = None) -> None:
        return None

    def terminate(self) -> None:
        return None


class InProcessContext:
    """Stands in for a multiprocessing context, so that an engine on each card is run one after another."""

    Queue = queue.Queue
    Process = InProcess


def run_settings(model: str, models: Path, **choices: Any) -> Any:
    """The run's settings for a model, with the choices a job's variables would give it.

    Args:
        model: A registered model.
        models: The folder that holds each model's folder of weights (``models_folder``).
        choices: Fields of ``StageSettings`` to set: ``document_types``, ``max_images``, ``preprocess_photographs``.
    """
    from common.stage_settings import StageSettings

    return StageSettings(
        task="",
        model=model,
        image_dir=None,
        output=models.parent / "output",
        ground_truth=None,
        clear_previous_output=False,
        models_dir=models,
        **choices,
    )


def preprocess_checked(
    model: str, models: Path, photographs: Path, *, preprocess_photographs: bool = True
) -> Any:
    """What the preprocess stage's check gives for a model and a folder of photographs.

    The switch is on, as a job sets it with ``preprocess_photographs=true``, unless the test says otherwise.
    """
    from stages import preprocess

    return preprocess.checked(
        run_settings(model, models, preprocess_photographs=preprocess_photographs), photographs
    )


def models_folder(folder: Path) -> Path:
    """A folder such as ``DOC_EXTRACT_MODELS_DIR`` names: a folder of weights for each registered model.

    Each holds an empty ``config.json`` and nothing else: no engine is started on it.
    """
    from models.registry import get_model, list_models

    models = folder / "models"
    for name in list_models():
        weights = models / get_model(name).weights_folder
        weights.mkdir(parents=True)
        (weights / "config.json").write_text("{}")
    return models


@contextlib.contextmanager
def recording(recorder: Recorder, model_folder: Path) -> Iterator[None]:
    """Put the stand-ins in place for vLLM and for worker processes, and take them away afterwards."""
    import common.vllm_dp as data_parallel
    from common import cards

    real_vllm = sys.modules.get("vllm")
    real_context = data_parallel.mp.get_context
    real_cards = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    # The machine has CARDS cards, as nvidia-smi would list them: there are none to count here.
    real_listed = cards.listed_by_nvidia_smi
    cards.listed_by_nvidia_smi = lambda: CARDS
    sys.modules["vllm"] = stand_in_vllm(recorder, model_folder)
    data_parallel.mp.get_context = lambda method=None: InProcessContext  # type: ignore[assignment]
    try:
        yield
    finally:
        data_parallel.mp.get_context = real_context  # type: ignore[assignment]
        cards.listed_by_nvidia_smi = real_listed
        if real_vllm is None:
            del sys.modules["vllm"]
        else:
            sys.modules["vllm"] = real_vllm
        if real_cards is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = real_cards


def record_model_inputs(model: str, folder: Path) -> dict[str, Any]:
    """Run every stage that uses the model, on invented pages, and return what the engine was given.

    Args:
        model: A registered model name.
        folder: An empty temporary folder to work in.

    Returns:
        The engines built, the calls made to them and the prompt texts those calls refer to.
    """
    from models.registry import get_model
    from stages import classify, extract, preprocess

    pages, output = folder / "pages", folder / "output"
    write_pages(pages)
    output.mkdir()
    write_receipts(folder / "receipts")
    models = models_folder(folder)
    recorder = Recorder()
    with recording(recorder, models / get_model(model).weights_folder):
        recorder.stage = "classify"
        settings = run_settings(model, models)
        classifications = classify.run(output / "classifications.jsonl", classify.checked(pages, settings))
        recorder.stage = "extract"
        extract.run(output / "raw_extractions.jsonl", extract.checked(classifications, settings))
        recorder.stage = "preprocess"
        preprocess.run(output / "preprocess", preprocess_checked(model, models, pages))
        # A set of pages that are all of one type (document_type=SROIE_RECEIPT): given their type by
        # classify, which asks no model, and read by extract by the prompt of that type.
        receipts = folder / "receipts"
        recorder.stage = "dataset_sroie_classify"
        given = classify.run(
            output / "dataset" / "classifications.jsonl",
            classify.checked(receipts, settings),
            given_type="SROIE_RECEIPT",
        )
        recorder.stage = "dataset_sroie"
        extract.run(output / "dataset" / "raw_extractions.jsonl", extract.checked(given, settings))
    return recorder.as_record()
