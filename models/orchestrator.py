"""The document processor: what the stages ask the model, and with which prompt and budget.

It knows the classification prompt, the prompt and completion budget of each document type, and how a
classification reply becomes a document type. It makes every request through one function, ``ask``,
which hands the engine call to the backend. The replies of an extraction are kept as written and parsed
later, by the clean stage.
"""

import sys
from pathlib import Path

from PIL import Image

from common import classification
from common.field_schema import get_field_schema
from common.prompt_catalog import PromptCatalog
from common.settings import Settings
from models.backends.vllm_backend import VllmBackend

# What a classification record says its page was read with.
DETECTION_PROMPT = "detection"


class DocumentOrchestrator:
    """Asks the model about pages, for the stages."""

    def __init__(self, backend: VllmBackend, *, model_type_key: str, app_config: Settings) -> None:
        self._backend = backend
        self._model_type_key = model_type_key
        self.app_config: Settings = app_config
        # The type of a page whose classification reply cannot be read.
        self._fallback_type = app_config.classification_fallback_type

    # -- Completion budgets ----------------------------------------------------

    def _calculate_max_tokens(self, field_count: int) -> int:
        """The completion budget of a page with this many fields (``pipeline.token_budgets``)."""
        base = self.app_config.get_token_budget("extract_base")
        return base + field_count * self.app_config.get_token_budget("extract_per_field")

    # -- Image loading ---------------------------------------------------------

    def load_document_image(self, image_path: str) -> Image.Image:
        """A page's image. A file that cannot be opened is the caller's to deal with."""
        return Image.open(image_path)

    # -- The engine ------------------------------------------------------------

    def ask(
        self,
        images: list[Image.Image],
        prompts: list[str],
        max_tokens: int,
    ) -> list[str]:
        """Put a prompt to the model for each image, all in one engine call; replies in the same order.

        Every request the pipeline makes goes through here. A failure is not caught: the caller decides
        whether to try the images again one at a time.

        Args:
            images: One image a request.
            prompts: The prompt of each request.
            max_tokens: The completion budget of every request.
        """
        return self._backend.chat(images, prompts, max_tokens)

    # -- Detection pipeline ----------------------------------------------------

    def _detection_inputs(self) -> tuple[str, str, int]:
        """The classification prompt, its key and its completion budget."""
        prompt = PromptCatalog().classification_prompt()
        return prompt, DETECTION_PROMPT, self.app_config.get_token_budget("classify")

    def _detection_result(self, response: str, detection_key: str) -> dict:
        """The classification record of one reply."""
        return {
            "document_type": classification.document_type(response, self._fallback_type),
            "confidence": 1.0,
            "raw_response": response,
            "prompt_used": detection_key,
        }

    def detect_many(self, image_paths: list[str], verbose: bool = False) -> list[dict]:
        """Detect the document type of several pages in one engine call.

        The same prompt and token budget for every page. A failure is not caught:
        the caller decides whether to retry one page at a time.

        Args:
            image_paths: Pages to classify.
            verbose: Print each page's detected type.

        Returns:
            One result per page, in input order.
        """
        if not image_paths:
            return []

        detection_prompt, detection_key, max_tokens = self._detection_inputs()
        images = [self.load_document_image(p) for p in image_paths]
        responses = self.ask(images, [detection_prompt] * len(images), max_tokens)

        results = [self._detection_result(r, detection_key) for r in responses]
        if verbose:
            for path, result in zip(image_paths, results, strict=True):
                sys.stdout.write(f"Detected document type: {Path(path).name}: {result['document_type']}\n")
            sys.stdout.flush()
        return results

    def detect_and_classify_document(self, image_path: str, verbose: bool = False) -> dict:
        """Detect the document type of one page: a call of one, with a failure recorded, not raised.

        A page that cannot be classified is given the fallback type and carries the error, so that
        one page does not end the run.
        """
        try:
            return self.detect_many([image_path], verbose=verbose)[0]
        except Exception as e:  # noqa: BLE001 - recorded on the page, not raised
            sys.stdout.write(f"DETECTION ERROR: {e}\n")
            sys.stdout.flush()
            return {
                "document_type": self._fallback_type,
                "confidence": 0.1,
                "raw_response": "",
                "prompt_used": "fallback_heuristic",
                "error": str(e),
            }

    # -- Reading the fields of pages --------------------------------------------

    def read_pages(self, image_paths: list[str], document_type: str) -> list[str]:
        """The model's reply for each page of one document type, all in one engine call.

        The one place a page's fields are asked for. The extract stage calls it for a group of pages of
        one type, and for a page on its own as a group of one, so that a page is asked the same way
        however it is grouped.

        Nothing here names a document type. A type's prompt is the one of its name in the model's
        prompt file. Its completion budget is the fixed one its block in the schema file gives, or
        else is worked out from the number of its fields.

        Args:
            image_paths: The pages.
            document_type: Their document type, in either case.

        Returns:
            The replies as the model wrote them, in the order of the pages.

        Raises:
            ValueError: If the schema file has no block for the document type.
        """
        if not image_paths:
            return []
        kind = document_type.lower()
        described = get_field_schema().document_type(kind)
        if described is None:
            known = ", ".join(get_field_schema().document_types)
            raise ValueError(
                f"What: the document type {document_type!r} has no block in the schema file.\n"
                f"Where: config/field_definitions.yaml, key document_types.{kind}.\n"
                f"Expected: one of {known}.\n"
                f"How to fix: add a block for {kind!r} with its fields, and a prompt of the same name in "
                f"the model's prompt file."
            )
        pages = [self.load_document_image(path) for path in image_paths]
        prompt = PromptCatalog().get_prompt(self._model_type_key, kind)
        budget = described.completion_tokens or self._calculate_max_tokens(len(described.fields))
        return self.ask(pages, [prompt] * len(pages), budget)
