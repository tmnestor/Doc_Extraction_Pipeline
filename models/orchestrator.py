"""The document processor: what the stages ask the model, and with which prompt and budget.

It knows the classification prompt, the prompt and completion budget of each document type, and how a
classification reply becomes a document type. It makes every request through one function, ``ask``,
which hands the engine call to the backend. The replies of an extraction are kept as written and parsed
later, by the clean stage.
"""

from PIL import Image

from common import classification
from common.config_schema import ConfigError
from common.field_schema import get_field_schema
from common.prompt_catalog import PromptCatalog
from common.settings import Settings
from models.backends.vllm_backend import VllmBackend

# Completion budgets: the most tokens a reply may run to. The same for every model, so that a comparison
# of two models is not one of two budgets. A page's fields have the base and so many more for each field
# of its document type, unless the type's block in the schema file gives it a fixed budget.
CLASSIFY_TOKENS = 150
EXTRACT_BASE_TOKENS = 512
EXTRACT_TOKENS_PER_FIELD = 64

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
        """The completion budget of a page with this many fields."""
        return EXTRACT_BASE_TOKENS + field_count * EXTRACT_TOKENS_PER_FIELD

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

    # -- The document type of pages ---------------------------------------------

    def ask_document_types(self, image_paths: list[str]) -> list[str]:
        """The model's reply to the classification prompt for each page, all in one engine call.

        The same prompt and completion budget for every page. The replies come back as the model wrote
        them: ``classified`` turns one into a document type, and is not part of the call, so that a
        reply never has to be asked for twice. A failure is not caught: the caller decides whether to
        try the pages again one at a time.

        Args:
            image_paths: The pages.

        Returns:
            One reply for each page, in the order given.
        """
        if not image_paths:
            return []
        prompt = PromptCatalog().classification_prompt()
        images = [self.load_document_image(path) for path in image_paths]
        return self.ask(images, [prompt] * len(images), CLASSIFY_TOKENS)

    def classified(self, reply: str) -> dict:
        """The classification of a page, from the model's reply about it, and how its type was reached."""
        kind, how = classification.typed(reply, self._fallback_type)
        return {
            "document_type": kind,
            "type_from": how,
            "confidence": 1.0,
            "raw_response": reply,
            "prompt_used": DETECTION_PROMPT,
        }

    def unclassified(self, error: str | None, reply: str = "") -> dict:
        """The classification of a page the model could not be asked about: the fallback type, with why.

        The stage records it, so that one page does not end the run.

        Args:
            error: Why no type could be worked out for the page.
            reply: The model's reply, when it gave one that could not then be read.
        """
        return {
            "document_type": self._fallback_type,
            "type_from": classification.FALLBACK_NOT_ASKED,
            "confidence": 0.1,
            "raw_response": reply,
            "prompt_used": "fallback_heuristic",
            "error": error or "",
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
            ConfigError: If the schema file has no block for the document type.
        """
        if not image_paths:
            return []
        kind = document_type.lower()
        described = get_field_schema().document_type(kind)
        if described is None:
            known = ", ".join(get_field_schema().document_types)
            raise ConfigError(
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
