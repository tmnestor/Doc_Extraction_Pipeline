"""DocumentOrchestrator — unified document extraction pipeline.

Replaces BaseDocumentProcessor + SimpleDocumentProcessor with a single
class that uses composition (has-a ModelBackend) instead of inheritance.

Satisfies the DocumentProcessor protocol from models/protocol.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from PIL import Image

from common import classification
from common.field_schema import get_field_schema
from common.prompt_catalog import PromptCatalog
from models.backend import BatchInference, GenerationParams, ModelBackend

if TYPE_CHECKING:
    from common.settings import Settings


class DocumentOrchestrator:
    """Unified document extraction orchestrator.

    Owns all shared logic: detection, classification, prompt resolution,
    and extraction. The replies are parsed later, by the clean stage.

    The backend (ModelBackend) provides only raw generate() / generate_batch().

    Attributes:
        model: Underlying model object (delegates to backend).
        tokenizer: Tokenizer / processor (delegates to backend).
        batch_size: Current batch size for inference.
    """

    def __init__(
        self,
        backend: ModelBackend,
        *,
        field_list: list[str],
        prompt_config: dict[str, Any],
        field_definitions: dict[str, list[str]] | None = None,
        debug: bool = False,
        verbose: bool = False,
        device: str = "cuda",
        model_type_key: str,
        app_config: Settings,
    ) -> None:
        self._backend = backend
        # debug → Tier C (dev-noise: PARSING DEBUG, prompt/response dumps, tracebacks).
        # verbose → Tier B (init/config details: auto-batch, gen-config, processing X).
        self.debug = debug
        self._verbose = verbose
        self.device = device
        self._model_type_key = model_type_key

        # Validate app_config
        self.app_config: Settings = app_config

        # Field configuration
        self.field_list = field_list
        self.field_count = len(field_list)

        self.prompt_config: dict[str, Any] = prompt_config
        missing = {"detection_file", "detection_key"} - set(self.prompt_config)
        if missing:
            raise ValueError(f"prompt_config missing required keys: {missing}")

        # The type of a page whose classification reply cannot be read.
        self._fallback_type = self.app_config.classification_fallback_type

        schema = get_field_schema()

        # Document-specific field lists
        self.document_field_lists: dict[str, list[str]] = (
            field_definitions if field_definitions is not None else schema.get_all_doc_type_fields()
        )

        # Generation config
        self._configure_generation()

        if self._verbose:
            print(
                f"DocumentOrchestrator initialized: {self.field_count} fields, model_type={model_type_key}"
            )

    # -- Protocol-required attributes ------------------------------------------

    @property
    def model(self) -> Any:
        """Underlying model object (for DocumentProcessor protocol)."""
        return self._backend.model

    @property
    def tokenizer(self) -> Any:
        """Tokenizer / processor (for DocumentProcessor protocol)."""
        return self._backend.processor

    # -- Generation config -----------------------------------------------------

    def _configure_generation(self) -> None:
        """Load generation hyper-parameters from Settings."""
        self.gen_config: dict[str, Any] = self.app_config.get_generation_config(self._model_type_key)

        self.fallback_max_tokens = max(
            int(
                self.gen_config.get(
                    "max_new_tokens_base",
                    self.app_config.get_token_budget("fallback_base"),
                )
            ),
            self.field_count
            * int(
                self.gen_config.get(
                    "max_new_tokens_per_field",
                    self.app_config.get_token_budget("fallback_per_field"),
                )
            ),
        )

        if self._verbose:
            print(
                f"Generation config: max_new_tokens={self.fallback_max_tokens}, "
                f"do_sample={self.gen_config.get('do_sample', False)}"
            )

    def _calculate_max_tokens(self, field_count: int) -> int:
        """The completion budget of a page with this many fields."""
        base = int(
            self.gen_config.get(
                "max_new_tokens_base",
                self.app_config.get_token_budget("fallback_base"),
            )
        )
        per_field = int(
            self.gen_config.get(
                "max_new_tokens_per_field",
                self.app_config.get_token_budget("fallback_per_field"),
            )
        )
        return base + (field_count * per_field)

    # -- Image loading ---------------------------------------------------------

    def load_document_image(self, image_path: str) -> Image.Image:
        """Load document image with error handling."""
        try:
            return Image.open(image_path)
        except Exception as e:
            if self.debug:
                print(f"Error loading image {image_path}: {e}")
            raise

    # -- Prompt loading --------------------------------------------------------

    # -- Document type parsing -------------------------------------------------

    # -- Bank structure classification -----------------------------------------

    # -- Engine calls ----------------------------------------------------------

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_tokens: int = 1024,
        extra: dict | None = None,
    ) -> str:
        """One request to the engine.

        Args:
            extra: Optional dict passed through to GenerationParams.extra
        """
        return self._backend.generate(
            image, prompt, GenerationParams(max_tokens=max_tokens, extra=extra or {})
        )

    @property
    def supports_generate_many(self) -> bool:
        """Whether the backend can run several generate() calls as one engine call."""
        return callable(getattr(self._backend, "generate_many", None))

    def generate_many(
        self,
        images: list[Image.Image],
        prompts: list[str],
        max_tokens: int = 1024,
        extra: dict | None = None,
    ) -> list[str]:
        """Run several generate() calls as one engine call, replies in input order.

        The batched counterpart of generate(): the backend builds each request
        as it would for a single call, and submits them together so the engine
        can run them concurrently.

        Args:
            images: One PIL image per request.
            prompts: One prompt per image, same order and length.
            max_tokens: Completion budget of every request.
            extra: Optional dict passed through to GenerationParams.extra.

        Raises:
            TypeError: If the backend has no generate_many.
        """
        if not self.supports_generate_many:
            raise TypeError(
                f"What: backend {type(self._backend).__name__} does not implement "
                f"generate_many, so independent calls cannot share an engine call.\n"
                f"Where: models/orchestrator.py.\n"
                f"Expected: a backend with generate_many(images, prompts, params), "
                f"as models/backends/vllm_backend.py has.\n"
                f"How to fix: implement generate_many on that backend, or set "
                f"pipeline.batching.bank_statement to 1 so each statement is its "
                f"own call."
            )
        params = GenerationParams(max_tokens=max_tokens, extra=extra or {})
        return self._backend.generate_many(images, prompts, params)  # type: ignore[attr-defined]

    def generate_batch(
        self,
        images: list[Image.Image],
        prompts: list[str],
        max_tokens: int = 1024,
        extra: dict | None = None,
    ) -> list[str]:
        """Run one batched inference over several images.

        A failure is the caller's to handle: the SROIE runner retries a failed
        batch one image at a time, so only the image that genuinely fails is lost.

        Args:
            images: One PIL image per request.
            prompts: One prompt per image, same order and length.
            max_tokens: Generation budget, shared by every request.
            extra: Passed through to GenerationParams.extra.

        Returns:
            Raw response strings, in submission order.
        """
        if not isinstance(self._backend, BatchInference):
            raise TypeError(
                f"What: backend {type(self._backend).__name__} does not implement "
                f"generate_batch, so it cannot run batched inference.\n"
                f"Where: models/orchestrator.py generate_batch().\n"
                f"Expected: a backend satisfying the BatchInference protocol "
                f"(models/backend.py).\n"
                f"How to fix: implement generate_batch on that backend."
            )

        params = GenerationParams(max_tokens=max_tokens, extra=extra or {})
        return self._backend.generate_batch(images, prompts, params)

    def cache_hit_summary(self) -> dict:
        """Proxy the backend's cumulative prefix-cache hit summary."""
        return self._backend.cache_hit_summary()

    # -- Detection pipeline ----------------------------------------------------

    def _detection_inputs(self) -> tuple[str, str, int]:
        """The classification prompt, its key and its completion budget."""
        detection_key = self.prompt_config["detection_key"]
        detection = yaml.safe_load(Path(self.prompt_config["detection_file"]).read_text())
        prompt = detection["prompts"][detection_key]["prompt"]
        return prompt, detection_key, self.app_config.get_token_budget("classify")

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
        responses = self.generate_many(images, [detection_prompt] * len(images), max_tokens)

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

        Two things differ by document type, and they are decided here and nowhere else:

        - A bank statement has its own prompt and a fixed completion budget, and is sent in the model's
          own request form.
        - Every other type has the prompt of its name in the model's prompt file and a budget worked out
          from the number of its fields, and is sent with the text before the image.

        Args:
            image_paths: The pages.
            document_type: Their document type, in either case.

        Returns:
            The replies as the model wrote them, in the order of the pages.
        """
        if not image_paths:
            return []
        kind = document_type.lower()
        if kind == "bank_statement":
            statements = [Image.open(path).convert("RGB") for path in image_paths]
            prompt = PromptCatalog().get_prompt("bank", "schema_fallback_extraction")
            budget = self.app_config.get_token_budget("extract_bank")
            return self.generate_many(statements, [prompt] * len(statements), budget)

        pages = [self.load_document_image(path) for path in image_paths]
        fields = self.document_field_lists.get(
            kind, self.document_field_lists.get("invoice", self.field_list)
        )
        prompt = PromptCatalog().get_prompt(self._model_type_key, kind)
        return self.generate_batch(pages, [prompt] * len(pages), self._calculate_max_tokens(len(fields)))

    # -- Model info ------------------------------------------------------------
