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

from common.field_schema import get_field_schema
from common.pipeline_config import strip_structure_suffixes
from common.prompt_catalog import PromptCatalog
from models.backend import BatchInference, GenerationParams, ModelBackend

if TYPE_CHECKING:
    from common.app_config import AppConfig


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
        batch_size: int | None = None,
        model_type_key: str,
        app_config: AppConfig,
    ) -> None:
        self._backend = backend
        # debug → Tier C (dev-noise: PARSING DEBUG, prompt/response dumps, tracebacks).
        # verbose → Tier B (init/config details: auto-batch, gen-config, processing X).
        self.debug = debug
        self._verbose = verbose
        self.device = device
        self._model_type_key = model_type_key

        # Validate app_config
        self.app_config: AppConfig = app_config

        # Field configuration
        self.field_list = field_list
        self.field_count = len(field_list)

        # Validate prompt_config
        if not prompt_config:
            raise ValueError(
                "prompt_config is required -- must contain "
                "'detection_file', 'detection_key', 'extraction_files'"
            )
        self.prompt_config: dict[str, Any] = prompt_config
        missing = {"detection_file", "detection_key", "extraction_files"} - set(self.prompt_config)
        if missing:
            raise ValueError(f"prompt_config missing required keys: {missing}")

        # Read fallback_type from centralized config (run_config.yml)
        self._fallback_type = self.app_config.classification_fallback_type

        # Resolve prompt YAML filename from extraction_files
        extraction_files = self.prompt_config["extraction_files"]
        if not extraction_files:
            raise ValueError(
                "prompt_config['extraction_files'] is empty -- "
                "must map document types to extraction YAML paths"
            )
        self._prompt_yaml = Path(next(iter(extraction_files.values()))).name

        schema = get_field_schema()

        # Document-specific field lists
        self.document_field_lists: dict[str, list[str]] = (
            field_definitions if field_definitions is not None else schema.get_all_doc_type_fields()
        )

        # Batch processing
        self._configure_batch_processing(model_type_key, batch_size)

        # Generation config
        self._configure_generation()

        if self._verbose:
            print(
                f"DocumentOrchestrator initialized: "
                f"{self.field_count} fields, batch_size={self.batch_size}, "
                f"model_type={model_type_key}"
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

    @property
    def supports_batch(self) -> bool:
        """Whether the backend supports batched inference."""
        return isinstance(self._backend, BatchInference)

    # -- Batch processing config -----------------------------------------------

    def _configure_batch_processing(self, model_type_key: str, batch_size: int | None) -> None:
        """Configure batch processing parameters."""
        if batch_size is not None:
            self.batch_size = max(1, batch_size)
            if self._verbose:
                print(f"Using manual batch size: {self.batch_size}")
        else:
            from common.gpu_memory import get_available_memory

            available_memory = get_available_memory(self.device)
            self.batch_size = self.app_config.get_auto_batch_size(model_type_key, available_memory)
            if self._verbose:
                print(f"Auto-detected batch size: {self.batch_size} (GPU Memory: {available_memory:.1f}GB)")

    # -- Generation config -----------------------------------------------------

    def _configure_generation(self) -> None:
        """Load generation hyper-parameters from AppConfig."""
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

    def _calculate_max_tokens(self, field_count: int, document_type: str) -> int:
        """Calculate token budget based on field count and document type."""
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
        tokens = base + (field_count * per_field)

        if document_type == "bank_statement":
            tokens = max(tokens, self.app_config.get_token_budget("bank_statement_floor"))
        return tokens

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

    def _parse_document_type_response(self, response: str, detection_config: dict) -> str:
        """Parse document type from model response using YAML-driven type mappings."""
        response_lower = response.lower().strip()

        if self.debug:
            sys.stdout.write(f"PARSING DEBUG - Raw response: '{response}'\n")
            sys.stdout.write(f"PARSING DEBUG - Cleaned response: '{response_lower}'\n")
            sys.stdout.flush()

        # Direct mapping check (YAML type_mappings)
        type_mappings = detection_config.get("type_mappings", {})
        for variant, canonical in type_mappings.items():
            if variant.lower() in response_lower:
                if self.debug:
                    sys.stdout.write(f"PARSING DEBUG - Found mapping: '{variant}' -> '{canonical}'\n")
                    sys.stdout.flush()
                return canonical

        # Fallback keyword detection
        fallback_keywords = detection_config.get("fallback_keywords", {})
        for canonical_type, keywords in fallback_keywords.items():
            if any(kw in response_lower for kw in keywords):
                if self.debug:
                    sys.stdout.write(f"PARSING DEBUG - Keyword match: {canonical_type}\n")
                    sys.stdout.flush()
                return canonical_type

        # Final fallback
        fallback = self._fallback_type
        if self.debug:
            sys.stdout.write(f"PARSING DEBUG - No matches found, using fallback: '{fallback}'\n")
            sys.stdout.flush()
        return fallback

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
                f"How to fix: call generate() per image instead, or implement "
                f"generate_batch on that backend. Check supports_batch before "
                f"calling this."
            )

        params = GenerationParams(max_tokens=max_tokens, extra=extra or {})
        return self._backend.generate_batch(images, prompts, params)

    def cache_hit_summary(self) -> dict:
        """Proxy the backend's cumulative prefix-cache hit summary."""
        return self._backend.cache_hit_summary()

    # -- Detection pipeline ----------------------------------------------------

    def _detection_inputs(self) -> tuple[str, dict, str, int]:
        """The detection prompt, its YAML config, its key and the classify token budget."""
        detection_path = Path(self.prompt_config["detection_file"])
        detection_key = self.prompt_config["detection_key"]

        with detection_path.open("r") as f:
            detection_config = yaml.safe_load(f)

        detection_prompt = detection_config["prompts"][detection_key]["prompt"]
        max_tokens = self.app_config.get_token_budget("classify")
        return detection_prompt, detection_config, detection_key, max_tokens

    def _detection_result(self, response: str, detection_config: dict, detection_key: str) -> dict:
        """Turn one detection reply into a classification result."""
        # Try enriched parsing (COLUMNS/PAID/ROWS) first, then legacy keywords
        from common.classification_parser import ClassificationParser

        parser = ClassificationParser(fallback_type=self._fallback_type)
        enriched = parser._parse_enriched(response)
        if enriched is not None:
            document_type = enriched["DOCUMENT_TYPE"]
        else:
            document_type = self._parse_document_type_response(response, detection_config)

        result = {
            "document_type": document_type,
            "confidence": 1.0,
            "raw_response": response,
            "prompt_used": detection_key,
        }
        if enriched and "column_mapping" in enriched:
            result["column_mapping"] = enriched["column_mapping"]
        return result

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

        detection_prompt, detection_config, detection_key, max_tokens = self._detection_inputs()
        images = [self.load_document_image(p) for p in image_paths]
        responses = self.generate_many(images, [detection_prompt] * len(images), max_tokens)

        results = [self._detection_result(r, detection_config, detection_key) for r in responses]
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

    # -- Extraction pipeline ---------------------------------------------------

    # -- Single image processing -----------------------------------------------

    # -- Several pages in one engine call --

    def extract_batch(
        self,
        image_paths: list[str],
        classification_infos: list[dict],
        verbose: bool = False,
    ) -> list[dict]:
        """Batch document extraction via backend.generate_batch().

        Called by the extract stage for a group of pages of one document type, and for a page
        on its own as a group of one, so that a page is asked the same way however it is grouped.
        """
        if not image_paths:
            return []

        # Build per-image prompts and field lists
        images = []
        prompts = []
        max_tokens_needed = 0

        for image_path, classification_info in zip(image_paths, classification_infos, strict=False):
            images.append(self.load_document_image(image_path))
            document_type = classification_info["document_type"].lower()

            # Get document-specific field list
            doc_type_fields = dict(self.document_field_lists)
            if "bank_statement" in doc_type_fields:
                doc_type_fields["bank_statement_flat"] = doc_type_fields["bank_statement"]
                doc_type_fields["bank_statement_date_grouped"] = doc_type_fields["bank_statement"]

            doc_field_list = doc_type_fields.get(
                document_type, doc_type_fields.get("invoice", self.field_list)
            )

            # Resolve extraction prompt
            extraction_prompt = self._resolve_extraction_prompt(document_type)
            prompts.append(extraction_prompt)

            # Track max tokens needed
            base_doc_type = strip_structure_suffixes(document_type)
            tokens = self._calculate_max_tokens(len(doc_field_list), base_doc_type)
            max_tokens_needed = max(max_tokens_needed, tokens)

        params = GenerationParams(max_tokens=max_tokens_needed, extra={})

        backend = self._backend
        if not isinstance(backend, BatchInference):
            # A bare assert here failed with an EMPTY message, so a caller's
            # fallback logged "Batch of 4 RECEIPT failed ()" and every batch
            # silently degraded to per-image with no clue why.
            raise TypeError(
                f"What: backend {type(backend).__name__} does not implement "
                f"generate_batch, so batched inference cannot run.\n"
                f"Where: models/orchestrator.py.\n"
                f"Expected: a backend satisfying the BatchInference protocol "
                f"(models/backend.py).\n"
                f"How to fix: implement generate_batch on that backend, or set "
                f"every pipeline.batching.* size to 1 so the stage stays on the "
                f"per-image path."
            )
        responses = backend.generate_batch(images, prompts, params)

        # The reply is all the extract stage keeps. The clean stage parses it.
        return [
            {"image_name": Path(path).name, "raw_response": response}
            for path, response in zip(image_paths, responses, strict=True)
        ]

    def _resolve_extraction_prompt(self, document_type: str) -> str:
        """Resolve extraction prompt for a document type."""
        doc_type_upper = strip_structure_suffixes(document_type).upper()
        extraction_files = self.prompt_config["extraction_files"]
        if doc_type_upper not in extraction_files:
            raise ValueError(
                f"No extraction file configured for '{doc_type_upper}'. "
                f"Available: {list(extraction_files.keys())}. "
                f"Add it to prompt_config['extraction_files']."
            )
        extraction_file = extraction_files[doc_type_upper]
        extraction_keys = self.prompt_config.get("extraction_keys", {})
        extraction_key = extraction_keys.get(doc_type_upper, document_type)

        catalog = PromptCatalog()
        return catalog.get_prompt(self._model_type_key, extraction_key)

    # -- Model info ------------------------------------------------------------
