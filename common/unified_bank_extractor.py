"""The bank statement extractor.

A bank statement is read in one model call with the schema prompt, and the reply is kept as the model
wrote it. Several statements can share one engine call, so the engine runs them concurrently.

Usage:
    from common.unified_bank_extractor import UnifiedBankExtractor

    extractor = UnifiedBankExtractor(generate_fn=processor.generate)
    schema, metadata = extractor.extract_bank_statement(image_path)
"""

import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SINGLE_PASS = "single_pass"
"""The name recorded for how a statement was read. Kept as it was when there were other ways."""


def serialise_bank_response(schema_fields: dict[str, str], metadata: dict[str, Any]) -> str:
    """Build the ``raw_response`` text of a bank statement's extraction record.

    The model's reply is returned unaltered in ``metadata["verbatim_response"]``, and that is written as
    it stands. Without one, the fields are written as flat ``FIELD: value`` lines for the clean stage's
    parser.
    """
    verbatim = metadata.get("verbatim_response")
    if verbatim is not None:
        return verbatim
    return "\n".join(f"{field}: {value}" for field, value in schema_fields.items())


def _safe_print(msg: str) -> None:
    """Print without triggering Rich console recursion in Jupyter.

    Writes directly to sys.__stdout__ to bypass Rich's file proxy
    which can cause RecursionError in Jupyter notebooks.
    """
    try:
        stdout = sys.__stdout__
        if stdout is not None:
            stdout.write(msg + "\n")
            stdout.flush()
    except Exception:
        pass


@contextmanager
def _bypass_rich_stdout():
    """Context manager to temporarily bypass Rich's stdout proxy.

    UnifiedBankExtractor uses print() statements internally which can
    cause RecursionError when Rich's console file proxy intercepts them.
    This context manager redirects stdout to the original sys.__stdout__
    during extraction to prevent the recursion.
    """
    original_stdout = sys.stdout
    try:
        if hasattr(original_stdout, "flush"):
            original_stdout.flush()
        sys.stdout = sys.__stdout__
        yield
    finally:
        sys.__stdout__.flush()
        sys.stdout = original_stdout
        if hasattr(sys.stdout, "flush"):
            sys.stdout.flush()


def _default_catalog():
    """Create a PromptCatalog with lazy import."""
    from common.prompt_catalog import PromptCatalog

    return PromptCatalog()


class UnifiedBankExtractor:
    """Reads bank statements, one model call each.

    Args:
        generate_fn: Callable(image, prompt, max_tokens, extra) -> str.
            Typically ``processor.generate`` from a DocumentProcessor.
        verbose: Enable verbose output.
        token_budgets: Completion-token budgets by name; ``extract_bank`` is the one used.
        generate_many_fn: Optional Callable(images, prompts, max_tokens, extra) -> list[str]:
            several independent calls submitted to the engine together.

    Example:
        extractor = UnifiedBankExtractor(generate_fn=processor.generate)
        schema, metadata = extractor.extract_bank_statement(image_path)
    """

    def __init__(
        self,
        generate_fn: Any,
        verbose: bool = True,
        token_budgets: dict[str, int] | None = None,
        *,
        generate_many_fn: Any = None,
    ):
        self.generate_fn = generate_fn
        self.generate_many_fn = generate_many_fn
        self.verbose = verbose
        self._extract_tokens = (token_budgets or {}).get("extract_bank", 4096)
        self._prompts = {
            "schema_fallback": _default_catalog().get_prompt("bank", "schema_fallback_extraction"),
        }

    def _log(self, msg: str) -> None:
        """Log a message to stdout only when verbose is enabled."""
        stdout = sys.__stdout__
        if self.verbose and stdout is not None:
            stdout.write(msg + "\n")
            stdout.flush()

    def _gen(self, image: Any, prompt: str, max_tokens: int) -> str:
        """Call the backend."""
        return self.generate_fn(image, prompt, max_tokens=max_tokens, extra=None)

    @property
    def can_batch(self) -> bool:
        """Whether statements can share an engine call: a batched call was given to make it."""
        return self.generate_many_fn is not None

    def _extract_single_pass(self, image: Any) -> tuple[dict[str, str], dict[str, Any]]:
        """Read a statement in one model call, keeping the reply as written.

        Sends the schema prompt alone. Nothing is repaired afterwards: the lists
        are not cut to a common length and the statement period is not recomputed
        from the transaction dates. The reply goes to the clean stage as the model
        wrote it, so the model is scored on exactly what it returned.

        Returns:
            Tuple of (schema_fields, metadata). ``metadata["verbatim_response"]``
            holds the unaltered reply, which ``serialise_bank_response`` writes
            to the extraction record in place of the parsed fields.
        """
        self._log("[UBE] Single pass: extracting with the schema prompt alone...")
        with _bypass_rich_stdout():
            response = self._gen(image, self._prompts["schema_fallback"], self._extract_tokens)
        self._log(f"[UBE]   Raw response length: {len(response)} chars")
        return self._single_pass_result(response)

    def _single_pass_result(self, response: str) -> tuple[dict[str, str], dict[str, Any]]:
        """The (schema_fields, metadata) of one single-pass reply, batched or not."""
        metadata: dict[str, Any] = {
            "strategy_used": SINGLE_PASS,
            "turns_executed": 1,
            "headers_detected": [],
            "column_mapping": {},
            "raw_responses": {SINGLE_PASS: response},
            "verbatim_response": response,
        }
        return self._parse_schema_response(response), metadata

    def extract_bank_statements_single_pass(
        self, image_paths: list[str | Path]
    ) -> list[tuple[dict[str, str], dict[str, Any]]]:
        """Read several statements single pass in one engine call.

        Every statement gets the request the one-at-a-time path gives it: the
        schema prompt alone, and the same completion budget. They
        are submitted together so the engine can run them concurrently, and each
        reply is kept as written.

        Args:
            image_paths: Paths of the statement images.

        Returns:
            One (schema_fields, metadata) per image, in input order, each equal
            to what extract_bank_statement returns for that reply.

        Raises:
            ValueError: If no batched call was supplied.
        """
        from PIL import Image

        if not self.can_batch:
            raise ValueError(
                "What: bank statements cannot share an engine call, because no batched call "
                "(generate_many_fn) was given to UnifiedBankExtractor.\n"
                "Where: the UnifiedBankExtractor built in stages/extract.py.\n"
                "Expected: generate_many_fn=processor.generate_many, from a backend that has "
                "generate_many.\n"
                "How to fix: call extract_bank_statement once per image, or build the extractor "
                "with a batched call."
            )

        images = [Image.open(path).convert("RGB") for path in image_paths]
        prompts = [self._prompts["schema_fallback"]] * len(images)
        self._log(f"[UBE] Single pass: {len(images)} statements in one engine call...")
        with _bypass_rich_stdout():
            responses = self.generate_many_fn(images, prompts, max_tokens=self._extract_tokens, extra=None)
        if len(responses) != len(images):
            raise ValueError(
                f"backend returned {len(responses)} replies for {len(images)} statements; "
                f"matching by position is unsafe"
            )
        return [self._single_pass_result(response) for response in responses]

    def _parse_schema_response(self, response: str) -> dict[str, str]:
        """Parse schema-format response into field dictionary."""
        result = {}
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Match "FIELD_NAME: value" or "**FIELD_NAME**: value" pattern
            # Models sometimes wrap field names in markdown bold
            if ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    # Remove markdown bold markers (**) from field name
                    field = parts[0].strip().replace("*", "").upper()
                    value = parts[1].strip()

                    # Only capture known schema fields
                    if field in [
                        "DOCUMENT_TYPE",
                        "STATEMENT_DATE_RANGE",
                        "TRANSACTION_DATES",
                        "LINE_ITEM_DESCRIPTIONS",
                        "TRANSACTION_AMOUNTS_PAID",
                    ]:
                        result[field] = value

        return result

    def extract_bank_statement(self, image_path: str | Path) -> tuple[dict[str, str], dict[str, Any]]:
        """Read one bank statement.

        Args:
            image_path: Path to the statement image.

        Returns:
            Tuple of (schema_fields, metadata).
        """
        from PIL import Image

        if self.verbose:
            _safe_print(f"\n[UBE] >>> START extract_bank_statement({Path(image_path).name})")
        image = Image.open(image_path).convert("RGB")
        return self._extract_single_pass(image)
