"""Typed dataclasses for batch processing pipeline phases.

These replace the anonymous dicts that flow between the detection,
extraction, and evaluation phases.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExtractionOutput:
    """Output of field extraction for a single image."""

    image_path: str
    image_name: str
    document_type: str
    extracted_data: dict[str, str]
    processing_time: float
    prompt_used: str
    raw_response: str = ""
    error: str | None = None
