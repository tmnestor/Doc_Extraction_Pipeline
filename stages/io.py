"""JSONL I/O helpers for stage artifact files.

Each stage writes one JSON object per line. Writers flush per-record
so that partial output survives crashes (critical at 10K images).
"""

import json
from pathlib import Path
from typing import Any, Protocol


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> int:
    """Write records to a JSONL file (atomic: writes all at once).

    Returns number of records written.
    """
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")
    return len(records)


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> int:
    """Append records to a JSONL file (created if absent).

    The append counterpart to :func:`write_jsonl`: used when resuming a stage so
    already-written records are preserved rather than truncated. ``write_jsonl``
    opens ``"w"`` (truncate); this opens ``"a"``.

    Returns number of records appended.
    """
    with path.open("a") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")
    return len(records)


class RecordSink(Protocol):
    """Anything a stage can write its records to, one at a time.

    The one-engine stage writes to a file as it goes (`StreamingJsonlWriter`). A worker on the two-engine
    path collects its records and hands them back. Naming what the two have in common lets the stage's own
    functions serve both.
    """

    def write(self, record: dict[str, Any]) -> None: ...


class StreamingJsonlWriter:
    """Append-mode JSONL writer that flushes per record.

    Use this for long-running stages (extract) where crash recovery matters.

    Usage:
        with StreamingJsonlWriter(path) as writer:
            for image in images:
                result = process(image)
                writer.write(result)
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._file = None
        self._count = 0

    def __enter__(self):
        self._file = self._path.open("a")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file is not None:
            self._file.close()
            self._file = None

    def write(self, record: dict[str, Any]) -> None:
        """Write a single record and flush to disk immediately."""
        if self._file is None:
            msg = "Writer not open. Use as context manager."
            raise RuntimeError(msg)
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()
        self._count += 1

    @property
    def count(self) -> int:
        """Number of records written so far."""
        return self._count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all records from a JSONL file."""
    records: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_completed_images(path: Path) -> set[str]:
    """Read image names from a partial JSONL output for resumption.

    Returns a set of image_name values already processed.
    """
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    name = record.get("image_name", "")
                    if name:
                        completed.add(name)
                except json.JSONDecodeError:
                    pass  # Skip corrupted trailing line from crash
    return completed


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


def discover_images(
    data_dir: Path,
    document_types: list[str] | None = None,
    max_images: int | None = None,
) -> list[Path]:
    """Discover images in a data directory, applying the configured selection.

    Both filters live HERE and not at the call sites. Until 21 September 2026 the stages
    called this with the folder alone and so ignored them: asking for 12 images processed
    all 165 while the YAML read as applied. Keeping the selection inside the discovery
    function means a caller cannot omit it.

    Args:
        data_dir: Directory to scan for images.
        document_types: Keep only images whose filename contains one of these
            (case-insensitive). ``None`` keeps every type.
        max_images: Keep at most this many, applied AFTER sorting and type
            filtering so the subset is deterministic and representative.
            ``None`` means no limit.

    Returns:
        Images sorted by lowercase filename, filtered and truncated as requested.
    """
    images: list[Path] = []

    for ext in IMAGE_EXTENSIONS:
        images.extend(data_dir.glob(f"*{ext}"))
        images.extend(data_dir.glob(f"*{ext.upper()}"))

    # Sort by filename for reproducibility
    images = sorted(images, key=lambda p: p.name.lower())

    # Filter by document type if specified (based on filename patterns)
    if document_types:
        filtered = []
        type_patterns = [t.lower() for t in document_types]
        for img in images:
            name_lower = img.name.lower()
            if any(pattern in name_lower for pattern in type_patterns):
                filtered.append(img)
        images = filtered

    # Truncate LAST: a limit applied before the type filter would eat images the
    # filter was going to discard, and silently return fewer than requested.
    if max_images is not None:
        images = images[:max_images]

    return images
