"""JSONL I/O helpers for stage artifact files.

Each stage writes one JSON object per line. Writers flush per-record
so that partial output survives crashes (critical at 10K images).
"""

import json
from pathlib import Path
from typing import Any


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


class Journal:
    """A stage's records, kept beside its output as each is made, until the stage has written its output.

    A stage writes its output when it has read every page, in the pages' order. Until then nothing of
    its work would be on disk, and a run that stopped part way would lose all of it. So each record is
    also written here the moment it is made, in whatever order the engines finish. When the stage ends
    it writes its output as it always has and throws the journal away.

    A journal still there when a stage starts is what an earlier run had read before it stopped.
    ``recover`` adds those records to the output, in the pages' own order, and the stage then carries on
    from them like any other earlier output.

    The output itself is never written out of order: the order of its pages decides how a later stage
    groups them into engine calls, which must not depend on which card finished first.
    """

    def __init__(self, output_path: Path) -> None:
        self.output = output_path
        self.path = output_path.with_name(output_path.name + ".journal")
        self._file: Any = None

    def recover(self, order: list[str]) -> list[dict[str, Any]]:
        """Add what an interrupted run had read to the output, and remove the journal.

        Args:
            order: The names of the stage's pages, in the stage's own order. The recovered records are
                added in that order, whatever order they were made in.

        Returns:
            The records added. Empty when there is no journal.
        """
        if not self.path.is_file():
            return []
        already = read_completed_images(self.output)
        kept: dict[str, dict[str, Any]] = {}
        for line in self.path.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # the line being written when the run stopped
            name = record.get("image_name") if isinstance(record, dict) else None
            if name and name not in already:
                kept.setdefault(name, record)
        place = {name: position for position, name in enumerate(order)}
        records = sorted(kept.values(), key=lambda record: place.get(record["image_name"], len(place)))
        if records:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            append_jsonl(self.output, records)
        self.path.unlink()
        return records

    def __enter__(self) -> "Journal":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close the journal. It is kept: only ``discard``, after the output is written, removes it."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def write(self, record: dict[str, Any]) -> None:
        """Keep one record, on disk before this returns."""
        if self._file is None:
            msg = "Journal not open. Use as context manager."
            raise RuntimeError(msg)
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()

    def discard(self) -> None:
        """Remove the journal, once the stage's output holds everything it held."""
        if self.path.is_file():
            self.path.unlink()


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
