"""JSONL I/O helpers for stage artifact files.

Each stage writes one JSON object per line. Writers flush per-record
so that partial output survives crashes (critical at 10K images).
"""

import json
import logging
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


logger = logging.getLogger("stages")


def _put_in_place(path: Path, lines: Iterable[str]) -> None:
    """Write a file beside itself and move it into place whole.

    A reader finds the file as it was or as it is to be, and never half of it: a run stopped while
    writing leaves the earlier file untouched.
    """
    rewritten = path.with_name(path.name + ".rewriting")
    with rewritten.open("w") as f:
        for line in lines:
            f.write(line)
    os.replace(rewritten, path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> int:
    """Write a stage's records as its output, in the place of any earlier output.

    The file is put in place whole. Until 25 November 2026 it was written where it stood, so a run
    stopped while writing left half a file under the name the next stage reads.

    Returns the number of records written.
    """
    _put_in_place(path, (json.dumps(record, default=str) + "\n" for record in records))
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


def start_afresh(asked_for: bool, *files: Path) -> None:
    """Delete a stage's own earlier output, if that was asked for. Never a folder, never a log.

    Args:
        asked_for: Whether the job asked for it (``CLEAR_PREV_OUTPUT=true``).
        files: The stage's output files. With each goes its journal: what an interrupted run had read
            and not yet written.
    """
    if not asked_for:
        return
    for output in files:
        for file in (output, Journal(output).path):
            if file.is_file():
                file.unlink()
                logger.info("CLEAR_PREV_OUTPUT: removed %s", file)


class Journal:
    """A stage's records, kept beside its output as each is made, until the stage has written its output.

    A stage writes its output when it has read every page, in the pages' order. Until then nothing of
    its work would be on disk, and a run that stopped part way would lose all of it. So each record is
    also written here the moment it is made, in whatever order the engines finish. When the stage ends
    it writes its output as it always has and throws the journal away.

    A journal still there when a stage starts is what an earlier run had read before it stopped.
    ``recover`` adds those records to the output, in the pages' own order, and the stage then carries on
    from them like any other earlier output.

    The output itself is never left out of order: the order of its pages decides how a later stage
    groups them into engine calls, which must not depend on which card finished first, nor on whether
    the run was stopped on its way. ``stages.carry_on`` is the one place a journal is used: it adds a
    run's records to the output and puts the whole in order (``in_the_order``) before it removes the
    journal.
    """

    def __init__(self, output_path: Path) -> None:
        self.output = output_path
        self.path = output_path.with_name(output_path.name + ".journal")
        self._file: Any = None

    def recover(
        self, order: list[str], *, made_of: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        """Add what an interrupted run had read to the output, and remove the journal.

        Args:
            order: The names of the stage's pages, in the stage's own order. The recovered records are
                added in that order, whatever order they were made in.
            made_of: What makes the stage's record of what the journal kept, for a stage whose journal
                keeps the model's reply and whose output holds what was done about it. None when the
                two are the same record.

        Returns:
            The records recovered, as the journal held them. Empty when there is no journal.
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
            append_jsonl(self.output, [made_of(record) for record in records] if made_of else records)
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


def take_out(path: Path, read_again: Callable[[dict[str, Any]], bool]) -> list[dict[str, Any]]:
    """Take out of a stage's output the records of pages the stage is to read again.

    A stage carries on from its earlier output by the names of the pages in it. A page recorded as
    failed is in it too, and until 25 November 2026 was therefore never read again: an engine that
    stopped answering part way left every page after it recorded as failed, and done.

    The file is written again without those records, and put in place whole, so that a run stopped
    during this leaves either the old file or the new one. A line that cannot be read, which a run
    stopped while writing leaves, is dropped with them. A file with nothing to take out is not touched.

    Args:
        path: The stage's output.
        read_again: Says of a record whether its page is to be read again.

    Returns:
        The records taken out.
    """
    if not path.is_file():
        return []
    kept: list[str] = []
    taken: list[dict[str, Any]] = []
    spoilt = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            spoilt += 1
            continue
        if isinstance(record, dict) and read_again(record):
            taken.append(record)
        else:
            kept.append(line)
    if taken or spoilt:
        _put_in_place(path, (line + "\n" for line in kept))
    return taken


def in_the_order(path: Path, order: list[str]) -> None:
    """Put a stage's records in the order of its pages, whatever order they were added in.

    A stage adds records as it makes them: those an earlier run left, those it reads, those it reads
    again. The file is then put in the one order a run straight through would have left it in, so that
    the same pages give the same file however the run went. The order of a file decides how the next
    stage groups its pages into engine calls.

    Each line is kept as the text it is. A record of a page that is not among ``order`` is kept, after
    those that are, in the place it had among its like: the folder or the selection of pages may have
    changed since it was made. Lines of one page stay together. A line that cannot be read, which a
    run stopped while writing leaves, is dropped and said. A file already in order is not touched.

    Args:
        path: The stage's output. Nothing is done if it is not there.
        order: The names of the stage's pages, in the order the output is to hold them.
    """
    if not path.is_file():
        return
    place = {name: position for position, name in enumerate(order)}
    as_it_is = path.read_text()
    placed: list[tuple[int, str]] = []
    spoilt = 0
    for line in as_it_is.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            spoilt += 1
            continue
        name = record.get("image_name") if isinstance(record, dict) else None
        placed.append((place.get(name, len(place)) if isinstance(name, str) else len(place), line))
    if spoilt:
        logger.warning(
            "%d line%s that could not be read %s dropped from %s",
            spoilt,
            "" if spoilt == 1 else "s",
            "was" if spoilt == 1 else "were",
            path,
        )
    # A stable sort: lines with the same place keep the order they had.
    in_order = "".join(line + "\n" for _, line in sorted(placed, key=lambda entry: entry[0]))
    if in_order != as_it_is:
        _put_in_place(path, [in_order])


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
