"""What made the files in an output folder: a note each stage that uses the cards leaves there, and checks.

A stage carries on from where an earlier run stopped, by the names of the pages already in its output. So
a second run into the same folder, with another model, another folder of pages or another version of the
code, skips every page the first one finished and reports a score that is partly the first run's. Nothing
in the result shows it.

So each of those stages writes down, in ``run_manifest.json`` in the output folder, what it was run with:
the model, the folder of pages, the commit of the code, and the settings file with the SHA-256 of its
contents. A stage that finds its own earlier note, and
is about to carry on from that run's output, checks that it is the same run. If it is not, it stops
before any work and says what differs.

The note is also the record of which commit produced a run, for whoever scores it later.

A stage that is starting afresh (``CLEAR_PREV_OUTPUT=true``) has had its earlier output deleted, so it
checks nothing and writes its note anew. The clean and evaluate stages write over their output every time
and may be run again anywhere, so they leave no note and check none; they say what the notes hold.
"""

import hashlib
import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from common.config_schema import ConfigError

logger = logging.getLogger(__name__)

FILE = "run_manifest.json"
# The stages that carry on from their own earlier output.
STAGES = ("split", "classify", "extract")
UNKNOWN = "unknown"
# What must be the same for a stage to carry on from an earlier run, and what each is called in a message.
_SAME_RUN = (("model", "the model"), ("image_dir", "the folder of pages"), ("commit", "the code's commit"))


@dataclass(frozen=True)
class StageNote:
    """What one stage was run with."""

    model: str
    image_dir: str
    # The short commit of the code, or "unknown" where it cannot be told (no git, or not a checkout).
    commit: str
    # Whether tracked files differed from that commit; None where it cannot be told.
    uncommitted_changes: bool | None
    started: str
    last_run: str
    # The settings file the stage read, and the SHA-256 of its contents. The commit says what the
    # shipped file held, and a job may name a file of its own (``run_config``), which no commit does.
    # None in a note written before 26 November 2026.
    settings_file: str | None = None
    settings_sha256: str | None = None


def code_version(repository: Path) -> tuple[str, bool | None]:
    """The short commit of the code in ``repository``, and whether tracked files differ from it.

    Where git is missing, or the folder is not a checkout, the commit is "unknown" and the second is None.
    """

    def git(*arguments: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", "-C", str(repository), *arguments],
                capture_output=True,
                text=True,
                check=False,
                timeout=20,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = git("rev-parse", "--short", "HEAD")
    if not commit:
        return UNKNOWN, None
    changed = git("status", "--porcelain", "--untracked-files=no")
    return commit, None if changed is None else bool(changed)


def _sha256(file: Path | None) -> str | None:
    """The SHA-256 of a file's contents; None if no file is given or it cannot be read."""
    if file is None:
        return None
    try:
        return hashlib.sha256(file.read_bytes()).hexdigest()
    except OSError:
        return None


def read(output: Path) -> dict[str, StageNote]:
    """The notes in an output folder, by stage. Empty if the folder has none.

    Raises:
        ConfigError: If the file is there and cannot be read as notes.
    """
    path = output / FILE
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {stage: StageNote(**note) for stage, note in raw["stages"].items()}
    except (ValueError, KeyError, TypeError, AttributeError) as err:
        raise ConfigError(
            f"What: the run's notes cannot be read ({type(err).__name__}: {err}).\n"
            f"Where: {path.resolve()}\n"
            f'Expected: the file a stage writes, e.g. {{"stages": {{"classify": {{"model": "...", '
            f'"image_dir": "...", "commit": "...", "uncommitted_changes": false, "started": "...", '
            f'"last_run": "..."}}}}}}\n'
            f"How to fix: give this run a folder of its own in `output`. If the folder's earlier output "
            f"is known to be this run's, delete {FILE} and start the stage again."
        ) from err


def _write(output: Path, notes: dict[str, StageNote]) -> None:
    body: dict[str, Any] = {"stages": {stage: asdict(note) for stage, note in notes.items()}}
    (output / FILE).write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")


def check_and_note(
    output: Path,
    stage: str,
    *,
    model: str,
    image_dir: Path | str,
    repository: Path,
    starting_afresh: bool,
    settings_file: Path | None = None,
    now: datetime | None = None,
) -> StageNote:
    """Check that a stage about to carry on in an output folder is the run that began there, and note it.

    Args:
        output: The run's output folder, which exists.
        stage: The stage, one of ``STAGES``.
        model: The model the stage is run with.
        image_dir: The folder of pages it reads.
        settings_file: The settings file it reads, noted with the SHA-256 of its contents. A stage
            that carries on with another file, or the same file changed, is allowed and says so.
        repository: The code's own folder, for its commit.
        starting_afresh: Whether the stage's earlier output has been deleted. Nothing is checked then.
        now: The time to note, or None for the present.

    Returns:
        The note written for the stage.

    Raises:
        ConfigError: If the stage's earlier note is of a run with another model, folder of pages or
            commit, or the notes cannot be read.
    """
    notes = read(output)
    commit, changed = code_version(repository)
    moment = (now or datetime.now()).isoformat(timespec="seconds")
    this = StageNote(
        model=model,
        image_dir=str(image_dir),
        commit=commit,
        uncommitted_changes=changed,
        started=moment,
        last_run=moment,
        settings_file=None if settings_file is None else str(settings_file),
        settings_sha256=_sha256(settings_file),
    )
    earlier = None if starting_afresh else notes.get(stage)
    if earlier is not None:
        differs = [
            f"{called} was {getattr(earlier, name)}, and is now {getattr(this, name)}"
            for name, called in _SAME_RUN
            if getattr(earlier, name) != getattr(this, name)
            # A commit that could not be told, then or now, is not evidence of another run.
            and UNKNOWN not in (getattr(earlier, name), getattr(this, name))
        ]
        if differs:
            afresh = (
                f"delete {output / 'split'}"
                if stage == "split"
                else "set CLEAR_PREV_OUTPUT=true, which deletes this stage's earlier output"
            )
            raise ConfigError(
                f"What: the output folder holds {stage} output from another run: {'; '.join(differs)}. "
                f"The stage would carry on from that output, skip every page it holds, and report a "
                f"result that is partly the other run's.\n"
                f"Where: {(output / FILE).resolve()} (stages.{stage}), against this job's model, "
                f"image_dir and the commit of the code.\n"
                f"Expected: a stage that carries on in a folder is the run that began there: "
                f"model={earlier.model}, image_dir={earlier.image_dir}, commit {earlier.commit}.\n"
                f"How to fix: give this run a folder of its own in `output`. To replace the earlier "
                f"{stage} output with this run's, {afresh}."
            )
        before, now_held = earlier.settings_sha256, this.settings_sha256
        if before is not None and now_held is not None and before != now_held:
            logger.warning(
                "%s carries on with settings that are not the ones it began with: the settings file was "
                "%s (SHA-256 %s) and is now %s (SHA-256 %s). The output is then of two sets of settings.",
                stage,
                earlier.settings_file,
                before[:12],
                this.settings_file,
                now_held[:12],
            )
        this = StageNote(**{**asdict(this), "started": earlier.started})
    notes[stage] = this
    _write(output, notes)
    logger.info(
        "Noted in %s: %s by %s at commit %s%s",
        output / FILE,
        stage,
        model,
        commit,
        " (with uncommitted changes)" if changed else "",
    )
    return this


def warn_if_made_another_way(output: Path, stage: str, after: str) -> None:
    """Say so if ``stage`` reads the output of ``after`` and the two were run with another model or commit.

    That is allowed: a stage may be run again on an earlier stage's output after the code has changed.
    It is said because the result is then of two versions, and should not be read as one's.
    """
    notes = read(output)
    mine, theirs = notes.get(stage), notes.get(after)
    if mine is None or theirs is None:
        return
    for name, called in (("model", "the model"), ("commit", "the code's commit")):
        ours, earlier = getattr(mine, name), getattr(theirs, name)
        if ours != earlier and UNKNOWN not in (ours, earlier):
            logger.warning(
                "%s reads the output of %s, which was run another way: %s was %s there and is %s here.",
                stage,
                after,
                called,
                earlier,
                ours,
            )


def describe(output: Path) -> list[str]:
    """A line for each stage's note, for a stage that reads the folder to say what made it."""
    return [
        f"{stage}: {note.model}, pages from {note.image_dir}, commit {note.commit}"
        f"{' with uncommitted changes' if note.uncommitted_changes else ''}, last run {note.last_run}"
        f"{f', settings {note.settings_file} (SHA-256 {note.settings_sha256[:12]})' if note.settings_sha256 else ''}"
        for stage, note in read(output).items()
    ]
