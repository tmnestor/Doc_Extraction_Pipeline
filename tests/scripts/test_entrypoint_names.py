"""The entry script refuses a variable set under the prefix it used before 21 November 2026.

The script's own variables were renamed. A job that still sets an old name must stop and say so: if the
old name were only ignored, a run would write its log where nobody asked and look as if it had worked.
"""

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OLD_PREFIX = "L" + "MM_"


def start(extra: dict[str, str]) -> subprocess.CompletedProcess[str]:
    environment = {k: v for k, v in os.environ.items() if not k.startswith(OLD_PREFIX)}
    return subprocess.run(
        ["bash", "entrypoint.sh"],
        cwd=REPO_ROOT,
        env={**environment, "DOC_EXTRACT_CONDA_ENV": "/nowhere/at/all", **extra},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_an_old_name_stops_the_run_with_a_full_diagnostic() -> None:
    result = start({OLD_PREFIX + "LOG_DIR": "/somewhere/logs"})

    assert result.returncode == 1
    assert OLD_PREFIX + "LOG_DIR" in result.stdout
    assert all(part in result.stdout for part in ("What:", "Where:", "Expected:", "Fix:"))
    assert "DOC_EXTRACT_LOG_DIR" in result.stdout


def test_the_new_names_get_past_that_check() -> None:
    """It then stops for want of the conda environment, which is not on this machine, naming the new variable."""
    result = start({"DOC_EXTRACT_LOG_DIR": "/somewhere/logs"})

    assert result.returncode == 1
    assert "no longer reads" not in result.stdout
    assert "DOC_EXTRACT_CONDA_ENV" in result.stdout


def test_the_old_prefix_is_written_nowhere_else() -> None:
    listed = subprocess.run(
        ["git", "grep", "-l", OLD_PREFIX[:-1], "--", ".", ":!README.md"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()

    assert listed == []
