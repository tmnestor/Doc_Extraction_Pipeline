"""How many cards a run has: counted in one place, one way.

Two things are decided by the number of cards: how many engines to start when a model fits on one card,
and how many cards one engine is split across when it does not. Until 25 November 2026 each was worked
out by its own count, one asking torch and the other asking nvidia-smi, so the two could disagree about
the machine they were both on.

The count is the settings' number when they give one. Otherwise it is every card the process can see:
those named in ``CUDA_VISIBLE_DEVICES`` if that is set, else those ``nvidia-smi`` lists. A machine on
which neither says anything is taken to have one.
"""

import logging
import os
import shutil
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


def cards_to_use(settings: Any) -> int:
    """How many cards the run has.

    Args:
        settings: The run's settings. ``num_gpus`` above 0 is the number; 0 is every card that can be
            seen.

    Returns:
        The number of cards, never less than 1.
    """
    if settings.num_gpus and settings.num_gpus > 0:
        return int(settings.num_gpus)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return max(1, len([card for card in visible.split(",") if card.strip()]))
    if shutil.which("nvidia-smi") is None:
        return 1
    listed = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
    if listed.returncode != 0:
        return 1
    return max(1, len([line for line in listed.stdout.splitlines() if line.strip()]))
