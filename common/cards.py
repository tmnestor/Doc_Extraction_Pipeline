"""How many cards a run has: counted in one place, one way.

Two things are decided by the number of cards: how many engines to start when a model fits on one card,
and how many cards one engine is split across when it does not. Until 25 November 2026 each was worked
out by its own count, one asking torch and the other asking nvidia-smi, so the two could disagree about
the machine they were both on.

The count is every card the process can see: those named in ``CUDA_VISIBLE_DEVICES`` if that is set, else
those ``nvidia-smi`` lists. A machine on which neither says anything is taken to have one. Until
26 November 2026 the settings file could give a number in their place, and no run ever did.
"""

import os
import shutil
import subprocess


def cards_to_use() -> int:
    """How many cards the run has, never less than 1."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return max(1, len([card for card in visible.split(",") if card.strip()]))
    return max(1, listed_by_nvidia_smi())


def listed_by_nvidia_smi() -> int:
    """How many cards nvidia-smi lists; 0 where it is not on the machine, or fails."""
    if shutil.which("nvidia-smi") is None:
        return 0
    listed = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
    if listed.returncode != 0:
        return 0
    return len([line for line in listed.stdout.splitlines() if line.strip()])
