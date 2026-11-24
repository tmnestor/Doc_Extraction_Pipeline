"""Write the recorded output that tests/replay/test_replay.py compares against.

    python tests/replay/make_reference.py

Run once, from the repository's top folder, on code known to be good. It copies each run's recorded model
replies into the reference folder, puts them through `clean` and `evaluate`, and leaves their output
beside them. It then does the same again into a temporary folder and stops with an error unless the two
agree, since a reference that differs from one making to the next can prove nothing.

It refuses to write over a reference that exists. To make a new one, move the old one away by hand.
"""

import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

spec = importlib.util.spec_from_file_location("test_replay", HERE / "test_replay.py")
assert spec is not None and spec.loader is not None
test_replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_replay)

OUTPUTS = ("cleaned_extractions.jsonl", "evaluation_results.jsonl")


def main() -> None:
    if test_replay.REFERENCE.exists():
        raise SystemExit(f"A reference already exists at {test_replay.REFERENCE}. Move it away first.")
    for run in test_replay.RUNS:
        folder = test_replay.REFERENCE / run
        folder.mkdir(parents=True)
        for name in ("raw_extractions.jsonl", "classifications.jsonl"):
            shutil.copyfile(test_replay.DATA / run / name, folder / name)
        test_replay.replay(run, into=folder)
        with tempfile.TemporaryDirectory() as again:
            test_replay.replay(run, into=Path(again))
            for name in OUTPUTS:
                if test_replay.comparable(Path(again) / name) != test_replay.comparable(folder / name):
                    raise SystemExit(f"{run}: {name} differs between two makings. Not a usable reference.")
        records = len((folder / "evaluation_results.jsonl").read_text().splitlines())
        print(f"{run}: {records} records, the same twice")


if __name__ == "__main__":
    main()
