#!/usr/bin/env python3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Isolated mode (-I) drops the script directory from sys.path, so add it back explicitly.
sys.path.insert(0, str(HERE))

try:
    from music_guide.cli import main
except ModuleNotFoundError as error:
    if (error.name or "").split(".")[0] == "music_guide":
        raise
    requirements = HERE.parent / "requirements.txt"
    print(f"music-guide: missing Python module '{error.name}'. Create a virtual environment and install "
          f"{requirements} into it: python3 -m venv .venv && .venv/bin/python -m pip install -r \"{requirements}\", "
          f"then run analyze.py with .venv/bin/python.", file=sys.stderr)
    raise SystemExit(3)

if __name__ == "__main__":
    raise SystemExit(main())
