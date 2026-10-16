#!/usr/bin/env python3
import sys
from pathlib import Path

# Checked before any package import: the modules use syntax and typing features that older interpreters reject.
if sys.version_info < (3, 10):
    print(f"music-guide: Python 3.10 or newer is required, but this is Python {'.'.join(map(str, sys.version_info[:3]))}. Run it with a "
          f"newer interpreter, for example: uv run --no-project --python 3.12 --with-requirements "
          f"\"{Path(__file__).resolve().parent.parent / 'requirements.txt'}\" python \"{Path(__file__).resolve()}\" ..., "
          f"or create the environment with python3.12 -m venv .venv.", file=sys.stderr)
    raise SystemExit(3)

HERE = Path(__file__).resolve().parent
# Isolated mode (-I) drops the script directory from sys.path, so add it back explicitly.
sys.path.insert(0, str(HERE))

try:
    from music_guide.cli import main
except ModuleNotFoundError as error:
    if (error.name or "").split(".")[0] == "music_guide":
        raise
    requirements = HERE.parent / "requirements.txt"
    print(f"music-guide: missing Python module '{error.name}'. Create a virtual environment with Python 3.10 or newer "
          f"(for example python3.12 -m venv .venv) and install {requirements} into it: "
          f".venv/bin/python -m pip install -r \"{requirements}\", then run analyze.py with .venv/bin/python. "
          f"With uv: uv run --no-project --python 3.12 --with-requirements \"{requirements}\" python \"{Path(__file__).resolve()}\" ...",
          file=sys.stderr)
    raise SystemExit(3)

if __name__ == "__main__":
    raise SystemExit(main())
