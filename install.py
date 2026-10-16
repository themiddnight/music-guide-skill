#!/usr/bin/env python3
"""Install a standalone copy of the Music Guide skill at an explicit path."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

PAYLOAD = Path(__file__).resolve().parent / "skills" / "music-guide"


def inventory(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts or path.name in {".DS_Store"} or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError(f"symlink is not allowed in a skill payload: {path}")
        if path.is_file():
            result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", required=True, type=Path, help="Exact destination directory for the skill")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    dest = args.dest.expanduser().absolute()
    try:
        payload_files = inventory(PAYLOAD)
        if dest.is_symlink():
            raise ValueError(f"destination is a symlink: {dest}")
        if dest.exists():
            if not dest.is_dir() or inventory(dest) != payload_files:
                raise ValueError(f"destination already exists with different content: {dest}")
            print(f"Already installed: {dest}")
            return 0
        if args.dry_run:
            print(f"Would install {len(payload_files)} files to {dest}")
            return 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".music-guide-", dir=dest.parent) as temporary:
            staged = Path(temporary) / "music-guide"
            shutil.copytree(PAYLOAD, staged, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
            if inventory(staged) != payload_files:
                raise ValueError("staged skill copy failed integrity check")
            if dest.exists():
                raise ValueError(f"destination appeared during install: {dest}")
            os.rename(staged, dest)
        print(f"Installed: {dest}")
        return 0
    except (OSError, ValueError) as error:
        print(f"music-guide install: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
