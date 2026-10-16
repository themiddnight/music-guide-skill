"""Shared parser limits and scalar validation."""

from __future__ import annotations
import math
from pathlib import Path

MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_MIDI_BYTES = 32 * 1024 * 1024
MAX_XML_BYTES = 8 * 1024 * 1024
MAX_ZIP_ENTRIES = 256
MAX_ZIP_BYTES = 64 * 1024 * 1024
MAX_AUDIO_BYTES = 4 * 1024 * 1024 * 1024


def _read_bounded(path: Path, limit: int) -> bytes:
    if path.stat().st_size > limit:
        raise ValueError(f"file exceeds {limit} byte limit")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"file exceeds {limit} byte limit")
    return data


def _finite_number(value: object, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{label} must be a finite positive number" if positive else f"{label} must be finite")
    return number


