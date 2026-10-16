"""Strict parsing of numbers that arrive as text from untrusted score files.

`Fraction(text)` accepts exponents, so a short string such as "1e100000000" asks for an enormous integer. These
helpers accept only plain integers, decimals and ratios with a bounded number of digits.
"""

from __future__ import annotations

import re
from fractions import Fraction

MAX_DIGITS = 18
MAX_BEAT = 1e12
_INTEGER = re.compile(r"[+-]?[0-9]{1,9}")
_DECIMAL = re.compile(rf"[+-]?(?:[0-9]{{1,{MAX_DIGITS}}}(?:\.[0-9]{{0,{MAX_DIGITS}}})?|\.[0-9]{{1,{MAX_DIGITS}}})")
_RATIO = re.compile(rf"([+-]?[0-9]{{1,{MAX_DIGITS}}})\s*/\s*([0-9]{{1,{MAX_DIGITS}}})")


def parse_integer(text: str) -> int | None:
    return int(text) if _INTEGER.fullmatch(text.strip()) else None


def parse_decimal(text: str) -> Fraction | None:
    text = text.strip()
    return Fraction(text) if _DECIMAL.fullmatch(text) else None


def parse_ratio(text: str) -> Fraction | None:
    """A decimal such as 0.5 or a ratio such as -3/4."""
    text = text.strip()
    match = _RATIO.fullmatch(text)
    if match is None:
        return parse_decimal(text)
    denominator = int(match.group(2))
    return Fraction(int(match.group(1)), denominator) if denominator else None


def beat_float(value: Fraction | float, label: str = "position") -> float:
    """Convert a position in quarter notes to a float, refusing values no real score reaches."""
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} exceeds finite numeric range") from error
    if not -MAX_BEAT <= number <= MAX_BEAT:
        raise ValueError(f"{label} {number:g} is outside the supported range")
    return number
