"""Rhythm formulas shared by audio estimates and stored notes.

Pure Python on purpose, so a run over MIDI or project files alone never pays about 0.7 s to import numpy.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Sequence

# Off-beat positions, as a fraction of the beat, that count toward swing: straight 0.5 up to 4:1 at 0.8.
SWING_WINDOW = (0.4, 0.8)


def _factors(count: int, what: str) -> list[int]:
    factors, rest = [], count
    for prime in (2, 3):
        while rest % prime == 0:
            factors.append(prime)
            rest //= prime
    if rest != 1 or count < 1:
        raise ValueError(f"no metric hierarchy for {count} {what}: only groupings of 2 and 3 are defined")
    return factors


@dataclass(frozen=True)
class Meter:
    numerator: int
    denominator: int

    def levels(self, subdivision: int) -> list[int]:
        """Divisions from the bar down to `subdivision` ticks per denominator note.

        Compound meters (6, 9 or 12 over 8) divide the bar into dotted beats of three eighths first.
        """
        compound = self.denominator == 8 and self.numerator % 3 == 0 and self.numerator > 3
        bar = _factors(self.numerator // 3, "beats") + [3] if compound else _factors(self.numerator, "beats")
        return bar + (_factors(subdivision, "subdivisions") if subdivision > 1 else [])

    def weights(self, subdivision: int) -> list[int]:
        """Longuet-Higgins and Lee metrical weight of each tick: 0 at the bar, one less per level below it."""
        levels = self.levels(subdivision)
        ticks = math.prod(levels)
        strides = [ticks // math.prod(levels[:depth]) for depth in range(len(levels) + 1)]
        return [-next(depth for depth, stride in enumerate(strides) if tick % stride == 0) for tick in range(ticks)]


def swing_ratio(onsets_in_beats: Sequence[float]) -> float | None:
    """Long-to-short ratio f / (1 - f) at the median off-beat position f inside SWING_WINDOW, or None without one."""
    fractions = [position % 1 for position in onsets_in_beats if SWING_WINDOW[0] < position % 1 < SWING_WINDOW[1]]
    if not fractions:
        return None
    middle = statistics.median(fractions)
    return middle / (1 - middle)
