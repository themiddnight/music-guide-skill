"""Rhythm formulas shared by audio estimates and stored notes.

Pure Python on purpose, so a run over MIDI or project files alone never pays about 0.7 s to import numpy.
"""
from __future__ import annotations

import math
import statistics
from bisect import bisect_right
from dataclasses import dataclass
from functools import cached_property, lru_cache
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


def stored_swing_ratio(onsets_in_beats: Sequence[float]) -> float | None:
    """Exact straight sixteenths read 1:1 rather than a mixed eighth/sixteenth median."""
    fractions = {round(position % 1, 6) for position in onsets_in_beats} - {0.0, 1.0}
    if fractions and all(min(abs(fraction - straight) for straight in (0.25, 0.5, 0.75)) <= 0.01
                         for fraction in fractions):
        return 1.0
    return swing_ratio(onsets_in_beats)


def onset_density(onsets_in_beats: Sequence[float], beats_per_bar: float) -> float:
    """Distinct note starts per bar that holds an onset, so a long rest is a phrase break."""
    if not onsets_in_beats or beats_per_bar <= 0:
        return 0.0
    positions = sorted(set(onsets_in_beats))
    bars = len({math.floor(position / beats_per_bar) for position in positions})
    return len(positions) / bars


@dataclass(frozen=True)
class BarGrid:
    """Meter changes at quarter-note beats. Each change starts a new bar."""

    points: tuple[tuple[float, Meter], ...]

    def __post_init__(self) -> None:
        if not self.points or any(not math.isfinite(beat) for beat, _ in self.points):
            raise ValueError("a bar grid needs finite meter positions")
        if any(right[0] <= left[0] for left, right in zip(self.points, self.points[1:])):
            raise ValueError("meter positions must increase")

    @cached_property
    def _segments(self) -> tuple[tuple[float, ...], tuple[int, ...]]:
        starts = tuple(point for point, _ in self.points)
        offsets = [0]
        for (start, meter), (end, _) in zip(self.points, self.points[1:]):
            length = meter.numerator * 4 / meter.denominator
            offsets.append(offsets[-1] + math.ceil((end - start) / length))
        return starts, tuple(offsets)

    def locate(self, beat: float) -> tuple[int, float, Meter] | None:
        """Absolute bar index, local quarter-note beat and meter, or None before the first known meter."""
        starts, offsets = self._segments
        segment = bisect_right(starts, beat) - 1
        if segment < 0:
            return None
        origin, meter = self.points[segment]
        length = meter.numerator * 4 / meter.denominator
        offset = beat - origin
        local_bar = math.floor((offset + 1e-9) / length)
        return offsets[segment] + local_bar, offset - local_bar * length, meter

    def density(self, onsets_in_beats: Sequence[float]) -> float | None:
        if not onsets_in_beats:
            return 0.0
        placed = [self.locate(beat) for beat in sorted(set(onsets_in_beats))]
        if any(item is None for item in placed):
            return None
        return len(placed) / len({item[0] for item in placed})


def syncopation(onsets_in_beats: Sequence[float], meter: Meter, subdivision: int = 4) -> float:
    """LHL metric displacement to the next stronger empty tick, averaged over occupied bars.

    Onsets are quarter-note beats. Polyphonic starts at one position count once. A tick's
    displacement is zero when that stronger tick also has an onset.
    """
    length = meter.numerator * 4 / meter.denominator
    origin = math.floor(min(onsets_in_beats, default=0) / length) * length
    grid = BarGrid(((origin, meter),))
    return syncopation_on_grid(onsets_in_beats, grid, subdivision)


@lru_cache(maxsize=128)
def _metric_pattern(meter: Meter, subdivision: int) -> tuple[tuple[int, ...], tuple[int | None, ...], int]:
    per_unit = subdivision * 4 / meter.denominator
    if per_unit < 1 or not per_unit.is_integer():
        raise ValueError(f"{meter.numerator}/{meter.denominator} cannot be placed on sixteenths")
    weights = tuple(meter.weights(int(per_unit)))
    stronger: list[int | None] = [None] * len(weights)
    stack: list[int] = []
    for tick in range(len(weights) - 1, -1, -1):
        while stack and weights[stack[-1]] <= weights[tick]:
            stack.pop()
        stronger[tick] = stack[-1] if stack else None
        stack.append(tick)
    upper = sum(weights[ahead] - weights[tick] for tick, ahead in enumerate(stronger) if ahead is not None)
    return weights, tuple(stronger), max(4 * meter.numerator, upper)


def syncopation_on_grid(onsets_in_beats: Sequence[float], grid: BarGrid, subdivision: int = 4) -> float:
    if not onsets_in_beats:
        return 0.0
    placed: dict[int, tuple[Meter, set[int]]] = {}
    tick_times: dict[tuple[int, int], float] = {}
    ordered = sorted(set(onsets_in_beats))
    for beat in ordered:
        location = grid.locate(beat)
        if location is None:
            raise ValueError("onset precedes the first known meter")
        bar, local, meter = location
        tick = round(local * subdivision)
        ticks = round(meter.numerator * 4 * subdivision / meter.denominator)
        if tick >= ticks:
            bar += 1
            tick = 0
        if bar in placed and placed[bar][0] != meter:
            raise ValueError("different meters occupy one bar")
        placed.setdefault(bar, (meter, set()))[1].add(tick)
        tick_times.setdefault((bar, tick), beat)
    score = 0
    bounds = {}
    for beat, following in zip(ordered, ordered[1:]):
        bounds[beat] = following
    for bar, (meter, ticks) in placed.items():
        weights, stronger, _ = _metric_pattern(meter, subdivision)
        for tick in ticks:
            here = tick_times[(bar, tick)]
            if here not in bounds or bounds[here] - here > meter.numerator * 4 / meter.denominator:
                continue
            ahead = stronger[tick]
            if ahead is not None and ahead not in ticks:
                score += weights[ahead] - weights[tick]
    max_per_bar = max(_metric_pattern(meter, subdivision)[2] for meter, _ in placed.values())
    return round(score / (max_per_bar * len(placed)), 6)
