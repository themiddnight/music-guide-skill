"""Pitch-class profile and the Krumhansl-Kessler key estimate and key timeline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model import Fact
from .common import Method, ReportContext, Skip
from .store import FeatureStore, bin_sum

PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
# Krumhansl and Kessler (1982) probe-tone ratings, tonic first.
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
KEYS = tuple((tonic, mode, np.roll(profile, index)) for mode, profile in (("major", KK_MAJOR), ("minor", KK_MINOR))
             for index, tonic in enumerate(PITCH_CLASSES))
MIN_KEY_SECONDS = 2.0
# Keys are scored per 5 s chroma cell. A key change costs KEY_SWITCH_PENALTY in summed
# correlation, so a new key must outscore the old one over about 25 to 50 s, not for one chord.
# On the 100 ms row grid, penalties from 0.65 to 4.75 match the three listener-labelled reference songs and the
# synthetic key tests; 0.6 splits a 15 s parallel-major passage and 5.0 misses a 30 s distant modulation.
KEY_CELL_SECONDS = 5.0
KEY_SWITCH_PENALTY = 2.0
MIN_KEY_TIMELINE_SECONDS = 30.0

PROFILE_REASON = ("Share of 65-2100 Hz spectral magnitude per pitch class (A4=440 Hz, about 11 kHz per channel)."
                  " FFT magnitudes are averaged across channels to avoid polarity cancellation."
                  " Overtones fold into other classes, so this is spectral energy, not played notes.")
KEY_METHOD = Method("krumhansl-kessler-chroma",
                    "mean correlation of each 5 s chroma cell with Krumhansl-Kessler major and minor profiles, so a"
                    " long passage outweighs a loud one. Modal, non-diatonic or pentatonic music (for example"
                    " Hirajoshi), key changes and detuning can be mislabelled; read audio.pitch_class_profile"
                    " directly, and audio.key_timeline for key changes.")
TIMELINE_METHOD = Method("krumhansl-kessler-chroma-windows",
                         "a Viterbi path over 5 s chroma cells scored against Krumhansl-Kessler profiles, where"
                         f" changing key costs {KEY_SWITCH_PENALTY:g} in summed correlation. A new segment needs"
                         " sustained evidence: a distant key shows within about 20 s, a closely related key (for"
                         " example the dominant) can need over a minute, and borrowed chords or short tonicizations"
                         " stay in the surrounding key. Boundaries are accurate to about one cell. The penalty was"
                         " calibrated on three listener-labelled songs only. The same modal and pentatonic caveats as"
                         " audio.key_estimate apply.")


@dataclass(frozen=True)
class PitchProfile:
    shares: np.ndarray  # (12,) summing to 1, C first

    def report(self, ctx: ReportContext) -> list[Fact]:
        value = {name: round(float(share), 4) for name, share in zip(PITCH_CLASSES, self.shares)}
        return [ctx.measurement("audio.pitch_class_profile", value, "ratio", "chroma-fft", PROFILE_REASON, "medium")]


@dataclass(frozen=True)
class KeySegment:
    start: float
    end: float
    scores: np.ndarray  # (24,) mean correlation per key over the segment's usable cells


@dataclass(frozen=True)
class KeyResult:
    scores: np.ndarray  # (24,) mean correlation per key over every usable cell, in KEYS order
    timeline: tuple[KeySegment, ...] | None  # None below MIN_KEY_TIMELINE_SECONDS
    method: Method = KEY_METHOD
    timeline_method: Method = TIMELINE_METHOD

    def report(self, ctx: ReportContext) -> list[Fact]:
        facts = [ctx.estimate("audio.key_estimate", _candidates(self.scores), "key", self.method)]
        if self.timeline is not None:
            value = {"cellSeconds": KEY_CELL_SECONDS,
                     "segments": [{"startSeconds": segment.start, "endSeconds": segment.end,
                                   **_candidates(segment.scores)} for segment in self.timeline]}
            facts.append(ctx.estimate("audio.key_timeline", value, "key", self.timeline_method))
        return facts


def _pitch_skip(store: FeatureStore) -> Skip | None:
    if store.seconds < MIN_KEY_SECONDS:
        return Skip(f"{store.seconds:.1f} s of audio is shorter than the {MIN_KEY_SECONDS:g} s a pitch-class"
                    " reading needs.")
    if not float(np.sum(store.pcp, dtype=np.float64)) > 0:
        return Skip("there is no spectral energy between 65 and 2100 Hz.")
    return None


def pitch_profile(store: FeatureStore) -> PitchProfile | Skip:
    skip = _pitch_skip(store)
    if skip is not None:
        return skip
    total = np.sum(store.pcp, axis=0, dtype=np.float64)
    return PitchProfile(total / total.sum())


def key_scores(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Correlation of each cell's chroma with the 24 Krumhansl-Kessler profiles, and which cells have pitch spread.

    Cells without spread (silence) score 0 for every key, so they never favour a change.
    """
    usable = np.std(cells, axis=1) > 0
    if not np.any(usable):
        return None
    centered = cells[usable] - cells[usable].mean(axis=1, keepdims=True)
    profiles = np.array([profile - profile.mean() for _, _, profile in KEYS])
    scores = np.zeros((len(cells), len(KEYS)))
    scores[usable] = (centered @ profiles.T) / np.outer(np.linalg.norm(centered, axis=1),
                                                        np.linalg.norm(profiles, axis=1))
    return scores, usable


def key(store: FeatureStore) -> KeyResult | Skip:
    skip = _pitch_skip(store)
    if skip is not None:
        return skip
    rows_per_cell = KEY_CELL_SECONDS * store.row_rate
    edges = np.arange(int(np.ceil(len(store.pcp) / rows_per_cell)) + 1) * KEY_CELL_SECONDS
    scored = key_scores(bin_sum(store.pcp, store.row_rate, edges))
    if scored is None:
        return Skip("no 5 s chroma cell has any spread across pitch classes.")
    scores, usable = scored
    timeline = None
    if store.seconds >= MIN_KEY_TIMELINE_SECONDS:
        timeline = tuple(KeySegment(round(start * KEY_CELL_SECONDS, 1),
                                    round(store.seconds if last else end * KEY_CELL_SECONDS, 1),
                                    np.mean(scores[start:end][usable[start:end]], axis=0))
                         for start, end, last in _viterbi_runs(scores))
    return KeyResult(np.mean(scores[usable], axis=0), timeline)


def _viterbi_runs(scores: np.ndarray) -> list[tuple[int, int, bool]]:
    """Cell runs of one key on the Viterbi path: stay in a key for free, switch for KEY_SWITCH_PENALTY."""
    total, back = scores[0].copy(), []
    for row in scores[1:]:
        switch = total.max() - KEY_SWITCH_PENALTY
        back.append(np.where(total >= switch, np.arange(len(KEYS)), int(total.argmax())))
        total = np.maximum(total, switch) + row
    path = [int(total.argmax())]
    for pointers in reversed(back):
        path.append(int(pointers[path[-1]]))
    path.reverse()
    runs: list[list[int]] = []
    for index, chosen in enumerate(path):
        if runs and runs[-1][2] == chosen:
            runs[-1][1] = index + 1
        else:
            runs.append([index, index + 1, chosen])
    return [(start, end, index == len(runs) - 1) for index, (start, end, _) in enumerate(runs)]


def _candidates(scores: np.ndarray) -> dict:
    order = np.argsort(-scores, kind="stable")[:3]
    return {"candidates": [{"tonic": KEYS[i][0], "mode": KEYS[i][1], "correlation": round(float(scores[i]), 3)}
                           for i in order]}
