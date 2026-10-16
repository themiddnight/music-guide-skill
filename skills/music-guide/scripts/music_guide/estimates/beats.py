"""Tempo candidates from onset autocorrelation."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import signal

from ..model import Fact
from .common import Method, ReportContext, Skip
from .store import FeatureStore

TEMPO_RANGE = (60.0, 200.0)
MIN_TEMPO_SECONDS = 8.0
# Autocorrelation is summed per segment, so a 6 hour envelope never needs one whole-file FFT.
ACF_SEGMENT = 2 ** 16

TEMPO_METHOD = Method("onset-autocorrelation",
                      "top autocorrelation peaks of a spectral-flux onset envelope within 60-200 BPM, using FFT"
                      " magnitudes averaged across channels before computing flux. Half- and double-time readings are"
                      " equally consistent with periodicity alone; rubato, swing and tempo changes are averaged away.")


@dataclass(frozen=True)
class TempoCandidates:
    candidates: tuple[tuple[float, float], ...]  # (bpm, strength relative to the best), best first, at most 3
    octave_related: bool
    method: Method = TEMPO_METHOD

    def report(self, ctx: ReportContext) -> list[Fact]:
        value = {"candidates": [{"bpm": bpm, "relativeStrength": strength} for bpm, strength in self.candidates],
                 "octaveRelatedCandidates": self.octave_related}
        return [ctx.estimate("audio.tempo_estimate", value, "BPM", self.method)]


def autocorrelation(envelope: np.ndarray, lags: int) -> np.ndarray:
    """Exact autocorrelation for lags 0..lags-1, summed over segments of ACF_SEGMENT frames."""
    total = np.zeros(lags)
    for start in range(0, len(envelope), ACF_SEGMENT):
        segment = envelope[start:start + ACF_SEGMENT]
        extended = envelope[start:start + ACF_SEGMENT + lags]
        size = 1 << int(len(segment) + len(extended) - 1).bit_length()
        spectrum = np.conj(np.fft.rfft(segment, size)) * np.fft.rfft(extended, size)
        total += np.fft.irfft(spectrum, size)[:lags]
    return total


def tempo(store: FeatureStore) -> TempoCandidates | Skip:
    if store.seconds < MIN_TEMPO_SECONDS:
        return Skip(f"{store.seconds:.1f} s of audio is shorter than the {MIN_TEMPO_SECONDS:g} s a periodicity"
                    " reading needs.")
    envelope = store.flux
    frame_rate = store.onset_rate
    shortest = int(60 * frame_rate / TEMPO_RANGE[1]) - 1
    longest = min(len(envelope) - 2, int(math.ceil(60 * frame_rate / TEMPO_RANGE[0])) + 1)
    acf = autocorrelation(envelope - np.mean(envelope), longest + 2)
    if not acf[0] > 0:
        return Skip("the onset envelope is flat (silence or one sustained sound), so it has no periodicity.")
    acf /= acf[0]
    peaks = [lag for lag in signal.find_peaks(acf)[0] if shortest <= lag <= longest]
    found = []
    for lag in peaks:
        before, here, after = acf[lag - 1], acf[lag], acf[lag + 1]
        curvature = before - 2 * here + after
        offset = 0.5 * (before - after) / curvature if curvature < 0 else 0.0
        bpm = 60 * frame_rate / (lag + offset)
        if TEMPO_RANGE[0] <= bpm <= TEMPO_RANGE[1] and here > 0:
            found.append((float(here), round(bpm, 1)))
    if not found:
        return Skip("the onset autocorrelation has no peak within 60-200 BPM.")
    found.sort(reverse=True)
    best = found[0][0]
    top = tuple((bpm, round(strength / best, 3)) for strength, bpm in found[:3])
    octave = any(abs(a / b - 2) < 0.06 for a, _ in top for b, _ in top)
    return TempoCandidates(top, octave)
