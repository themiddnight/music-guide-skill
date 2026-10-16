"""Rhythm descriptors from the shared onset envelope and tracked beat grid."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal

from .. import rhythm
from .beats import BeatGrid, MeterEstimate, TempoCandidates, autocorrelation
from .common import Method, Outcome, Report, ReportContext, Skip
from .store import FeatureStore


METHOD = Method(
    "onset-grid-groove",
    "spectral-flux peaks above 5% of the upper-decile flux are placed against the tracked beat grid. "
    "Onset density counts distinct attacks, LHL syncopation scores attacks on weak metric positions "
    "before an empty stronger position, pulse clarity compares beat-period autocorrelation with nearby lags, "
    "and microtiming is the median absolute deviation from a sixteenth-note grid. Danceability is the "
    "stated weighted formula, not a listener rating. Mixed instruments, reverberation, missing attacks, "
    "half-time grids and an uncertain meter can change every descriptor."
)
FORMULA = "0.35*pulseClarity + 0.30*tempoStability + 0.20*densityFit + 0.15*timingRegularity"


@dataclass(frozen=True)
class Rhythm:
    onsets_per_second: float
    onsets_per_bar: float | None
    syncopation: float | None
    pulse_clarity: float
    microtiming_ms: float
    danceability: float
    method: Method = METHOD

    def report(self, ctx: ReportContext) -> Report:
        value: dict[str, Any] = {
            "onsetsPerSecond": round(self.onsets_per_second, 3),
            "onsetsPerBar": None if self.onsets_per_bar is None else round(self.onsets_per_bar, 3),
            "syncopation": self.syncopation,
            "pulseClarity": round(self.pulse_clarity, 3),
            "microtimingMs": round(self.microtiming_ms, 2),
            "danceability": round(self.danceability, 3),
            "danceabilityFormula": FORMULA,
        }
        return ctx.estimate("audio.rhythm", value, "rhythm", self.method)


def onset_times(store: FeatureStore) -> np.ndarray:
    flux = store.flux
    if not len(flux) or not np.any(flux):
        return np.empty(0)
    prominence = 0.05 * float(np.percentile(flux, 90))
    peaks, _ = signal.find_peaks(flux, prominence=max(prominence, 1e-7),
                                 distance=max(1, round(store.onset_rate * 0.08)))
    if len(flux) > 1 and flux[0] > flux[1] and flux[0] > max(prominence, 1e-7):
        peaks = np.r_[0, peaks]
    return peaks[peaks / store.onset_rate < store.seconds] / store.onset_rate


def _pulse_clarity(store: FeatureStore, grid: BeatGrid) -> float:
    period = int(round(store.onset_rate * 60 / grid.bpm))
    spread = max(2, int(period * 0.2))
    flux = store.flux - np.mean(store.flux)
    acf = autocorrelation(flux, period + spread + 1)
    if not acf[0] > 0:
        return 0.0
    peak = max(0.0, float(acf[period] / acf[0]))
    neighbors = np.r_[acf[period - spread:period - max(1, spread // 3)],
                      acf[period + max(1, spread // 3):period + spread + 1]]
    background = max(0.0, float(np.median(neighbors) / acf[0])) if len(neighbors) else 0.0
    return float(np.clip((peak - background) / max(0.1, 1 - background), 0, 1))


def groove(store: FeatureStore, beats: BeatGrid, tempo: TempoCandidates, *,
           meter: Outcome | MeterEstimate) -> Rhythm | Skip:
    """Compute onset measures only when a beat grid exists; meter-dependent values may be null."""
    del tempo  # The BeatGrid already encodes the selected tempo candidate.
    times = onset_times(store)
    if len(times) < 4:
        return Skip(f"only {len(times)} distinct onset peaks; rhythm needs at least four.")
    density = len(times) / store.seconds
    beat_times = beats.times
    end = beat_times[-1] + np.median(np.diff(beat_times))
    times = times[(times >= beat_times[0] - 0.03) & (times < end)]
    if len(times) < 4:
        return Skip("fewer than four onset peaks lie on the tracked beat span.")
    positions = np.interp(times, beat_times, np.arange(len(beat_times)))
    past_last = times > beat_times[-1]
    positions[past_last] = len(beat_times) - 1 + (times[past_last] - beat_times[-1]) / np.median(np.diff(beat_times))
    beat_period = float(np.median(np.diff(beat_times)))
    pulse_in_quarters = 1.0
    if isinstance(meter, MeterEstimate):
        _, denominator = map(int, meter.candidates[0][0].split("/"))
        pulse_in_quarters = 4 / denominator
    quarter_positions = positions * pulse_in_quarters
    deviations = np.abs(quarter_positions * 4 - np.round(quarter_positions * 4)) / 4
    microtiming = float(np.median(deviations) * beat_period / pulse_in_quarters * 1000)
    per_bar = sync = None
    if isinstance(meter, MeterEstimate):
        signature = meter.candidates[0][0]
        top, bottom = map(int, signature.split("/"))
        metric = rhythm.Meter(top, bottom)
        bar_positions = (positions - meter.downbeat_phase) * pulse_in_quarters
        per_bar = rhythm.onset_density(bar_positions.tolist(), top * 4 / bottom)
        sync = rhythm.syncopation(bar_positions.tolist(), metric)
    clarity = _pulse_clarity(store, beats)
    density_fit = float(np.clip(density / 2, 0, 1) * np.clip(10 / max(density, 1e-9), 0, 1))
    timing_regularity = float(np.clip(1 - microtiming / (beat_period / pulse_in_quarters * 125), 0, 1))
    danceability = float(np.clip(0.35 * clarity + 0.30 * beats.stability
                                 + 0.20 * density_fit + 0.15 * timing_regularity, 0, 1))
    return Rhythm(density, per_bar, sync, clarity, microtiming, danceability)
