"""Tempo candidates, beat grid, meter with downbeat phase, and swing, all from the onset envelope."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import signal

from .. import rhythm
from .common import FULL_LIMIT, Method, Report, ReportContext, Skip
from .store import FeatureStore

TEMPO_RANGE = (60.0, 200.0)
# A peak at a range edge interpolates up to about 1% past it, so the acceptance test allows this share beyond.
TEMPO_EDGE = 0.02
MIN_TEMPO_SECONDS = 8.0
# Autocorrelation is summed per segment, so a 6 hour envelope never needs one whole-file FFT.
ACF_SEGMENT = 2 ** 16
# Ellis (2007) picks among octave-related tempi with a log-normal prior around 120 BPM.
PRIOR_BPM, PRIOR_OCTAVES = 120.0, 1.0
# The prior only chooses an octave: a candidate competes when it is within this share of half, once or twice the
# strongest peak. A peak at 4/3 or 3/2 of the beat, from sixteenths or a three-against-two figure, never wins.
OCTAVE_TOLERANCE = 0.06
TIGHTNESS = 100.0
# Normalized onset autocorrelation at the beat period: white noise and a slow pad read at most 0.04, the three
# reference songs 0.51 to 0.63.
MIN_PULSE = 0.1
MIN_BEATS = 8
# A tracked beat lands on an onset when its envelope peak reaches this share of the median beat's.
ON_ONSET_SHARE = 0.25
# Beat intervals within this share of the median count as steady.
STEADY_SHARE = 0.1
MIN_METER_BEATS = 24
# A pitch-class row's window spans about 0.37 s each side of its centre, so the rows this far before and after a
# beat show a chord change at that beat without smearing it onto the neighbouring beats.
CHANGE_OFFSET = 0.4
METERS = ((2, "2/4"), (3, "3/4"), (4, "4/4"), (6, "6/8"))
# Share of beat-accent variance a bar must explain beyond chance. Unaccented and randomly jittered click tracks
# reach at most 0.08; the three reference songs score 0.17 (itipiso) to 0.77 (Winterstory).
MIN_METER_SCORE = 0.12
MIN_SWING_PAIRS = 16
# An off-beat peak counts toward swing only when it reaches this share of the onset on the beat before it.
OFFBEAT_SHARE = 0.2
# The mean onset envelope across beats, resampled to this many points per beat, shows the sub-beat pattern. Swing
# is read only when one peak inside SWING_SEARCH stands out: a second peak at this share of the tallest means
# sixteenths or triplets compete with the eighth. Log flux is compressive: on the test signals, sixteenths at 3/8
# of the eighth's gain still peak at 0.67 of it, and an eighth at 4/7 of a louder third sixteenth peaks at 0.81.
SUB_BEAT_BINS = 48
DISTINCT_PEAK_SHARE = 0.75
SWING_SEARCH = (0.2, 0.85)

TEMPO_METHOD = Method("onset-autocorrelation",
                      "top autocorrelation peaks of a spectral-flux onset envelope within 60-200 BPM, using FFT"
                      " magnitudes averaged across channels before computing flux. Half- and double-time readings are"
                      " equally consistent with periodicity alone; rubato, swing and tempo changes are averaged away.")


@dataclass(frozen=True)
class TempoCandidates:
    candidates: tuple[tuple[float, float], ...]  # (bpm, strength relative to the best), best first, at most 3
    octave_related: bool
    method: Method = TEMPO_METHOD

    def report(self, ctx: ReportContext) -> Report:
        value = {"candidates": [{"bpm": bpm, "relativeStrength": strength} for bpm, strength in self.candidates],
                 "octaveRelatedCandidates": self.octave_related}
        return ctx.estimate("audio.tempo_estimate", value, "BPM", self.method)


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
        if TEMPO_RANGE[0] * (1 - TEMPO_EDGE) <= bpm <= TEMPO_RANGE[1] * (1 + TEMPO_EDGE) and here > 0:
            found.append((float(here), round(bpm, 1)))
    if not found:
        return Skip("the onset autocorrelation has no peak within 60-200 BPM.")
    found.sort(reverse=True)
    best = found[0][0]
    top = tuple((bpm, round(strength / best, 3)) for strength, bpm in found[:3])
    octave = any(abs(a / b - 2) < 0.06 for a, _ in top for b, _ in top)
    return TempoCandidates(top, octave)


BEAT_METHOD = Method("ellis-dp-beats",
                     "beats tracked by dynamic programming (Ellis 2007) over a three-band spectral-flux onset envelope"
                     " toward one global tempo, chosen among the tempo candidates by a log-normal prior around 120 BPM."
                     " Half- and double-time grids fit the onsets equally well; tempo changes and rubato bend the grid"
                     " only as far as the tightness penalty allows. tempoStability is low when the beat intervals vary"
                     " or when beats miss the onsets, as they do through a tempo change the grid cannot follow.")
METER_METHOD = Method("accent-pattern-meter",
                      "bars of 2, 3, 4 or 6 tracked beats scored by the share of per-beat accent variance (low-band"
                      " onset strength and pitch-class change) that the bar's per-position means explain beyond"
                      " chance; the downbeat is the position with the strongest mean accent. 2/4 and 4/4, and 3/4"
                      " and 6/8, differ only by a weaker accent mid-bar, so they are often close; a pickup or an"
                      " intro without accents can shift the downbeat.")
SWING_METHOD = Method("offbeat-onset-position",
                      "median long-to-short ratio of the strongest onset between consecutive beats, counted only at"
                      " 40-80% of the beat. It skips when the mean onset pattern between beats has two similar peaks,"
                      " such as straight or swung sixteenths, because the subdivision is then ambiguous. A shuffle"
                      " felt against a half-time grid and loose timing are not told apart.")


@dataclass(frozen=True)
class BeatGrid:
    times: np.ndarray  # seconds, at least MIN_BEATS, finite and strictly increasing
    bpm: float  # 60 over the median beat interval
    # 0..1, the lowest of three: 1 minus the interquartile range of beat intervals over their median, the share of
    # intervals within STEADY_SHARE of the median, and the share of beats on an onset. A tempo change the grid
    # cannot follow leaves its intervals even but its beats off the onsets.
    stability: float
    method: Method = BEAT_METHOD

    def report(self, ctx: ReportContext) -> Report:
        value = {"bpm": round(self.bpm, 1), "beatCount": len(self.times),
                 "firstBeatSeconds": round(float(self.times[0]), 3), "tempoStability": round(self.stability, 3)}
        limit = None
        if ctx.full:
            value["beatTimes"], limit = ctx.cap(np.round(self.times, 3).tolist(), FULL_LIMIT, "beat times")
        return ctx.estimate("audio.beat_grid", value, "BPM", self.method, [limit])


@dataclass(frozen=True)
class MeterEstimate:
    candidates: tuple[tuple[str, float], ...]  # (signature, score), best first
    downbeat_phase: int  # index into BeatGrid.times of the first downbeat
    downbeats: np.ndarray  # seconds
    method: Method = METER_METHOD

    def report(self, ctx: ReportContext) -> Report:
        value = {"candidates": [{"signature": signature, "score": score} for signature, score in self.candidates],
                 "downbeatPhase": self.downbeat_phase, "barCount": len(self.downbeats)}
        limit = None
        if ctx.full:
            value["downbeatTimes"], limit = ctx.cap(np.round(self.downbeats, 3).tolist(), FULL_LIMIT, "downbeat times")
        return ctx.estimate("audio.meter_estimate", value, "meter", self.method, [limit])


@dataclass(frozen=True)
class Swing:
    ratio: float  # long to short, 1.0 straight, 2.0 triplet swing
    pairs: int  # beat intervals with an off-beat onset inside rhythm.SWING_WINDOW
    method: Method = SWING_METHOD

    def report(self, ctx: ReportContext) -> Report:
        value = {"ratio": round(self.ratio, 3), "pairs": self.pairs}
        return ctx.estimate("audio.swing_ratio", value, "ratio", self.method)


def _prior(bpm: float) -> float:
    return math.exp(-0.5 * (math.log2(bpm / PRIOR_BPM) / PRIOR_OCTAVES) ** 2)


def beat_grid(store: FeatureStore, tempo: TempoCandidates) -> BeatGrid | Skip:
    strongest = tempo.candidates[0][0]
    octaves = [candidate for candidate in tempo.candidates
               if any(abs(candidate[0] / (strongest * ratio) - 1) < OCTAVE_TOLERANCE for ratio in (0.5, 1, 2))]
    bpm = max(octaves, key=lambda candidate: candidate[1] * _prior(candidate[0]))[0]
    period = 60 * store.onset_rate / bpm
    envelope = store.flux
    acf = autocorrelation(envelope - np.mean(envelope), int(2 * period) + 2)
    lag = int(round(period))
    pulse = float(np.max(acf[lag - 1:lag + 2]) / acf[0])
    if pulse < MIN_PULSE:
        return Skip(f"no steady pulse: the onset autocorrelation at the {bpm:g} BPM period is {pulse:.2f}, below"
                    f" {MIN_PULSE:g}.")
    scaled = envelope / np.std(envelope)
    frames = _track(scaled, period)
    strength = np.array([scaled[max(0, frame - 1):frame + 2].max() for frame in frames])
    on_onset = strength >= ON_ONSET_SHARE * np.median(strength)
    strong = np.flatnonzero(on_onset)
    frames, on_onset = frames[strong[0]:strong[-1] + 1], on_onset[strong[0]:strong[-1] + 1]
    if len(frames) < MIN_BEATS:
        return Skip(f"only {len(frames)} beats were tracked; a beat grid needs {MIN_BEATS}.")
    times = (frames + _peak_offsets(scaled, frames)) / store.onset_rate
    intervals = np.diff(times)
    middle = float(np.median(intervals))
    spread = float(np.subtract(*np.percentile(intervals, [75, 25])))
    steady = float(np.mean(np.abs(intervals / middle - 1) <= STEADY_SHARE))
    stability = min(1 - spread / middle, steady, float(np.mean(on_onset)))
    return BeatGrid(times, 60 / middle, float(np.clip(stability, 0, 1)))


def _peak_offsets(envelope: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Sub-frame position of each frame that is a local envelope peak, by parabolic interpolation; 0 elsewhere."""
    inside = np.clip(frames, 1, len(envelope) - 2)
    before, here, after = envelope[inside - 1], envelope[inside], envelope[inside + 1]
    curvature = before - 2 * here + after
    peak = (inside == frames) & (here >= before) & (here >= after) & (curvature < 0)
    return np.where(peak, 0.5 * (before - after) / np.where(peak, curvature, -1.0), 0.0)


def _track(envelope: np.ndarray, period: float) -> np.ndarray:
    """Ellis (2007) beat frames: each frame adds the best predecessor 0.5 to 2 periods back, less a log-ratio penalty.

    A frame whose best predecessor total is not positive starts a new path, so the first beat is not dropped for a
    penalized lag back to the start of the file. Frames are scored in blocks no longer than the shortest lag, so no
    frame in a block depends on another in it.
    """
    lags = np.arange(max(1, math.ceil(period / 2)), math.floor(2 * period) + 1)
    penalty = -TIGHTNESS * np.log(lags / period) ** 2
    score = envelope.copy()
    back = np.full(len(envelope), -1)
    for start in range(int(lags[0]), len(envelope), int(lags[0])):
        frames = np.arange(start, min(len(envelope), start + int(lags[0])))
        previous = frames[:, None] - lags[None, :]
        candidate = np.where(previous >= 0, score[np.maximum(previous, 0)] + penalty, -np.inf)
        best = np.argmax(candidate, axis=1)
        rows = np.arange(len(frames))
        gain = candidate[rows, best]
        chained = gain > 0
        score[frames] += np.where(chained, gain, 0.0)
        back[frames] = np.where(chained, previous[rows, best], -1)
    tail = len(envelope) - int(round(period))
    beat = int(max(0, tail) + np.argmax(score[max(0, tail):]))
    path = []
    while beat >= 0:
        path.append(beat)
        beat = int(back[beat])
    return np.array(path[::-1])


def _unit(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return np.divide(rows, norms, out=np.zeros(rows.shape), where=norms > 0)


def _contrast(values: np.ndarray) -> np.ndarray:
    mean = float(np.mean(values))
    return values / mean - 1 if mean > 0 else np.zeros_like(values)


def meter(store: FeatureStore, beats: BeatGrid) -> MeterEstimate | Skip:
    times = beats.times
    if len(times) < MIN_METER_BEATS:
        return Skip(f"only {len(times)} beats; meter needs {MIN_METER_BEATS}, four bars of the longest candidate.")
    frames = np.round(times * store.onset_rate).astype(int)
    low = np.array([store.onset[max(0, frame - 2):frame + 3, 0].max(initial=0.0) for frame in frames])
    rows = len(store.pcp)
    before = np.clip(np.round((times - CHANGE_OFFSET) * store.row_rate).astype(int), 0, rows - 1)
    after = np.clip(np.round((times + CHANGE_OFFSET) * store.row_rate).astype(int), 0, rows - 1)
    change = 1 - np.sum(_unit(store.pcp[before]) * _unit(store.pcp[after]), axis=1)
    # Low-band strength is relative to its mean because its scale depends on the mix; cosine distance already has
    # an absolute scale, so a near-constant chroma (a click track) adds near zero instead of amplified noise.
    accent = _contrast(low) + change - np.mean(change)
    centred = accent - np.mean(accent)
    total = float(np.sum(centred ** 2))
    if not total > 0:
        return Skip("the tracked beats carry no low-band or pitch-class accent, so no bar pattern can be read.")
    scored = []
    for size, signature in METERS:
        position = np.arange(len(accent)) % size
        counts = np.bincount(position, minlength=size)
        means = np.bincount(position, weights=centred, minlength=size) / counts
        explained = float(np.sum(counts * means ** 2)) / total
        # Under a random order of the same accents a bar of `size` positions explains (size - 1) / (beats - 1) of the
        # variance on average, so the raw share would favour longer bars.
        scored.append((round(explained - (size - 1) / (len(accent) - 1), 3), signature, int(np.argmax(means)), size))
    scored.sort(key=lambda item: -item[0])
    score, signature, phase, size = scored[0]
    if score < MIN_METER_SCORE:
        return Skip(f"no bar pattern stands out: the best, {signature}, explains {score:.2f} of the beat-accent"
                    f" variance beyond chance, below {MIN_METER_SCORE:g}.")
    return MeterEstimate(tuple((signature, score) for score, signature, _, _ in scored), phase, times[phase::size])


def _sub_beat_peaks(flux: np.ndarray, rate: float, times: np.ndarray) -> list[float]:
    """Beat fractions, in order, of the distinct peaks of the mean onset envelope inside SWING_SEARCH."""
    fractions = np.arange(SUB_BEAT_BINS) / SUB_BEAT_BINS
    positions = (times[:-1, None] + fractions[None, :] * np.diff(times)[:, None]) * rate
    profile = np.interp(positions, np.arange(len(flux)), flux).mean(axis=0)
    low, high = math.ceil(SWING_SEARCH[0] * SUB_BEAT_BINS), math.floor(SWING_SEARCH[1] * SUB_BEAT_BINS) + 1
    peaks = low + signal.find_peaks(profile[low:high])[0]
    if not len(peaks):
        return []
    distinct = peaks[profile[peaks] >= DISTINCT_PEAK_SHARE * profile[peaks].max()]
    return [float(fractions[peak]) for peak in distinct]


def swing(store: FeatureStore, beats: BeatGrid) -> Swing | Skip:
    flux, rate = store.flux, store.onset_rate
    peaks = _sub_beat_peaks(flux, rate, beats.times)
    if len(peaks) > 1:
        where = ", ".join(f"{peak:.2f}" for peak in peaks)
        return Skip(f"the onsets between beats peak at {where} of the beat with similar strength, so the subdivision"
                    " is ambiguous and a long-to-short ratio would depend on which peak is read.")
    positions = []
    for index, (here, after) in enumerate(zip(beats.times[:-1], beats.times[1:])):
        low, high = (math.ceil((here + SWING_SEARCH[0] * (after - here)) * rate),
                     math.floor((here + SWING_SEARCH[1] * (after - here)) * rate))
        if high <= low:
            continue
        peak = low + int(np.argmax(flux[low:high + 1]))
        on_beat = flux[max(0, round(here * rate) - 1):round(here * rate) + 2].max(initial=0.0)
        if not flux[peak] > OFFBEAT_SHARE * on_beat or flux[peak] <= 0:
            continue
        before, after_peak = flux[peak - 1], flux[min(peak + 1, len(flux) - 1)]
        curvature = before - 2 * flux[peak] + after_peak
        offset = 0.5 * (before - after_peak) / curvature if curvature < 0 else 0.0
        positions.append(index + ((peak + offset) / rate - here) / (after - here))
    pairs = sum(rhythm.SWING_WINDOW[0] < position % 1 < rhythm.SWING_WINDOW[1] for position in positions)
    ratio = rhythm.swing_ratio(positions)
    if ratio is None or pairs < MIN_SWING_PAIRS:
        return Skip(f"only {pairs} beats have a distinct off-beat onset; swing needs {MIN_SWING_PAIRS}.")
    return Swing(ratio, pairs)
