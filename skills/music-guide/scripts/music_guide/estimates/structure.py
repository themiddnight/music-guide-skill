"""Bounded local self-similarity and neutral letter labels for musical sections."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .common import FULL_LIMIT, Method, Report, ReportContext, Skip
from .store import FeatureStore, bin_sum

SUMMARY_SEGMENTS = 64
MAX_SEGMENTS = 500
MIN_SECONDS = 45.0
# A side covers eight 4/4 bars on the beat grid or 16 s on the fixed grid, comparing loops rather than chords.
KERNEL_RADIUS = 32


@dataclass(frozen=True)
class Structure:
    boundaries: np.ndarray
    letters: tuple[str, ...]
    novelty: np.ndarray
    basis: str
    novelty_times: np.ndarray | None = None
    cuts: tuple[str, ...] = ()

    def report(self, ctx: ReportContext) -> Report:
        segments = [{"letter": letter, "startSeconds": round(float(start), 3),
                     "endSeconds": round(float(end), 3)}
                    for letter, start, end in zip(self.letters, self.boundaries[:-1], self.boundaries[1:])]
        segments, cut = ctx.cap(segments, FULL_LIMIT if ctx.full else SUMMARY_SEGMENTS, "structure segments")
        value = {"basis": self.basis, "segments": segments, "distinctLetters": len(set(self.letters))}
        cuts = [*self.cuts, cut]
        if ctx.full:
            times = self.novelty_times if self.novelty_times is not None else np.linspace(
                self.boundaries[0], self.boundaries[-1], len(self.novelty), endpoint=False)
            novelty = [{"startSeconds": round(float(time), 3), "strength": round(float(strength), 4)}
                       for time, strength in zip(times, self.novelty)]
            value["novelty"], cut = ctx.cap(novelty, FULL_LIMIT, "novelty points")
            cuts.append(cut)
        method = Method("local-self-similarity-novelty",
                        "Foote-style checkerboard novelty of standardized chroma and log-band timbre on a "
                        f"{self.basis} grid, with at most {KERNEL_RADIUS} units each side of each boundary. "
                        "Gaussian-tapered kernel sums use filtered feature vectors instead of a quadratic similarity matrix. "
                        "Peaks are snapped to inferred downbeats when available. Segment mean cosine similarity "
                        "assigns letters, which describe acoustic resemblance and cannot identify verse or chorus. "
                        "Gradual changes, repeated adjacent sections and changes in orchestration may be missed.")
        return ctx.estimate("audio.structure", value, "sections", method, cuts)


def unit_edges(seconds: float, beats) -> tuple[np.ndarray, str]:
    if isinstance(beats, Skip) or beats is None:
        return np.unique(np.r_[np.arange(0, seconds, 0.5), seconds]), "fixed"
    times = np.asarray(beats.times)
    return np.unique(np.r_[0.0, times[(times > 0) & (times < seconds)], seconds]), "beats"


def _normalize(rows: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(rows, axis=1, keepdims=True)
    return np.divide(rows, norm, out=np.zeros_like(rows, dtype=float), where=norm > 1e-12)


def _label(index: int) -> str:
    result = ""
    while True:
        result = chr(65 + index % 26) + result
        index = index // 26 - 1
        if index < 0:
            return result


def structure(store: FeatureStore, *, beats, meter) -> Structure | Skip:
    if store.seconds < MIN_SECONDS:
        return Skip(f"structure needs at least {MIN_SECONDS:g} s; this file has {store.seconds:.1f} s.")
    chroma = getattr(store, "treble", None)
    if chroma is None:
        chroma = store.pcp
    if not np.any(chroma > 0):
        return Skip("structure has no pitched spectral energy to compare.")
    edges, basis = unit_edges(store.seconds, beats)
    duration = np.diff(edges)
    chroma = bin_sum(chroma, store.row_rate, edges)
    totals = np.sum(chroma, axis=1, keepdims=True)
    chroma = np.divide(chroma, totals, out=np.zeros_like(chroma), where=totals > 0)
    timbre = getattr(store, "timbre", None)
    if timbre is None:
        timbre = np.zeros((len(store.pcp), 12))
    timbre = bin_sum(timbre, store.row_rate, edges) / np.maximum(duration[:, None] * store.row_rate, 1)
    features = np.concatenate([chroma, timbre], axis=1)
    spread = np.std(features, axis=0)
    features = _normalize(np.divide(features - np.mean(features, axis=0), spread,
                                    out=np.zeros_like(features), where=spread > 1e-8))
    radius = min(KERNEL_RADIUS, 64, max(1, len(features) // 4))
    weights = np.exp(-0.5 * ((np.arange(radius) + 0.5) / (radius * 0.5)) ** 2)
    weights /= weights.sum()
    trailing = signal.lfilter(weights, [1.0], features, axis=0)
    leading = signal.lfilter(weights, [1.0], features[::-1], axis=0)[::-1]
    positions = np.arange(radius, len(features) - radius + 1)
    novelty = np.zeros(len(features))
    novelty[positions] = np.maximum(0, np.sum((trailing[positions - 1] - leading[positions]) ** 2, axis=1) / 4)
    minimum = max(radius, int(round(8 / np.median(duration))))
    peaks = signal.find_peaks(novelty, height=max(0.08, float(novelty.max()) * 0.25),
                              prominence=0.03, distance=minimum)[0]
    downbeats = None if isinstance(meter, Skip) or meter is None else np.asarray(meter.downbeats)
    candidates = []
    for index in peaks:
        time = float(edges[index])
        if downbeats is not None and len(downbeats):
            snapped = float(downbeats[np.argmin(np.abs(downbeats - time))])
            if abs(snapped - time) <= 4:
                time = snapped
        if time >= 8 and store.seconds - time >= 8:
            candidates.append((float(novelty[index]), time))
    chosen = []
    for _, time in sorted(candidates, reverse=True):
        if all(abs(time - other) >= 8 for other in chosen):
            chosen.append(time)
    cuts = ()
    if len(chosen) >= MAX_SEGMENTS:
        cuts = (f"Only the strongest {MAX_SEGMENTS - 1} of {len(chosen)} structure boundaries were retained.",)
        chosen = sorted(chosen, key=lambda t: novelty[np.argmin(np.abs(edges[:-1] - t))], reverse=True)[:MAX_SEGMENTS - 1]
    boundaries = np.array([0.0, *sorted(chosen), store.seconds])
    centroids, counts, letters = [], [], []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        inside = (edges[:-1] >= start) & (edges[:-1] < end)
        mean = _normalize(np.mean(features[inside], axis=0, keepdims=True))[0]
        similarity = np.asarray(centroids) @ mean if centroids else np.array([])
        if similarity.size and float(similarity.max()) >= 0.5:
            label = int(np.argmax(similarity))
            centroids[label] = _normalize((centroids[label] * counts[label] + mean)[None, :])[0]
            counts[label] += 1
        else:
            label = len(centroids)
            centroids.append(mean)
            counts.append(1)
        letters.append(_label(label))
    keep = [i for i, letter in enumerate(letters) if i == 0 or letter != letters[i - 1]]
    boundaries = np.r_[boundaries[keep], boundaries[-1]]
    names = {}
    letters = tuple(names.setdefault(letters[i], _label(len(names))) for i in keep)
    return Structure(boundaries, letters, novelty, basis, edges[:-1], cuts)
