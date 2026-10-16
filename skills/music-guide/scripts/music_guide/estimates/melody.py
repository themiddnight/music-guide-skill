"""Predominant harmonic pitch with an explicit unvoiced Viterbi state."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import FULL_LIMIT, Method, Report, ReportContext, Skip
from .store import FeatureStore
from .structure import unit_edges

MAX_SECONDS = 1800.0
SUMMARY_POINTS = 256
LOW_MIDI, HIGH_MIDI = 40, 96
ACTIVATION_LOW_MIDI = 33


@dataclass(frozen=True)
class Melody:
    pitches: np.ndarray  # MIDI or NaN for unvoiced frames, internal only
    row_rate: float
    contour: tuple[dict, ...]
    covered_seconds: float
    basis: str
    cuts: tuple[str, ...] = ()
    role: str = "mixdown"

    def report(self, ctx: ReportContext) -> Report:
        voiced = self.pitches[np.isfinite(self.pitches)]
        contour, cut = ctx.cap(self.contour, FULL_LIMIT if ctx.full else SUMMARY_POINTS, "melody contour points")
        value = {"lowMidi": float(np.min(voiced)) if len(voiced) else None,
                 "highMidi": float(np.max(voiced)) if len(voiced) else None,
                 "medianMidi": float(np.median(voiced)) if len(voiced) else None,
                 "voicedShare": round(float(len(voiced) / len(self.pitches)), 4),
                 "basis": self.basis, "contour": contour, "coveredSeconds": round(self.covered_seconds, 3)}
        cuts = [*self.cuts, cut]
        if ctx.full:
            frames = [{"startSeconds": round(i / self.row_rate, 3),
                       "midi": float(pitch) if np.isfinite(pitch) else None}
                      for i, pitch in enumerate(self.pitches)]
            value["frames"], cut = ctx.cap(frames, FULL_LIMIT, "melody frames")
            cuts.append(cut)
        reason = ("Viterbi path over tuning-corrected harmonic note salience from E2 to C7 with an unvoiced state, "
                  "subtracting a small pitch-class accompaniment baseline and penalizing pitch jumps. "
                  "Voicing depends on salience level and concentration, not vocal detection. "
                  "Predominant pitch in a polyphonic mix can follow bass, chords or harmonics; this does not "
                  "identify the singer or recover a score. Unvoiced contour intervals have null MIDI.")
        if self.role == "stem":
            reason += " A stem isolates the part and reduces accompaniment ambiguity."
        return ctx.estimate("audio.melody", value, "MIDI", Method("harmonic-salience-viterbi", reason), cuts)


def _track(salience: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    pitches = np.arange(LOW_MIDI, HIGH_MIDI + 1)
    states = len(pitches) + 1
    transition = -0.12 * np.abs(pitches[:, None] - pitches[None, :])
    costs = np.full((states, states), -0.5)
    costs[:-1, :-1] = transition
    costs[-1, -1] = 0
    peak = np.max(salience, axis=1, keepdims=True)
    normalized = np.divide(salience, peak, out=np.zeros_like(salience), where=peak > 0)
    emissions = np.log(np.maximum(normalized, 0.001))
    emissions[~voiced] = -8.0
    emissions = np.column_stack([emissions, np.where(voiced, -2.0, 0.0)])
    back = np.zeros((len(salience), states), dtype=np.uint8)
    score = emissions[0]
    for i in range(1, len(salience)):
        candidate = score[:, None] + costs
        back[i] = np.argmax(candidate, axis=0)
        score = candidate[back[i], np.arange(states)] + emissions[i]
    path = np.empty(len(salience), dtype=int)
    path[-1] = int(np.argmax(score))
    for i in range(len(path) - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return np.where(path == states - 1, np.nan, LOW_MIDI + path).astype(float)


def melody(store: FeatureStore, *, beats, role: str = "mixdown") -> Melody | Skip:
    activations = getattr(store, "activations", None)
    if store.seconds < 2 or activations is None or not len(activations):
        return Skip("melody needs at least 2 s and harmonic note activations.")
    covered = min(store.seconds, MAX_SECONDS, len(activations) / store.row_rate)
    rows = min(len(activations), int(np.ceil(covered * store.row_rate)))
    salience = np.asarray(activations[:rows, LOW_MIDI - ACTIVATION_LOW_MIDI:
                                     HIGH_MIDI - ACTIVATION_LOW_MIDI + 1], dtype=float).copy()
    treble = getattr(store, "treble", None)
    if treble is not None and len(treble) >= rows:
        chroma = np.asarray(treble[:rows], dtype=float)
        maxima = np.max(chroma, axis=1, keepdims=True)
        normalized = np.divide(chroma, maxima, out=np.zeros_like(chroma), where=maxima > 0)
        baseline = normalized[:, np.arange(LOW_MIDI, HIGH_MIDI + 1) % 12]
        salience = np.maximum(0, salience - 0.12 * np.max(salience, axis=1, keepdims=True) * baseline)
    maximum = np.max(salience, axis=1)
    reference = float(np.percentile(maximum, 90))
    if reference <= 1e-12:
        return Skip("melody has no harmonic salience in E2-C7.")
    concentration = maximum / np.maximum(np.sum(salience, axis=1), 1e-12)
    voiced = (maximum >= reference * 0.04) & (concentration >= 0.08)
    pitches = _track(salience, voiced)
    edges, basis = unit_edges(covered, beats)
    indices = np.clip(np.ceil(edges * store.row_rate - 1e-9).astype(int), 0, len(pitches))
    contour = []
    for time, start, end in zip(edges[:-1], indices[:-1], indices[1:]):
        part = pitches[start:end]
        valid = part[np.isfinite(part)]
        pitch = float(np.median(valid)) if len(valid) and len(valid) >= len(part) / 2 else None
        contour.append({"startSeconds": round(float(time), 3), "midi": pitch})
    cuts = ()
    if store.seconds > covered + 1e-6:
        cuts = (f"Melody covers only the first {covered:g} s of {store.seconds:g} s.",)
    return Melody(pitches, store.row_rate, tuple(contour), covered, basis, cuts, role)
