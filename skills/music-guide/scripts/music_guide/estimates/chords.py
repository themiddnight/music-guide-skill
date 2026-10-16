"""Chord evidence from beat-synchronous, tuned harmonic-suppressed chroma.

A small hand-built vocabulary and a uniform-switch Viterbi decoder make these low-confidence estimates.
The decoder cannot identify a composer's intended spelling or separate instruments in a mix.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .beats import BeatGrid
from .common import FULL_LIMIT, Method, Report, ReportContext, Skip
from .store import FeatureStore

PITCH_CLASSES = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
QUALITY_INTERVALS = {'maj': (0, 4, 7), 'min': (0, 3, 7), 'sus2': (0, 2, 7), 'sus4': (0, 5, 7),
                     'dim': (0, 3, 6), 'maj7': (0, 4, 7, 11), 'min7': (0, 3, 7, 10), '7': (0, 4, 7, 10)}
LABELS = tuple((root, quality) for root in range(12) for quality in QUALITY_INTERVALS)
TEMPLATES = np.zeros((len(LABELS), 12))
for _index, (_root, _quality) in enumerate(LABELS):
    TEMPLATES[_index, (np.asarray(QUALITY_INTERVALS[_quality]) + _root) % 12] = 1
NORMALIZED = TEMPLATES / np.linalg.norm(TEMPLATES, axis=1, keepdims=True)
ROOTS = np.asarray([root for root, _ in LABELS])
SELF_TRANSITION = 0.9
EMISSION_SCALE = 24.0
MIN_COSINE = 0.68
SEGMENT_LIMIT = 256
METHOD = Method('harmonic-chroma-viterbi',
                'binary chord-template cosine scores from median tuned, harmonic-suppressed treble and bass chroma,'
                ' decoded with a 0.9 self-transition and uniform switches. The note dictionary uses eight partials,'
                ' octave-local whitening and eighteen multiplicative nonnegative least-squares updates; it is not'
                ' trained. Global tuning uses spectral peaks in the first sixty seconds. Ambiguous suspended chords'
                ' list equivalent pitch-class sets when bass evidence cannot separate them. Bass notes are low-confidence'
                ' evidence; instruments, chord spelling, omitted notes and extensions outside the vocabulary are not'
                ' identified. Weak or non-tonal units are N; long sustained chords can hide changes.')


@dataclass(frozen=True)
class Chord:
    root: str | None
    quality: str | None
    bass: str | None = None

    @property
    def label(self) -> str:
        return 'N' if self.root is None else f'{self.root}:{self.quality}'


@dataclass(frozen=True)
class ChordTimeline:
    per_unit: tuple[Chord, ...]
    unit_times: np.ndarray  # seconds, interval edges, one more than per_unit
    basis: Literal['beats', 'fixed']
    alternatives: tuple[tuple[str, ...], ...] = ()
    beat_positions: tuple[float, ...] = ()
    register: str = 'wide'

    def _unit(self, index: int) -> dict:
        chord = self.per_unit[index]
        value = {'startSeconds': round(float(self.unit_times[index]), 3),
                 'endSeconds': round(float(self.unit_times[index + 1]), 3), 'chord': chord.label}
        if self.basis == 'beats':
            positions = self.beat_positions or tuple(range(len(self.unit_times)))
            value.update(startBeat=round(float(positions[index]), 3), endBeat=round(float(positions[index + 1]), 3))
        if chord.bass is not None:
            value['bass'] = chord.bass
        if self.alternatives and self.alternatives[index]:
            value['alternatives'] = list(self.alternatives[index])
        return value

    def segments(self) -> list[dict]:
        segments = []
        for index in range(len(self.per_unit)):
            unit = self._unit(index)
            identity = (unit['chord'], unit.get('bass'), unit.get('alternatives'))
            previous = segments[-1] if segments else None
            if previous is not None and identity == (previous['chord'], previous.get('bass'), previous.get('alternatives')):
                previous['endSeconds'] = unit['endSeconds']
                if self.basis == 'beats':
                    previous['endBeat'] = unit['endBeat']
            else:
                segments.append(unit)
        return segments

    def shares(self) -> list[dict]:
        durations: dict[str, float] = {}
        for chord, duration in zip(self.per_unit, np.diff(self.unit_times)):
            durations[chord.label] = durations.get(chord.label, 0.0) + float(duration)
        total = sum(durations.values())
        return [{'chord': label, 'seconds': round(seconds, 3), 'share': round(seconds / total, 5) if total else 0.0}
                for label, seconds in sorted(durations.items(), key=lambda item: (-item[1], item[0]))]

    def _share(self, qualities: tuple[str | None, ...]) -> float:
        durations = np.diff(self.unit_times)
        total = float(durations.sum())
        return sum(float(duration) for chord, duration in zip(self.per_unit, durations)
                   if chord.quality in qualities) / total if total else 0.0

    @property
    def major_share(self) -> float:
        return self._share(('maj', 'maj7', '7'))

    @property
    def minor_share(self) -> float:
        return self._share(('min', 'min7'))

    @property
    def no_chord_share(self) -> float:
        return self._share((None,))

    def report(self, ctx: ReportContext) -> Report:
        segments, segment_cut = ctx.cap(self.segments(), SEGMENT_LIMIT, 'chord segments')
        value = {'basis': self.basis, 'segments': segments}
        unit_cut = None
        if ctx.full:
            count = min(len(self.per_unit), FULL_LIMIT)
            value['perUnit'] = [self._unit(index) for index in range(count)]
            if count < len(self.per_unit):
                unit_cut = f'Only the first {FULL_LIMIT} of {len(self.per_unit)} per-unit chords are listed.'
        basis_reason = (' Units follow the available beat-grid boundaries, with the intro and tail covered.'
                        if self.basis == 'beats' else ' No reliable beat grid was available; units use a fixed 0.5 s grid.')
        method = Method(METHOD.name, METHOD.reason + basis_reason + f' The {self.register} treble register fit best.'
                        ' Spectral rows below a -70 dBFS sine peak or with flatness above 0.65 are excluded from note salience.')
        timeline = ctx.estimate('audio.chord_timeline', value, 'seconds', method, (segment_cut, unit_cut))
        shares, share_cut = ctx.cap(self.shares(), 12, 'chord shares')
        summary = {'shares': shares, 'majorShare': round(self.major_share, 5),
                   'minorShare': round(self.minor_share, 5), 'noChordShare': round(self.no_chord_share, 5),
                   'otherShare': round(max(0.0, 1 - self.major_share - self.minor_share - self.no_chord_share), 5)}
        return timeline + ctx.estimate('audio.chord_summary', summary, 'share', method, (share_cut,))


def _median_rows(values: np.ndarray, edges: np.ndarray, rate: float) -> np.ndarray:
    index = np.clip(np.ceil(edges * rate - 1e-9).astype(int), 0, len(values))
    output = np.zeros((len(edges) - 1, 12))
    for unit, (start, end) in enumerate(zip(index[:-1], index[1:])):
        if end > start:
            output[unit] = np.median(values[start:end], axis=0)
    return output


def _decode(scores: np.ndarray) -> np.ndarray:
    states = scores.shape[1]
    switch = np.log((1 - SELF_TRANSITION) / (states - 1) / SELF_TRANSITION)
    previous = scores[0] * EMISSION_SCALE
    back = np.empty((len(scores), states), dtype=np.uint8)
    for index in range(1, len(scores)):
        best = int(np.argmax(previous))
        alternatives = np.full(states, best, dtype=np.uint8)
        runner = int(np.argmax(np.where(np.arange(states) == best, -np.inf, previous)))
        alternatives[best] = runner
        switched = previous[alternatives] + switch
        stay = previous >= switched
        back[index] = np.where(stay, np.arange(states), alternatives)
        previous = scores[index] * EMISSION_SCALE + np.maximum(previous, switched)
        previous -= previous.max()
    path = np.empty(len(scores), dtype=np.uint8)
    path[-1] = np.argmax(previous)
    for index in range(len(path) - 1, 0, -1):
        path[index - 1] = back[index, path[index]]
    return path


def chords(store: FeatureStore, *, beats: BeatGrid | Skip) -> ChordTimeline | Skip:
    if store.seconds < 0.5 or not len(store.treble):
        return Skip('at least 0.5 s and harmonic chroma are needed for chord templates.')
    if not np.max(store.treble) > 1e-7:
        return Skip('the harmonic chroma is silent, so no chord evidence is available.')
    basis: Literal['beats', 'fixed'] = 'fixed' if isinstance(beats, Skip) else 'beats'
    boundaries = np.arange(0.5, store.seconds, 0.5) if basis == 'fixed' else beats.times
    edges = np.unique(np.concatenate(([0.0], boundaries[(boundaries > 0) & (boundaries < store.seconds)],
                                      [store.seconds])))
    bass = _median_rows(store.bass, edges, store.row_rate) if len(store.bass) else np.zeros((len(edges) - 1, 12))
    bass_norm = np.divide(bass, bass.sum(axis=1, keepdims=True), out=np.zeros_like(bass),
                          where=bass.sum(axis=1, keepdims=True) > 0)
    rooted = bass_norm[:, ROOTS]
    candidates = [('wide', store.treble)]
    if len(store.treble_narrow):
        candidates.append(('narrow G2-A5', store.treble_narrow))
    best = None
    reference_energy = _median_rows(store.treble, edges, store.row_rate).sum(axis=1)
    for register, rows in candidates:
        treble = _median_rows(rows, edges, store.row_rate)
        amplitude = np.linalg.norm(treble, axis=1)
        normalized = np.divide(treble, amplitude[:, None], out=np.zeros_like(treble), where=amplitude[:, None] > 0)
        cosine = normalized @ NORMALIZED.T
        scores = cosine + 0.12 * rooted + 0.04 * (bass_norm @ TEMPLATES.T)
        silent = amplitude < max(float(np.max(amplitude)) * 0.005, 1e-7)
        unpitched = np.max(cosine, axis=1) < MIN_COSINE
        scores = np.column_stack([scores, np.where(silent | unpitched, 1.0, 0.35)])
        scores[silent, :-1] = 0
        path = _decode(scores)
        selected = scores[np.arange(len(path)), path].copy()
        selected[(path == len(LABELS)) & (reference_energy > 1e-7)] = 0
        fit = float(np.sum(selected))
        fit += float(np.sum(path[1:] != path[:-1])) / EMISSION_SCALE * np.log((1 - SELF_TRANSITION)
                                                                          / (len(LABELS) * SELF_TRANSITION))
        if best is None or fit > best[0]:
            best = fit, path, scores, register
    _, path, scores, register = best
    decoded, ambiguity = [], []
    for unit, state in enumerate(path):
        if state == len(LABELS):
            decoded.append(Chord(None, None))
            ambiguity.append(())
            continue
        root, quality = LABELS[state]
        strongest = int(np.argmax(bass_norm[unit]))
        ordered = np.sort(bass_norm[unit])
        bass_known = ordered[-1] > 0.35 and ordered[-1] > ordered[-2] * 1.25
        bass_note = (PITCH_CLASSES[strongest] if bass_known and strongest != root and TEMPLATES[state, strongest]
                     else None)
        decoded.append(Chord(PITCH_CLASSES[root], quality, bass_note))
        twins = np.flatnonzero(np.all(TEMPLATES == TEMPLATES[state], axis=1))
        ambiguity.append(tuple(f'{PITCH_CLASSES[LABELS[twin][0]]}:{LABELS[twin][1]}' for twin in twins
                               if twin != state and abs(scores[unit, twin] - scores[unit, state]) < 0.04))
    edges.flags.writeable = False
    positions = ()
    if basis == 'beats':
        period = 60 / beats.bpm
        beat_times = np.concatenate(([beats.times[0] - period], beats.times, [beats.times[-1] + period]))
        beat_indices = np.arange(-1, len(beats.times) + 1)
        positions_array = np.interp(edges, beat_times, beat_indices)
        positions_array[edges < beats.times[0]] = (edges[edges < beats.times[0]] - beats.times[0]) / period
        positions_array[edges > beats.times[-1]] = (len(beats.times) - 1
                                                  + (edges[edges > beats.times[-1]] - beats.times[-1]) / period)
        positions = tuple(float(value) for value in positions_array)
    return ChordTimeline(tuple(decoded), edges, basis, tuple(ambiguity), positions, register)
