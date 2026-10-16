#!/usr/bin/env python3
"""Compare baseline and experimental chord-aware keys on one audio file.

Usage: key_ab.py MIXDOWN [--truth TRUTH.json]
Only key metrics are scored; no other family can inflate the comparison.
The candidate remains offline unless the available checks justify promotion.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'skills/music-guide/scripts'))
from music_guide.adapters.audio import _measure
from music_guide.estimates import beats, harmony, render
from music_guide.estimates.chords import chords
from music_guide.estimates.common import ReportContext, Skip, Method
from music_guide.estimates.store import FeatureStore
from music_guide.estimates.harmony import (KeyResult, KeySegment, KEY_CELL_SECONDS, KEY_SWITCH_PENALTY,
                                         MIN_KEY_TIMELINE_SECONDS, PITCH_CLASSES, key_scores, _pitch_skip, _viterbi_runs)
from evaluate import evaluate


def chord_key(store: FeatureStore, timeline) -> KeyResult | Skip:
    """Offline chord-function hypothesis. Chord labels can amplify transcription errors."""
    from music_guide.estimates.chords import QUALITY_INTERVALS

    if isinstance(timeline, Skip) or _pitch_skip(store) is not None:
        return Skip("chord-aware key needs both usable pitch classes and a chord timeline.")
    cells = int(np.ceil(store.seconds / KEY_CELL_SECONDS))
    profile = np.zeros((cells, 12))
    membership = np.zeros((cells, 24))
    durations = np.zeros(cells)
    scales = ((0, 2, 4, 5, 7, 9, 11), (0, 2, 3, 5, 7, 8, 10, 11))
    key_sets = [set((np.array(scales[index // 12]) + index % 12) % 12) for index in range(24)]
    function_weights = {0: 2.0, 7: 1.5, 5: 1.5}
    for chord, start, end in zip(timeline.per_unit, timeline.unit_times[:-1], timeline.unit_times[1:]):
        if chord.root is None or chord.quality is None:
            continue
        root = PITCH_CLASSES.index(chord.root)
        notes = {(root + interval) % 12 for interval in QUALITY_INTERVALS[chord.quality]}
        tones = np.array([float(index in notes) for index in range(12)])
        fit = np.array([function_weights.get((root - index % 12) % 12, 1.) / 2
                        if notes <= scale else 0. for index, scale in enumerate(key_sets)])
        first = max(0, int(start // KEY_CELL_SECONDS))
        last = min(cells, int(end // KEY_CELL_SECONDS) + 1)
        for cell in range(first, last):
            overlap = max(0., min(end, (cell + 1) * KEY_CELL_SECONDS) - max(start, cell * KEY_CELL_SECONDS))
            profile[cell] += tones * overlap
            membership[cell] += fit * overlap
            durations[cell] += overlap
    scored = key_scores(profile)
    if scored is None:
        return Skip("the chord timeline has no pitched cells for a key estimate.")
    correlation, usable = scored
    membership = np.divide(membership, durations[:, None], out=np.zeros_like(membership),
                           where=durations[:, None] > 0)
    scores = np.where(usable[:, None], .5 * (correlation + membership), 0.)
    method = Method("experimental-chord-function-key", "offline candidate, mean of chord-tone Krumhansl-Kessler"
                    " correlation and duration-weighted diatonic chord fit. Tonic weight 2, dominant/subdominant 1.5,"
                    " other roots 1, normalized by 2. Minor accepts natural and raised seventh. Five-second cells"
                    f" and Viterbi switch penalty {KEY_SWITCH_PENALTY:g} discourage borrowed-chord modulation."
                    " Chord errors propagate; scores are heuristic ranks, not correlations or probabilities.")
    segments = None
    if store.seconds >= MIN_KEY_TIMELINE_SECONDS:
        segments = tuple(KeySegment(round(start * KEY_CELL_SECONDS, 1),
                                    round(store.seconds if last else end * KEY_CELL_SECONDS, 1),
                                    np.mean(scores[start:end][usable[start:end]], axis=0))
                         for start, end, last in _viterbi_runs(scores) if np.any(usable[start:end]))
    return KeyResult(np.mean(scores[usable], axis=0), segments, method, method, "score")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mixdown', type=Path)
    parser.add_argument('--truth', type=Path)
    args = parser.parse_args()
    with sf.SoundFile(args.mixdown) as audio:
        store = _measure(audio)['features']
    tempo = beats.tempo(store)
    grid = tempo if isinstance(tempo, Skip) else beats.beat_grid(store, tempo)
    timeline = chords(store, beats=grid)
    truth = json.loads(args.truth.read_text()) if args.truth else {}
    truth = {'keys': truth['keys']} if 'keys' in truth else {}
    result = {}
    for label, outcome in [('baseline', harmony.key(store)), ('candidate', chord_key(store, timeline))]:
        facts, diagnostics = render([(label, outcome)], ReportContext('full', ''))
        report = {'observations': [{'property': f.property, 'value': f.value, 'scope': 'mixdown'} for f in facts],
                  'diagnostics': [{'code': d.code, 'message': d.message} for d in diagnostics]}
        result[label] = {'report': report, 'metrics': [metric._asdict() for metric in evaluate(report, truth)]}
    print(json.dumps(result, allow_nan=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
