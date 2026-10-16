"""Offline key candidates retain explicit scoring and cannot silently replace the baseline."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from key_ab import chord_key
from music_guide.estimates import harmony
from music_guide.estimates.common import ReportContext, Skip
from music_guide.estimates.store import FeatureStore


def tonal_store() -> FeatureStore:
    profile = np.zeros((1200, 12), dtype=np.float32)
    for pitch, weight in [(2, 6), (5, 4), (9, 4), (0, 2), (7, 2), (10, 2), (4, 1)]:
        profile[600:, pitch] = weight
    return FeatureStore(onset=np.zeros((2400, 3), dtype=np.float32), onset_rate=20,
                        pcp=profile, row_rate=10, seconds=120., channels=1, folded=False)


class KeyCandidateTest(unittest.TestCase):
    def test_chord_function_scores_are_not_called_correlations(self):
        from music_guide.estimates.chords import Chord, ChordTimeline
        timeline = ChordTimeline(tuple(Chord('D', 'min') if index % 2 == 0 else Chord('A', '7')
                                       for index in range(60)), np.arange(61) * 2., 'fixed')
        result = chord_key(tonal_store(), timeline)
        self.assertNotIsInstance(result, Skip)
        report = result.report(ReportContext('full', ''))
        for fact in report.facts:
            if fact.property == 'audio.key_estimate':
                self.assertIn('score', fact.value['candidates'][0])
                self.assertNotIn('correlation', fact.value['candidates'][0])
                self.assertEqual(fact.value['candidates'][0]['tonic'], 'D')
                self.assertEqual(fact.value['candidates'][0]['mode'], 'minor')
            self.assertIn('offline candidate', fact.reason)

    def test_missing_chords_does_not_replace_baseline(self):
        recorded = tonal_store()
        self.assertIsInstance(chord_key(recorded, Skip('unavailable')), Skip)
        result = harmony.key(recorded)
        report = result.report(ReportContext('summary', ''))
        self.assertEqual(report.facts[0].method, 'krumhansl-kessler-chroma')
        self.assertTrue(all(np.isfinite(segment.scores).all() for segment in result.timeline))


if __name__ == '__main__':
    unittest.main()
