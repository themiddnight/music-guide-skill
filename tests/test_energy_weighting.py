"""Energy rows retain sample-duration weights rather than amplifying a short tail."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from dataclasses import replace

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signals import record
from music_guide.estimates.energy import energy, _weighted_means
from music_guide.estimates.common import Skip
from music_guide.loudness import LoudnessMeter, k_weighting_sos


class EnergyWeightingTest(unittest.TestCase):
    def test_single_sample_tail_matches_per_sample_weighted_energy(self):
        rate = 16000
        audio = np.zeros((32001, 1))
        audio[-1] = .5
        meter = LoudnessMeter(rate, 1)
        meter.feed(audio)
        store = replace(record(audio, rate), loudness=meter.segment_means())
        filtered = signal.sosfilt(k_weighting_sos(rate), audio, axis=0)
        expected = -.691 + 10 * np.log10(np.sum(filtered ** 2) / len(audio))
        result = energy(store, structure=Skip('fixture'), beats=Skip('fixture'))
        self.assertAlmostEqual(result.mean_loudness, expected, delta=.001)
        self.assertAlmostEqual(result.segments[0]['loudnessLufs'], expected, delta=.001)

    def test_interior_edges_integrate_partial_rows(self):
        store = record(np.zeros(2 * 11025), 11025)
        values = np.full(20, .25)
        actual = _weighted_means(store, values, np.array([.013, .177, .291, 1.999]))
        np.testing.assert_allclose(actual, .25, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
