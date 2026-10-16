from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from signals import RATE, _burst, click_track, record  # noqa: E402
from music_guide import rhythm  # noqa: E402
from music_guide.estimates import beats, groove  # noqa: E402
from music_guide.estimates.common import Skip  # noqa: E402


class GrooveTest(unittest.TestCase):
    def test_syncopation_orders_straight_offbeat_syncopated(self):
        meter = rhythm.Meter(4, 4)
        straight = rhythm.syncopation([0, 1, 2, 3], meter)
        offbeat = rhythm.syncopation([0.5, 1.5, 2.5, 3.5], meter)
        displaced = rhythm.syncopation([0.75, 1.75, 2.75, 3.75], meter)
        self.assertEqual(straight, 0)
        self.assertLess(straight, offbeat)
        self.assertLess(offbeat, displaced)

    def test_long_rest_does_not_turn_a_phrase_ending_into_syncopation(self):
        self.assertEqual(rhythm.syncopation([0.75, 16, 17, 18, 19], rhythm.Meter(4, 4)), 0)

    def test_audio_syncopation_matches_rhythm_module(self):
        samples = np.zeros(32 * RATE)
        click = _burst(RATE, None, 0.035, 2)
        positions = [4 * bar + position for bar in range(15) for position in (0, 0.75, 2, 2.75)]
        for position in positions:
            at = int((0.5 + position * 0.5) * RATE)
            samples[at:at + len(click)] += click * 0.4
        store = record(samples, RATE)
        tempo = beats.TempoCandidates(((120.0, 1.0),), False)
        grid = beats.BeatGrid(np.arange(0.5, 30.5, 0.5), 120, 1)
        result = groove.groove(store, grid, tempo, meter=Skip("unknown"))
        self.assertIsInstance(result, groove.Rhythm)
        self.assertIsNone(result.syncopation)
        from music_guide.estimates.beats import MeterEstimate
        meter = MeterEstimate((("4/4", 1.0),), 0, grid.times[::4])
        result = groove.groove(store, grid, tempo, meter=meter)
        expected = rhythm.syncopation(positions, rhythm.Meter(4, 4))
        self.assertAlmostEqual(result.syncopation, expected, delta=0.05)

    def test_danceability_prefers_steady_pulse(self):
        steady, _, _ = click_track(120, 4, 32)
        rng = np.random.default_rng(15)
        random = np.zeros_like(steady)
        for second in rng.uniform(0.5, 31, 63):
            i = int(second * RATE)
            random[i:i + 100] = steady[int(0.5 * RATE):int(0.5 * RATE) + 100]
        values = []
        for samples in (steady, random):
            store = record(samples, RATE)
            tempo = beats.tempo(store)
            if isinstance(tempo, Skip):
                values.append(0.0)
                continue
            grid = beats.beat_grid(store, tempo)
            values.append(0.0 if isinstance(grid, Skip) else groove.groove(store, grid, tempo,
                                                                            meter=Skip("unknown")).danceability)
        self.assertGreater(values[0], values[1])

    def test_six_eighth_pulses_are_one_bar(self):
        samples = np.zeros(10 * RATE)
        click = _burst(RATE, None, 0.035, 4)
        for pulse in range(24):
            at = int((0.5 + pulse / 3) * RATE)
            samples[at:at + len(click)] += click * 0.4
        store = record(samples, RATE)
        grid = beats.BeatGrid(0.5 + np.arange(24) / 3, 180, 1)
        tempo = beats.TempoCandidates(((180.0, 1.0),), False)
        meter = beats.MeterEstimate((("6/8", 1.0),), 0, grid.times[::6])
        result = groove.groove(store, grid, tempo, meter=meter)
        self.assertAlmostEqual(result.onsets_per_bar, 6, delta=0.3)
        self.assertEqual(result.syncopation, 0)


if __name__ == "__main__":
    unittest.main()
