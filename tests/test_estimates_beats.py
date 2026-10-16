from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from signals import RATE, click_track, record  # noqa: E402
from support import run_cli  # noqa: E402

from music_guide import rhythm  # noqa: E402
from music_guide.estimates import beats  # noqa: E402
from music_guide.estimates.common import Skip  # noqa: E402


def grid_of(samples: np.ndarray):
    store = record(samples, RATE, chunk=65536)
    tempo = beats.tempo(store)
    return store, tempo if isinstance(tempo, Skip) else beats.beat_grid(store, tempo)


def nearest(truth: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    return np.min(np.abs(truth[:, None] - estimate[None, :]), axis=1)


class BeatGridTest(unittest.TestCase):
    def test_beats_follow_clicks(self):
        for bpm in (90, 120, 150):
            with self.subTest(bpm=bpm):
                samples, truth, _ = click_track(bpm, 4, 30)
                _, grid = grid_of(samples)
                self.assertIsInstance(grid, beats.BeatGrid)
                self.assertAlmostEqual(grid.bpm, bpm, delta=1)
                self.assertGreaterEqual(np.mean(nearest(truth, grid.times) <= 0.03), 0.95)
                self.assertTrue(np.all(np.diff(grid.times) > 0))

    def test_offset_start_and_tempo_ramp(self):
        steady, truth, _ = click_track(100, 4, 40, start=1.37)
        ramp, ramp_truth, _ = click_track(100, 4, 40, end_bpm=108, start=1.37)
        _, steady_grid = grid_of(steady)
        _, ramp_grid = grid_of(ramp)
        self.assertAlmostEqual(steady_grid.times[0], 1.37, delta=0.03)
        self.assertGreaterEqual(np.mean(nearest(ramp_truth, ramp_grid.times) <= 0.03), 0.9)
        self.assertLess(ramp_grid.stability, steady_grid.stability)

    def test_noise_has_no_beat_grid(self):
        noise = np.random.default_rng(11).standard_normal(30 * RATE) * 0.1
        _, outcome = grid_of(noise)
        self.assertIsInstance(outcome, Skip)
        self.assertEqual(outcome.code, "estimate_skipped")
        self.assertGreater(len(outcome.reason), 10)


class MeterTest(unittest.TestCase):
    def test_meter_and_downbeat_from_accents(self):
        for beats_per_bar in (3, 4):
            for pickup in range(beats_per_bar):
                with self.subTest(beats_per_bar=beats_per_bar, pickup=pickup):
                    samples, truth, downbeats = click_track(120, beats_per_bar, 40, pickup=pickup)
                    store, grid = grid_of(samples)
                    self.assertAlmostEqual(grid.times[0], truth[0], delta=0.03)
                    estimate = beats.meter(store, grid)
                    self.assertEqual(estimate.candidates[0][0], f"{beats_per_bar}/4")
                    self.assertEqual(estimate.downbeat_phase, pickup)
                    self.assertGreaterEqual(np.mean(nearest(downbeats, estimate.downbeats) <= 0.03), 0.95)

    def test_unaccented_clicks_do_not_claim_a_confident_meter(self):
        samples, _, _ = click_track(120, 4, 40, accent=False)
        store, grid = grid_of(samples)
        estimate = beats.meter(store, grid)
        self.assertEqual({signature for signature, _ in estimate.candidates}, {"2/4", "3/4", "4/4", "6/8"})
        self.assertLess(estimate.candidates[0][1] - estimate.candidates[1][1], 0.1)


class SwingTest(unittest.TestCase):
    def test_swing_ratio(self):
        for ratio, delta in ((1.0, 0.15), (2.0, 0.35)):
            with self.subTest(ratio=ratio):
                samples, _, _ = click_track(120, 4, 30, swing=ratio, offbeats=True)
                store, grid = grid_of(samples)
                swing = beats.swing(store, grid)
                self.assertIsInstance(swing, beats.Swing)
                self.assertAlmostEqual(swing.ratio, ratio, delta=delta)
                self.assertGreaterEqual(swing.pairs, 16)

    def test_beats_without_offbeats_skip_swing(self):
        samples, _, _ = click_track(120, 4, 30)
        store, grid = grid_of(samples)
        self.assertIsInstance(beats.swing(store, grid), Skip)


class RhythmTest(unittest.TestCase):
    def test_rhythm_meter_weights(self):
        self.assertEqual(rhythm.Meter(4, 4).weights(4),
                         [0, -4, -3, -4, -2, -4, -3, -4, -1, -4, -3, -4, -2, -4, -3, -4])
        self.assertEqual(rhythm.Meter(3, 4).weights(2), [0, -2, -1, -2, -1, -2])
        self.assertEqual(rhythm.Meter(6, 8).weights(1), [0, -2, -2, -1, -2, -2])
        with self.assertRaises(ValueError):
            rhythm.Meter(5, 4).weights(4)

    def test_swing_ratio_formula(self):
        self.assertAlmostEqual(rhythm.swing_ratio([0.0, 0.5, 1.0, 1.5]), 1.0)
        self.assertAlmostEqual(rhythm.swing_ratio([2 / 3, 1 + 2 / 3, 2.66]), 2.0, places=1)
        self.assertIsNone(rhythm.swing_ratio([0.0, 1.0, 2.25]))


class BeatFactsTest(unittest.TestCase):
    def test_summary_and_full_fact_shapes(self):
        samples, _, _ = click_track(120, 4, 30, offbeats=True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "clicks.wav"
            sf.write(path, samples, RATE, subtype="FLOAT")
            facts = []
            for detail in ("summary", "full"):
                result = run_cli("--mixdown", path, "--detail", detail, cwd=Path(folder))
                self.assertEqual(result.returncode, 0, result.stderr)
                facts.append({item["property"]: item["value"] for item in json.loads(result.stdout)["observations"]})
        summary, full = facts
        self.assertEqual(set(summary["audio.beat_grid"]), {"bpm", "beatCount", "firstBeatSeconds", "tempoStability"})
        self.assertEqual(set(full["audio.beat_grid"]) - set(summary["audio.beat_grid"]), {"beatTimes"})
        self.assertEqual(len(full["audio.beat_grid"]["beatTimes"]), full["audio.beat_grid"]["beatCount"])
        self.assertEqual(set(summary["audio.meter_estimate"]), {"candidates", "downbeatPhase", "barCount"})
        self.assertEqual(set(full["audio.meter_estimate"]) - set(summary["audio.meter_estimate"]), {"downbeatTimes"})
        self.assertEqual(summary["audio.meter_estimate"]["candidates"][0]["signature"], "4/4")
        self.assertEqual(set(summary["audio.swing_ratio"]), {"ratio", "pairs"})


if __name__ == "__main__":
    unittest.main()
