"""Public backend selection and shared loudness feature contracts."""
from __future__ import annotations

import sys
import json
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import run_cli
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import benchmark
from music_guide.cli import build
from music_guide.estimates import BACKENDS
from music_guide.loudness import LoudnessMeter


class BackendSelectionTest(unittest.TestCase):
    def test_unknown_backend_exits_2(self):
        result = run_cli('--analysis-backend', 'unknown', cwd=Path.cwd())
        self.assertEqual(result.returncode, 2)
        self.assertIn('invalid choice', result.stderr)

    def test_registry_matches_cli_choices(self):
        self.assertEqual(set(BACKENDS), {'builtin'})
        result = run_cli('--help', cwd=Path.cwd())
        self.assertIn('--analysis-backend {builtin}', result.stdout)

    def test_build_rejects_unknown_backend(self):
        with self.assertRaisesRegex(ValueError, 'unknown analysis backend'):
            build(None, [], None, analysis_backend='unknown')


class BenchmarkEvidenceTest(unittest.TestCase):
    def test_fast_rejections_and_missing_analysis_do_not_pass(self):
        self.assertTrue(benchmark.report_errors('{}'))
        self.assertIn('missing audio.chord_timeline', benchmark.report_errors(json.dumps({
            'observations': [{'property': 'audio.duration'}], 'diagnostics': []})))
        self.assertIn('boom', benchmark.report_errors(json.dumps({
            'observations': [], 'diagnostics': [{'code': 'estimate_failed', 'message': 'boom'}]})))


class LoudnessFeatureTest(unittest.TestCase):
    def test_partial_segment_and_chunk_invariance(self):
        rate = 11025
        samples = np.random.default_rng(1).standard_normal((rate + 137, 2)) * .05
        whole = LoudnessMeter(rate, 2)
        whole.feed(samples)
        chunked = LoudnessMeter(rate, 2)
        for start in range(0, len(samples), 733):
            chunked.feed(samples[start:start + 733])
        actual = whole.segment_means()
        self.assertEqual(len(actual), 11)
        self.assertFalse(actual.flags.writeable)
        self.assertTrue(np.all(actual >= 0))
        np.testing.assert_allclose(actual, chunked.segment_means(), atol=1e-14)
        # A complete 400 ms window must equal four weighted 100 ms segments.
        bounds = np.arange(5) * rate // 10
        expected = np.average(actual[:4], weights=np.diff(bounds))
        self.assertAlmostEqual(expected, whole.windows(4)[0][0], places=13)

    def test_empty_and_sub_100ms(self):
        meter = LoudnessMeter(48000, 1)
        self.assertEqual(meter.segment_means().size, 0)
        meter.feed(np.ones((100, 1)) * .01)
        self.assertEqual(meter.segment_means().size, 1)
        self.assertGreater(meter.segment_means()[0], 0)


if __name__ == '__main__':
    unittest.main()
