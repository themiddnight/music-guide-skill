"""Behavioral checks for compact, tuned harmonic features."""
from __future__ import annotations

import unittest

import numpy as np

from signals import midi_hz, record, tone
from music_guide.estimates.store import FeatureRecorder, projected_bytes


class HarmonicStoreTests(unittest.TestCase):
    rate = 22050

    def test_detuned_chord_keeps_its_pitch_classes(self):
        audio = sum(tone(midi_hz(note + 0.3), 5, self.rate, harmonics=6) for note in (60, 64, 67))
        features = record(audio, self.rate)
        self.assertAlmostEqual(features.tuning_cents, 30, delta=10)
        self.assertEqual(set(np.argsort(features.treble[10:-10].mean(axis=0))[-3:]), {0, 4, 7})

    def test_overtones_do_not_create_a_major_third(self):
        features = record(tone(midi_hz(62), 5, self.rate, harmonics=8), self.rate)
        chroma = features.treble[10:-10].mean(axis=0)
        self.assertLess(chroma[6], 0.1 * chroma[2])

    def test_bass_chroma_follows_the_bass_note(self):
        audio = sum(tone(midi_hz(note), 5, self.rate, amplitude=0.12, harmonics=6) for note in (65, 69, 72))
        audio += tone(midi_hz(45), 5, self.rate, amplitude=0.25, harmonics=6)
        features = record(audio, self.rate)
        self.assertEqual(int(np.argmax(features.bass[10:-10].mean(axis=0))), 9)

    def test_new_features_are_finite_read_only_and_chunk_invariant(self):
        audio = sum(tone(midi_hz(note), 4, self.rate, harmonics=6) for note in (60, 64, 67))
        left, right = record(audio, self.rate, chunk=113), record(audio, self.rate, chunk=65536)
        for field in ('treble', 'bass', 'treble_narrow', 'activations', 'timbre', 'brightness'):
            with self.subTest(field=field):
                a, b = getattr(left, field), getattr(right, field)
                self.assertTrue(np.all(np.isfinite(a)))
                self.assertFalse(a.flags.writeable)
                np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
        self.assertEqual(left.activations.shape, (40, 73))
        self.assertEqual(left.timbre.shape, (40, 12))
        self.assertEqual(left.brightness.shape, (40,))

    def test_loudness_is_optional_and_copied_from_the_shared_meter(self):
        recorder = FeatureRecorder(self.rate, 1, self.rate)
        recorder.feed(np.zeros((self.rate, 1)))
        energy = np.arange(10, dtype=float)
        features = recorder.finish(loudness_segments=energy)
        energy[:] = -1
        np.testing.assert_array_equal(features.loudness, np.arange(10))
        self.assertFalse(features.loudness.flags.writeable)

    def test_melody_storage_stops_at_thirty_minutes(self):
        rate, frames = 8000, 1801 * 8000
        recorder = FeatureRecorder(rate, 1, frames)
        for begin in range(0, frames, 65536):
            recorder.feed(np.zeros((min(65536, frames - begin), 1)))
        store = recorder.finish()
        self.assertEqual(store.activations.shape, (18000, 73))
        self.assertEqual(store.treble.shape, (18010, 12))
        self.assertEqual(store.timbre.shape, (18010, 12))

    def test_tuning_lock_is_invariant_to_block_size(self):
        rate = 8000
        audio = tone(midi_hz(62.3), 64, rate, harmonics=6)
        small, large = record(audio, rate, chunk=113), record(audio, rate, chunk=65536)
        self.assertAlmostEqual(small.tuning_cents, 30, delta=10)
        np.testing.assert_allclose(small.treble, large.treble, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(small.activations, large.activations, rtol=1e-5, atol=1e-5)

    def test_projected_memory_with_activations_is_bounded(self):
        self.assertLess(projected_bytes(6 * 3600, self.rate), 400 * 2 ** 20)


if __name__ == '__main__':
    unittest.main()
