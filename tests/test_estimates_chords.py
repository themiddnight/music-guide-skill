"""Chord labels, ambiguity, inversion and bounded evidence reports."""
from __future__ import annotations

import unittest

import numpy as np

from signals import midi_hz, record, tone
from music_guide.estimates.common import ReportContext, Skip
from music_guide.estimates.store import FeatureStore
from music_guide.estimates.beats import BeatGrid
from music_guide.estimates.chords import Chord, ChordTimeline, QUALITY_INTERVALS, chords


def synthetic_chroma(notes, *, bass=None, seconds=4):
    rows = int(seconds * 10)
    chroma = np.zeros((rows, 12), dtype=np.float32)
    chroma[:, np.asarray(notes, dtype=int) % 12] = 1
    low = np.zeros_like(chroma)
    if bass is not None:
        low[:, bass % 12] = 1
    return FeatureStore(onset=np.zeros((rows, 3)), onset_rate=10, pcp=chroma, treble=chroma, bass=low,
                        seconds=seconds, channels=1, folded=False)


class ChordTests(unittest.TestCase):
    def test_triads_and_sevenths_in_all_roots(self):
        for root in range(12):
            for quality, intervals in QUALITY_INTERVALS.items():
                with self.subTest(root=root, quality=quality):
                    result = chords(synthetic_chroma([(root + i) % 12 for i in intervals], bass=root),
                                    beats=Skip('no beat grid'))
                    expected = f'{("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")[root]}:{quality}'
                    self.assertGreaterEqual(sum(item.label == expected for item in result.per_unit)
                                            / len(result.per_unit), 0.8)

    def test_additive_audio_covers_every_root_and_quality(self):
        rate = 22050
        from signals import progression
        expected, notes = [], []
        names = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
        for root in range(12):
            for quality, intervals in QUALITY_INTERVALS.items():
                expected.append(f'{names[root]}:{quality}')
                notes.append(tuple(60 + root + interval for interval in intervals))
        result = chords(record(progression(notes, 4, 120, rate=rate), rate, chunk=65536),
                        beats=Skip('known fixed test grid'))
        truth = np.repeat(expected, 4)
        actual = [chord.label for chord in result.per_unit]
        overlap = np.mean([here == wanted for here, wanted in zip(actual, truth)])
        root_overlap = np.mean([here.split(':')[0] == wanted.split(':')[0] for here, wanted in zip(actual, truth)])
        self.assertGreaterEqual(overlap, 0.8)
        self.assertGreaterEqual(root_overlap, 0.9)

    def test_audio_triad_progression_is_recognized(self):
        rate = 22050
        audio = []
        for root, quality in ((0, 'maj'), (9, 'min'), (5, 'maj7'), (7, '7')):
            notes = [60 + root + interval for interval in QUALITY_INTERVALS[quality]]
            audio.append(sum(tone(midi_hz(n), 4, rate, amplitude=0.12, harmonics=6) for n in notes)
                         + tone(midi_hz(36 + root), 4, rate, amplitude=0.16, harmonics=6))
        result = chords(record(np.concatenate(audio), rate), beats=Skip('no beat grid'))
        self.assertEqual([segment['chord'] for segment in result.segments()], ['C:maj', 'A:min', 'F:maj7', 'G:7'])

    def test_low_pad_is_not_relabelled_by_a_high_lead(self):
        rate = 22050
        pad = sum(tone(midi_hz(note), 6, rate, amplitude=0.1, harmonics=6) for note in (50, 52, 57))
        lead = tone(midi_hz(85), 6, rate, amplitude=0.14, harmonics=6)
        result = chords(record(pad + lead, rate), beats=Skip('pad with no beat'))
        self.assertEqual(result.segments()[0]['chord'], 'D:sus2')

    def test_high_pad_keeps_its_seventh(self):
        rate = 22050
        pad = sum(tone(midi_hz(note), 6, rate, amplitude=0.1, harmonics=6) for note in (84, 88, 91, 95))
        result = chords(record(pad, rate), beats=Skip('pad with no beat'))
        self.assertEqual(result.segments()[0]['chord'], 'C:maj7')

    def test_near_silence_does_not_become_a_chord(self):
        rate = 22050
        quiet = sum(tone(midi_hz(note), 4, rate, amplitude=1e-6, harmonics=6) for note in (60, 64, 67))
        store = record(quiet, rate)
        self.assertFalse(np.any(store.treble))
        self.assertIsInstance(chords(store, beats=Skip('near silence')), Skip)

    def test_broadband_noise_does_not_become_tuning_or_a_chord(self):
        rate = 22050
        store = record(np.random.default_rng(2).normal(0, 0.1, rate * 5), rate)
        self.assertEqual(store.tuning_cents, 0)
        self.assertIsInstance(chords(store, beats=Skip('noise')), Skip)

    def test_silence_skips_and_silent_spans_have_no_chord(self):
        store = synthetic_chroma([0, 4, 7], bass=0)
        store.treble[10:30] = 0
        store.bass[10:30] = 0
        result = chords(store, beats=Skip('no beat grid'))
        self.assertIn('N', [item.label for item in result.per_unit])
        self.assertIsInstance(chords(synthetic_chroma([], bass=None), beats=Skip('no beat grid')), Skip)

    def test_slash_chord_reports_bass(self):
        result = chords(synthetic_chroma([5, 9, 0], bass=9), beats=Skip('no beat grid'))
        self.assertEqual(result.per_unit[0], Chord('F', 'maj', 'A'))
        self.assertEqual(result.segments()[0]['bass'], 'A')

    def test_sus2_lists_its_sus4_twin_without_bass_evidence(self):
        result = chords(synthetic_chroma([2, 4, 9]), beats=Skip('no beat grid'))
        segment = result.segments()[0]
        self.assertEqual({segment['chord'], *segment['alternatives']}, {'D:sus2', 'A:sus4'})
        rooted = chords(synthetic_chroma([2, 4, 9], bass=2), beats=Skip('no beat grid'))
        self.assertEqual(rooted.segments()[0]['chord'], 'D:sus2')
        self.assertNotIn('alternatives', rooted.segments()[0])

    def test_without_a_beat_grid_uses_fixed_units(self):
        result = chords(synthetic_chroma([0, 4, 7]), beats=Skip('flat onset'))
        self.assertEqual(result.basis, 'fixed')
        np.testing.assert_allclose(result.unit_times, np.arange(0, 4.1, 0.5))
        full = result.report(ReportContext('full', ' Lossy input.'))
        for fact in full.facts:
            self.assertEqual((fact.provenance, fact.confidence), ('inferred', 'low'))
            self.assertIn('Lossy input', fact.reason)
        self.assertIn('perUnit', full.facts[0].value)

    def test_beat_units_cover_intro_and_tail(self):
        result = chords(synthetic_chroma([0, 4, 7]), beats=BeatGrid(np.array([0.3, 0.8, 1.3, 1.8]), 120, 1))
        self.assertEqual(result.basis, 'beats')
        self.assertEqual(result.unit_times[0], 0)
        self.assertEqual(result.unit_times[-1], 4)
        self.assertAlmostEqual(result.segments()[0]['startBeat'], -0.6)
        self.assertEqual(result.shares()[0]['share'], 1)
        self.assertEqual(result.major_share, 1)

    def test_summary_is_capped_with_a_warning(self):
        result = ChordTimeline(tuple(Chord('C' if i % 2 else 'D', 'maj', None) for i in range(300)),
                               np.arange(301) * 0.5, 'fixed')
        report = result.report(ReportContext('summary', ''))
        self.assertEqual(len(report.facts[0].value['segments']), 256)
        self.assertTrue(any(limit.property == 'audio.chord_timeline' for limit in report.limits))


if __name__ == '__main__':
    unittest.main()
