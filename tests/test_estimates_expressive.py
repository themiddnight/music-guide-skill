"""Known feature fixtures for section, energy, mood and predominant-pitch behavior."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills" / "music-guide" / "scripts"))

from music_guide.estimates import energy, melody, structure
from music_guide.estimates.common import ReportContext, Skip


NONE = Skip("no pulse")


def features(pattern="ABAB", section_seconds=16):
    size = section_seconds * 10
    chroma = np.vstack([np.tile(np.eye(12)[0 if letter == "A" else 6], (size, 1)) for letter in pattern])
    timbre = np.vstack([np.tile(np.eye(12)[1 if letter == "A" else 10], (size, 1)) for letter in pattern])
    seconds = len(chroma) / 10
    onset = np.zeros((int(seconds * 100), 3))
    onset[::50] = 1
    return SimpleNamespace(pcp=chroma, treble=chroma, timbre=timbre, seconds=seconds,
                           row_rate=10.0, source_rate=None, onset=onset, flux=onset.sum(axis=1), onset_rate=100.0,
                           loudness=np.repeat([0.001, 0.1], len(chroma) // 2),
                           brightness=np.repeat([800.0, 3500.0], len(chroma) // 2))


class StructureTest(unittest.TestCase):
    def test_abab_and_aaba(self):
        for pattern, expected in (("ABAB", ("A", "B", "A", "B")), ("AABA", ("A", "B", "A"))):
            with self.subTest(pattern=pattern):
                result = structure.structure(features(pattern), beats=NONE, meter=NONE)
                self.assertEqual(result.letters, expected)
                boundaries = [0, 16, 32, 48, 64] if pattern == "ABAB" else [0, 32, 48, 64]
                np.testing.assert_allclose(result.boundaries, boundaries, atol=2)
                self.assertEqual(result.basis, "fixed")

    def test_homogeneous_piece_is_one_segment(self):
        result = structure.structure(features("AAAA"), beats=NONE, meter=NONE)
        self.assertEqual(result.letters, ("A",))
        np.testing.assert_array_equal(result.boundaries, [0, 64])

    def test_short_file_skips(self):
        self.assertIsInstance(structure.structure(features("A"), beats=NONE, meter=NONE), Skip)

    def test_downbeats_snap_boundaries(self):
        meter = SimpleNamespace(downbeats=np.arange(1, 64, 2))
        result = structure.structure(features(), beats=NONE, meter=meter)
        self.assertTrue(all(time in meter.downbeats for time in result.boundaries[1:-1]))

    def test_summary_and_full_bounds(self):
        result = structure.Structure(np.arange(71.0), tuple("A" for _ in range(70)),
                                     np.zeros(70), "fixed", np.arange(70.0))
        report = result.report(ReportContext("summary", ""))
        self.assertEqual(len(report.facts[0].value["segments"]), 64)
        self.assertEqual(len(report.limits), 1)
        self.assertEqual(len(result.report(ReportContext("full", "")).facts[0].value["segments"]), 70)


class EnergyMoodTest(unittest.TestCase):
    def test_energy_follows_sections(self):
        store = features("AB")
        sections = SimpleNamespace(boundaries=np.array([0., 16., 32.]))
        result = energy.energy(store, structure=sections, beats=NONE)
        self.assertLess(result.segments[0]["loudnessLufs"], result.segments[1]["loudnessLufs"])
        self.assertLess(result.segments[0]["brightnessHz"], result.segments[1]["brightnessHz"])
        self.assertEqual(result.segmented_by, "structure")
        self.assertAlmostEqual(result.onset_rate, 2)

    def test_uniform_segments_without_structure(self):
        result = energy.energy(features(), structure=NONE, beats=NONE)
        self.assertEqual(result.segmented_by, "uniform")
        self.assertEqual(result.segments[-1]["endSeconds"], 64)

    def test_uniform_long_file_covers_whole_duration(self):
        store = features("A", 940)
        result = energy.energy(store, structure=NONE, beats=NONE)
        self.assertEqual(len(result.segments), 64)
        self.assertEqual(result.segments[-1]["endSeconds"], 940)

    def test_energy_summary_cap_reports_cut(self):
        result = energy.energy(features(), structure=NONE, beats=NONE)
        result = replace(result, segments=tuple(result.segments[0] for _ in range(70)))
        report = result.report(ReportContext("summary", ""))
        self.assertEqual(len(report.facts[0].value["segments"]), 64)
        self.assertEqual(len(report.limits), 1)
        self.assertIn("64 of 70", report.limits[0].message)

    def test_bar_detail_requires_meter(self):
        store = features()
        beats = SimpleNamespace(times=np.arange(0., 64., 0.5))
        result = energy.energy(store, structure=NONE, beats=beats)
        self.assertEqual(result.detail_key, "perBeat")
        meter = SimpleNamespace(downbeats=np.arange(0., 64., 1.5))
        result = energy.energy(store, structure=NONE, beats=beats, meter=meter)
        value = result.report(ReportContext("full", "")).facts[0].value
        self.assertEqual(value["perBar"][0]["endSeconds"], 1.5)
        self.assertNotIn("perWindow", value)

    def test_loud_bright_major_fast_is_energetic(self):
        timeline = energy.EnergyTimeline((), (), "uniform", -8., 4000., 8., "fixed")
        result = energy.mood(timeline, tempo=SimpleNamespace(candidates=((180, 1),)),
                             chords=SimpleNamespace(major_share=1, minor_share=0))
        self.assertEqual(result.labels[0], "energetic/uplifting")
        self.assertEqual(result.inputs["majorShare"], 1)
        self.assertIn("not trained probabilities", result.report(ReportContext("summary", "")).facts[0].value["formula"])

    def test_quiet_dark_minor_slow_is_calm_or_sad(self):
        timeline = energy.EnergyTimeline((), (), "uniform", -30., 500., 0., "fixed")
        result = energy.mood(timeline, tempo=SimpleNamespace(candidates=((60, 1),)),
                             chords=SimpleNamespace(major_share=0, minor_share=1))
        self.assertEqual(result.labels[0], "sad/dark")
        self.assertLess(result.arousal, 0.1)

    def test_mood_inputs_null_for_skipped_dependencies(self):
        result = energy.energy(features(), structure=NONE, beats=NONE)
        mood = energy.mood(result, tempo=NONE, rhythm=NONE, chords=NONE, key=NONE)
        self.assertIsNone(mood.inputs["tempo"])
        self.assertIsNone(mood.inputs["majorShare"])
        self.assertIsNone(mood.inputs["keyMode"])

    def test_missing_loudness_is_not_fabricated(self):
        store = features()
        store.loudness = None
        result = energy.energy(store, structure=NONE, beats=NONE)
        self.assertIsNone(result.segments[0]["loudnessLufs"])
        self.assertIsNone(energy.mood(result).inputs["loudness"])


class MelodyTest(unittest.TestCase):
    def store(self, accompaniment=0.0):
        store = features("A", 12)
        activations = np.zeros((120, 73), dtype=np.float32)
        truth = np.repeat([60, 64, 67, 72, 67, 64], 20)
        activations[np.arange(120), truth - 33] = 1
        activations[40:50] = 0
        activations[:, [48 - 33, 55 - 33]] += accompaniment
        store.activations = activations
        return store

    def test_melody_alone(self):
        result = melody.melody(self.store(), beats=NONE)
        fact = result.report(ReportContext("full", "")).facts[0]
        self.assertEqual(fact.value["lowMidi"], 60)
        self.assertEqual(fact.value["highMidi"], 72)
        self.assertAlmostEqual(fact.value["medianMidi"], 64, delta=1)
        self.assertAlmostEqual(fact.value["voicedShare"], 110 / 120, delta=0.03)
        self.assertTrue(any(point["midi"] is None for point in fact.value["contour"]))
        self.assertEqual(len(fact.value["frames"]), 120)
        self.assertEqual(fact.provenance, "inferred")
        self.assertEqual(fact.confidence, "low")

    def test_melody_over_pad(self):
        result = melody.melody(self.store(0.2), beats=NONE)
        value = result.report(ReportContext("summary", "")).facts[0].value
        self.assertAlmostEqual(value["medianMidi"], 64, delta=2)
        self.assertEqual(value["highMidi"], 72)

    def test_stem_melody_reason(self):
        result = melody.melody(self.store(), beats=NONE, role="stem")
        self.assertIn("stem isolates", result.report(ReportContext("summary", "")).facts[0].reason)

    def test_melody_cap(self):
        with patch.object(melody, "MAX_SECONDS", 5):
            result = melody.melody(self.store(), beats=NONE)
        report = result.report(ReportContext("summary", ""))
        self.assertEqual(report.facts[0].value["coveredSeconds"], 5)
        self.assertEqual(len(report.limits), 1)
        self.assertIn("first 5 s", report.limits[0].message)

    def test_silence_skips(self):
        store = self.store()
        store.activations[:] = 0
        self.assertIsInstance(melody.melody(store, beats=NONE), Skip)

    def test_contour_and_frame_caps_warn(self):
        result = melody.melody(self.store(), beats=NONE)
        with patch.object(melody, "SUMMARY_POINTS", 5):
            report = result.report(ReportContext("summary", ""))
        self.assertEqual(len(report.facts[0].value["contour"]), 5)
        self.assertEqual(len(report.limits), 1)
        with patch.object(melody, "FULL_LIMIT", 10):
            report = result.report(ReportContext("full", ""))
        self.assertEqual(len(report.facts[0].value["frames"]), 10)
        self.assertEqual(len(report.limits), 2)


class RecordedSignalTest(unittest.TestCase):
    def setUp(self):
        from music_guide.estimates.store import FeatureStore
        if "activations" not in FeatureStore.__dataclass_fields__:
            self.skipTest("harmonic FeatureStore is provided by the foundation change")

    def test_recorded_melody_with_and_without_pad(self):
        from signals import midi_hz, record, tone
        rate = 16000
        notes = [60, 64, 67, 72, 67, 64]
        voice = np.concatenate([tone(midi_hz(note), 2., rate, 0.15, 6) for note in notes])
        for pad_gain in (0., 0.025):
            with self.subTest(pad_gain=pad_gain):
                pad = sum(tone(midi_hz(note), 12., rate, pad_gain, 3) for note in (48, 55))
                store = record(voice + pad, rate)
                result = melody.melody(store, beats=NONE)
                value = result.report(ReportContext("summary", "")).facts[0].value
                self.assertAlmostEqual(value["medianMidi"], np.median(notes), delta=1)
                self.assertAlmostEqual(value["lowMidi"], min(notes), delta=2)
                self.assertAlmostEqual(value["highMidi"], max(notes), delta=2)
                self.assertGreater(value["voicedShare"], 0.90)
                predicted = np.array([point["midi"] for point in value["contour"]], dtype=float)
                truth = np.repeat(notes, 4)
                self.assertGreater(np.mean(np.abs(predicted - truth) <= 1), 0.90)

    def test_recorded_chord_sections_with_drums(self):
        from signals import RATE, _burst, click_track, progression, record
        from music_guide.estimates import beats
        chords = {"A": [(62, 65, 69), (58, 62, 65), (60, 64, 67), (57, 61, 64)],
                  "B": [(64, 68, 71), (61, 64, 68), (66, 69, 73), (59, 63, 66)]}
        hat = _burst(RATE, None, 0.03, 9)
        parts = {letter: progression(notes, 8, 120) for letter, notes in chords.items()}
        for at in np.arange(0, len(parts["B"]), int(RATE / 4)):
            parts["B"][at:at + len(hat)] += 0.2 * hat[:len(parts["B"]) - at]
        clicks, _, _ = click_track(120, 4, 64.6, start=0.)
        for form, expected in (("ABAB", ("A", "B", "A", "B")),
                               ("AABA", ("A", "B", "A")), ("AAAA", ("A",))):
            with self.subTest(form=form):
                audio = np.concatenate([parts[letter] for letter in form])
                store = record(audio + clicks[:len(audio)], RATE, chunk=65536)
                grid = beats.beat_grid(store, beats.tempo(store))
                meter = beats.meter(store, grid)
                result = structure.structure(store, beats=grid, meter=meter)
                self.assertEqual(result.letters, expected)
                wanted = [0, 16, 32, 48, 64] if form == "ABAB" else [0, 32, 48, 64] if form == "AABA" else [0, 64]
                np.testing.assert_allclose(result.boundaries, wanted, atol=2)
                if form == "ABAB":
                    fixed = structure.structure(store, beats=NONE, meter=NONE)
                    self.assertEqual(fixed.letters, expected)
                    np.testing.assert_allclose(fixed.boundaries, wanted, atol=2)
                    with patch.object(structure, "MAX_SEGMENTS", 2):
                        capped = structure.structure(store, beats=NONE, meter=NONE).report(ReportContext("full", ""))
                    self.assertLessEqual(len(capped.facts[0].value["segments"]), 2)
                    self.assertTrue(capped.limits)

    def test_recorded_repeated_sections(self):
        from signals import midi_hz, record, tone
        rate = 16000
        sounds = [tone(midi_hz(note), 16., rate, 0.15, 6) for note in (60, 66, 60, 66)]
        result = structure.structure(record(np.concatenate(sounds), rate), beats=NONE, meter=NONE)
        self.assertEqual(result.letters, ("A", "B", "A", "B"))
        np.testing.assert_allclose(result.boundaries, [0, 16, 32, 48, 64], atol=2)


if __name__ == "__main__":
    unittest.main()
