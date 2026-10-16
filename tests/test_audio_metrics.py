"""Audio measurements checked against values derived from the published standards.

BS.1770-4 defines L = -0.691 + 10*log10(sum G_i * z_i) with G = 1.0 for L, R and C, and
states that a 0 dBFS 1 kHz sine in one of those channels reads -3.01 LKFS. The K-weighting
gain at 1 kHz therefore contributes +0.691 dB, so a sine of peak A dBFS in n such channels
reads A - 3.01 + 10*log10(n) LUFS. For -23 dBFS in both stereo channels that is -23.0 LUFS,
the EBU Tech 3341 test case 1 expectation. The other EBU cases are cited where they are used.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import jsonschema
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import SKILL, assert_matches_subprocess, run_cli  # noqa: E402

SCHEMA = json.loads((SKILL / "schemas" / "evidence.schema.json").read_text())
RATE = 48000


def tone(dbfs: float, seconds: float, rate: int = RATE, hz: float = 1000.0, phase: float = 0.0) -> np.ndarray:
    n = np.arange(int(round(seconds * rate)))
    return 10 ** (dbfs / 20) * np.sin(2 * np.pi * hz * n / rate + phase)


def stereo(mono: np.ndarray) -> np.ndarray:
    return np.stack([mono, mono], axis=1)


def note(midi: int, seconds: float) -> np.ndarray:
    n = np.arange(int(seconds * RATE))
    fade = np.minimum(1, np.minimum(n, n[::-1]) / (0.01 * RATE))
    return 0.2 * fade * np.sin(2 * np.pi * 440 * 2 ** ((midi - 69) / 12) * n / RATE)


class AudioMetricsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def analyze(self, samples: np.ndarray, rate: int = RATE, name: str = "mix.wav", **write) -> dict:
        path = self.dir / name
        sf.write(path, samples, rate, **({"subtype": "FLOAT"} | write))
        return self.analyze_path(path)

    def analyze_path(self, path: Path) -> dict:
        result = run_cli("--mixdown", path, cwd=self.dir)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        data = json.loads(result.stdout)
        jsonschema.validate(data, SCHEMA)
        data["facts"] = {item["property"]: item for item in data["observations"]}
        data["codes"] = {item["code"] for item in data["diagnostics"]}
        return data

    def value(self, data: dict, prop: str):
        return data["facts"][prop]["value"]

    def test_subprocess_smoke_matches_the_in_process_runner(self):
        path = self.dir / "mix.wav"
        sf.write(path, stereo(tone(-23, 5)), RATE, subtype="FLOAT")
        result = assert_matches_subprocess(self, "--mixdown", path, cwd=self.dir)
        self.assertEqual(result.returncode, 0, result.stderr)
        loudness = next(item["value"] for item in json.loads(result.stdout)["observations"]
                        if item["property"] == "audio.integrated_loudness")
        self.assertAlmostEqual(loudness, -23.0, delta=0.1)
        broken = self.dir / "broken.mp3"
        broken.write_text("not audio at all\n" * 20)
        failed = assert_matches_subprocess(self, "--mixdown", broken, cwd=self.dir)
        self.assertEqual(failed.returncode, 2)
        self.assertIn("not a decodable audio file", failed.stderr)

    def test_bs1770_reference_levels(self):
        cases = [
            ("0 dBFS 1 kHz in one stereo channel (BS.1770-4 statement)",
             np.stack([tone(0, 20), np.zeros(20 * RATE)], axis=1), RATE, -3.01),
            ("-20 dBFS 1 kHz in both channels", stereo(tone(-20, 20)), RATE, -20.0),
            ("EBU 3341 case 1, stereo -23 dBFS", stereo(tone(-23, 20)), RATE, -23.0),
            ("EBU 3341 case 3, relative gate", stereo(np.concatenate([tone(-36, 10), tone(-23, 60), tone(-36, 10)])),
             RATE, -23.0),
            ("EBU 3341 case 4, absolute gate", stereo(np.concatenate([tone(-72, 10), tone(-36, 10), tone(-23, 60),
                                                                     tone(-36, 10), tone(-72, 10)])), RATE, -23.0),
            ("EBU 3341 case 5, -26/-20/-26", stereo(np.concatenate([tone(-26, 20), tone(-20, 20.1), tone(-26, 20)])),
             RATE, -23.0),
            ("mono -20 dBFS counts one channel: -20 - 3.01", tone(-20, 10), RATE, -23.01),
            ("44.1 kHz coefficients", stereo(tone(-23, 20, 44100)), 44100, -23.0),
            ("96 kHz coefficients", stereo(tone(-23, 20, 96000)), 96000, -23.0),
        ]
        for label, samples, rate, expected in cases:
            with self.subTest(label):
                data = self.analyze(samples, rate)
                self.assertAlmostEqual(self.value(data, "audio.integrated_loudness"), expected, delta=0.1)
                self.assertEqual(data["facts"]["audio.integrated_loudness"]["unit"], "LUFS")
                self.assertEqual(data["facts"]["audio.integrated_loudness"]["confidence"], "high")
        data = self.analyze(stereo(tone(-23, 20)))
        self.assertAlmostEqual(self.value(data, "audio.max_momentary_loudness"), -23.0, delta=0.1)
        self.assertAlmostEqual(self.value(data, "audio.max_short_term_loudness"), -23.0, delta=0.1)
        self.assertIn("Mono counted as one channel", self.analyze(tone(-20, 1))["facts"]["audio.integrated_loudness"]["confidenceReason"])

    def test_ebu_3342_loudness_range(self):
        cases = [((-20, -30), 10), ((-20, -15), 5), ((-40, -20), 20), ((-50, -35, -20, -35, -50), 15)]
        for levels, expected in cases:
            with self.subTest(levels=levels):
                data = self.analyze(stereo(np.concatenate([tone(level, 20) for level in levels])))
                self.assertAlmostEqual(self.value(data, "audio.loudness_range"), expected, delta=1.0)
                self.assertEqual(data["facts"]["audio.loudness_range"]["unit"], "LU")

    def test_short_term_timeline(self):
        data = self.analyze(stereo(np.concatenate([tone(-20, 20), tone(-30, 20)])))
        timeline = data["facts"]["audio.short_term_loudness_timeline"]
        self.assertEqual(timeline["timeDomain"], "seconds")
        points = timeline["value"]["points"]
        self.assertLessEqual(len(points), 64)
        starts = [point["startSeconds"] for point in points]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(starts[0], 0)
        self.assertLessEqual(starts[-1], 37.0)
        self.assertAlmostEqual(points[0]["lufs"], -20.0, delta=0.1)
        self.assertAlmostEqual(points[-1]["lufs"], -30.0, delta=0.1)
        short = self.analyze(stereo(tone(-20, 2)))
        self.assertIsNone(short["facts"]["audio.short_term_loudness_timeline"]["value"])

    def test_silence_has_null_loudness_with_diagnostic(self):
        data = self.analyze(np.zeros((5 * RATE, 2)))
        for prop in ("audio.integrated_loudness", "audio.max_momentary_loudness", "audio.max_short_term_loudness",
                     "audio.loudness_range", "audio.true_peak", "audio.crest_factor", "audio.band_energy"):
            with self.subTest(prop=prop):
                self.assertIsNone(self.value(data, prop))
        self.assertIn("loudness_unmeasurable", data["codes"])
        self.assertIn("digital_silence", data["codes"])
        self.assertIn("estimate_skipped", data["codes"])
        self.assertNotIn("audio.tempo_estimate", data["facts"])

    def test_true_peak_exceeds_sample_peak(self):
        # An fs/4 sine phased at 45 degrees is sampled at +-0.7071 (-3.01 dBFS) while its
        # continuous peak is 1.0 (0 dBTP). 3 s spans several 65536-frame read blocks.
        for rate in (RATE, 192000):
            with self.subTest(rate=rate):
                data = self.analyze(stereo(tone(0, 3, rate, rate / 4, np.pi / 4)), rate)
                self.assertAlmostEqual(self.value(data, "audio.sample_peak"), -3.0103, delta=0.01)
                true_peak = self.value(data, "audio.true_peak")
                self.assertGreater(true_peak, -0.4)
                self.assertLess(true_peak, 0.2)
                self.assertEqual(data["facts"]["audio.true_peak"]["unit"], "dBTP")
        steady = self.analyze(stereo(tone(-6, 3)))
        self.assertAlmostEqual(self.value(steady, "audio.true_peak"), -6.0, delta=0.05)
        self.assertAlmostEqual(self.value(steady, "audio.crest_factor"), 3.0103, delta=0.01)

    def test_stereo_correlation_and_side_to_mid(self):
        signal = tone(-12, 2)
        same = self.analyze(stereo(signal))
        self.assertAlmostEqual(self.value(same, "audio.stereo_correlation"), 1.0, places=9)
        self.assertIsNone(self.value(same, "audio.side_to_mid_ratio"))
        inverted = self.analyze(np.stack([signal, -signal], axis=1))
        self.assertAlmostEqual(self.value(inverted, "audio.stereo_correlation"), -1.0, places=9)
        self.assertIsNone(self.value(inverted, "audio.side_to_mid_ratio"))
        noise = np.random.default_rng(7).standard_normal((4 * RATE, 2)) * 0.1
        independent = self.analyze(noise)
        self.assertLess(abs(self.value(independent, "audio.stereo_correlation")), 0.05)
        self.assertAlmostEqual(self.value(independent, "audio.side_to_mid_ratio"), 0.0, delta=0.2)
        half = self.analyze(np.stack([signal, 0.5 * signal], axis=1))
        # M = 0.75 s and S = 0.25 s, so side/mid = 20*log10(1/3) = -9.54 dB.
        self.assertAlmostEqual(self.value(half, "audio.side_to_mid_ratio"), -9.542, delta=0.01)
        mono = self.analyze(signal)
        self.assertNotIn("audio.stereo_correlation", mono["facts"])
        self.assertNotIn("audio.side_to_mid_ratio", mono["facts"])

    def test_band_energy_and_centroid(self):
        for hz, band in ((40, "sub"), (100, "low"), (350, "low_mid"), (1000, "mid"), (3000, "high_mid"), (10000, "high")):
            with self.subTest(hz=hz):
                data = self.analyze(stereo(tone(-12, 2, hz=hz)))
                bands = self.value(data, "audio.band_energy")
                self.assertEqual(data["facts"]["audio.band_energy"]["unit"], "dB_rel")
                self.assertEqual(set(bands), {"sub", "low", "low_mid", "mid", "high_mid", "high"})
                self.assertGreater(bands[band], -0.1)
                self.assertTrue(all(value is None or value < -30 for name, value in bands.items() if name != band))
                self.assertAlmostEqual(self.value(data, "audio.spectral_centroid"), hz, delta=hz * 0.02)

    def test_tempo_estimate_is_low_confidence_inference(self):
        rng = np.random.default_rng(3)
        burst = rng.standard_normal(600) * np.exp(-np.arange(600) / 80) * 0.5
        clicks = np.zeros(30 * RATE)
        for start in range(0, len(clicks) - 600, RATE // 2):
            clicks[start:start + 600] += burst
        fact = self.analyze(clicks)["facts"]["audio.tempo_estimate"]
        self.assertEqual((fact["provenance"], fact["confidence"], fact["unit"]), ("inferred", "low", "BPM"))
        candidates = fact["value"]["candidates"]
        self.assertLessEqual(len(candidates), 3)
        self.assertTrue(all(60 <= item["bpm"] <= 200 for item in candidates))
        bpms = [item["bpm"] for item in candidates]
        if not any(abs(bpm - 120) <= 1.5 for bpm in bpms):
            self.assertTrue(any(abs(bpm - 60) <= 1 or abs(bpm - 240) <= 2 for bpm in bpms))
            self.assertTrue(fact["value"]["octaveRelatedCandidates"])
        self.assertIn("Half- and double-time", fact["confidenceReason"])

    def test_key_estimate_and_pitch_class_profile(self):
        scale = [60, 62, 64, 65, 67, 69, 71, 72]
        triad = note(60, 2) + note(64, 2) + note(67, 2)
        data = self.analyze(np.concatenate([np.concatenate([note(m, 0.5) for m in scale] + [triad])] * 3))
        fact = data["facts"]["audio.key_estimate"]
        self.assertEqual((fact["provenance"], fact["confidence"]), ("inferred", "low"))
        self.assertEqual(fact["value"]["candidates"][0], fact["value"]["candidates"][0] | {"tonic": "C", "mode": "major"})
        self.assertEqual(len(fact["value"]["candidates"]), 3)
        self.assertIn("Hirajoshi", fact["confidenceReason"])
        # E Hirajoshi (E F A B C): the profile must concentrate on those five classes so a
        # host can reason about the scale even though no major/minor label fits it.
        hirajoshi = self.analyze(np.concatenate([note(m, 0.5) for m in [64, 65, 69, 71, 72] * 4]))
        profile = self.value(hirajoshi, "audio.pitch_class_profile")
        self.assertEqual(list(profile), ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
        self.assertAlmostEqual(sum(profile.values()), 1.0, places=3)
        self.assertGreater(sum(profile[name] for name in ("E", "F", "A", "B", "C")), 0.75)
        self.assertTrue(all(profile[name] < 0.05 for name in ("C#", "D", "D#", "F#", "G", "G#", "A#")))

    def progression(self, chords: tuple[tuple[int, int], ...], seconds: float) -> np.ndarray:
        """Two-second triads cycled for `seconds`; each chord is (root, third) in semitones."""
        sounds = [note(root, 2) + note(root + third, 2) + note(root + 7, 2) for root, third in chords]
        return np.concatenate([sounds[index % len(sounds)] for index in range(int(seconds // 2))])

    C_MAJOR = ((60, 4), (65, 4), (67, 4))
    E_MAJOR = ((64, 4), (69, 4), (71, 4))
    G_MAJOR = ((67, 4), (60, 4), (62, 4))
    D_MINOR = ((62, 3), (67, 3), (69, 4))

    def tops(self, data: dict) -> list[tuple]:
        return [(item["candidates"][0]["tonic"], item["candidates"][0]["mode"], item["startSeconds"],
                 item["endSeconds"]) for item in self.value(data, "audio.key_timeline")["segments"]]

    def test_key_timeline_shows_a_modulation_between_distant_keys(self):
        mono = np.concatenate([self.progression(self.E_MAJOR, 30), self.progression(self.G_MAJOR, 30)])
        data = self.analyze(stereo(mono))
        fact = data["facts"]["audio.key_timeline"]
        self.assertEqual((fact["provenance"], fact["confidence"]), ("inferred", "low"))
        self.assertEqual(self.tops(data), [("E", "major", 0, 30), ("G", "major", 30, 60)])

    def test_key_timeline_ignores_a_short_parallel_major_passage_and_a_single_chord(self):
        minor = self.progression(self.D_MINOR, 40)
        passages = {"15 s of D major": self.progression(((62, 4), (67, 4), (69, 4)), 15),
                    "one D major chord": self.progression(((62, 4),), 2)}
        for label, passage in passages.items():
            with self.subTest(label):
                data = self.analyze(stereo(np.concatenate([minor, passage, minor])))
                self.assertEqual([top[:2] for top in self.tops(data)], [("D", "minor")])
                candidate = self.value(data, "audio.key_estimate")["candidates"][0]
                self.assertEqual((candidate["tonic"], candidate["mode"]), ("D", "minor"))

    def test_key_timeline_is_one_segment_without_modulation(self):
        data = self.analyze(stereo(self.progression(self.C_MAJOR, 48)))
        segments = self.value(data, "audio.key_timeline")["segments"]
        self.assertEqual(len(segments), 1)
        self.assertEqual((segments[0]["startSeconds"], segments[0]["endSeconds"]), (0, 48))
        self.assertEqual(segments[0]["candidates"][0]["tonic"], "C")

    def test_key_timeline_needs_enough_audio(self):
        data = self.analyze(stereo(self.progression(self.C_MAJOR, 12)))
        self.assertIn("audio.key_estimate", data["facts"])
        self.assertNotIn("audio.key_timeline", data["facts"])

    def test_key_and_pitch_profile_survive_antiphase_stereo(self):
        chord = note(60, 4) + note(64, 4) + note(67, 4)
        control = self.analyze(stereo(chord))
        inverted = self.analyze(np.stack([chord, -chord], axis=1))
        for prop in ("audio.pitch_class_profile", "audio.key_estimate"):
            self.assertIn(prop, list(inverted["facts"]))
            self.assertEqual(self.value(inverted, prop), self.value(control, prop))
        candidate = self.value(inverted, "audio.key_estimate")["candidates"][0]
        self.assertEqual((candidate["tonic"], candidate["mode"]), ("C", "major"))
        self.assertNotIn("No key estimate", " ".join(item["message"] for item in inverted["diagnostics"]))

    def test_many_channels_fold_into_bounded_groups_and_say_so(self):
        chord = note(60, 4) + note(64, 4) + note(67, 4)
        samples = np.zeros((len(chord), 12))
        samples[:, 0], samples[:, 1] = chord, -chord
        data = self.analyze(samples)
        candidate = self.value(data, "audio.key_estimate")["candidates"][0]
        self.assertEqual((candidate["tonic"], candidate["mode"]), ("C", "major"))
        for prop in ("audio.pitch_class_profile", "audio.key_estimate"):
            self.assertIn("12 channels were folded into 8 groups", data["facts"][prop]["confidenceReason"])
        stereo_data = self.analyze(np.stack([chord, -chord], axis=1), name="stereo.wav")
        self.assertNotIn("folded", stereo_data["facts"]["audio.pitch_class_profile"]["confidenceReason"])

    def test_folded_channels_match_the_same_signal_in_fewer_channels(self):
        from music_guide.analysis import MusicFeatures
        rng = np.random.default_rng(5)
        base = rng.standard_normal((3 * RATE, 8)) * 0.1
        doubled = np.concatenate([base, base], axis=1)
        results = []
        for samples in (base, doubled):
            features = MusicFeatures(RATE)
            features.feed(samples)
            features.finish()
            results.append(features.pitch_class_profile())
        self.assertEqual(results[0].keys(), results[1].keys())
        for name in results[0]:
            self.assertAlmostEqual(results[0][name], results[1][name], places=9)

    def test_periodicity_survives_antiphase_stereo(self):
        burst = np.random.default_rng(3).standard_normal(600) * np.exp(-np.arange(600) / 80) * 0.5
        clicks = np.zeros(10 * RATE)
        for start in range(0, len(clicks) - 600, RATE // 2):
            clicks[start:start + 600] += burst
        control = self.analyze(stereo(clicks))
        inverted = self.analyze(np.stack([clicks, -clicks], axis=1))
        self.assertIn("audio.tempo_estimate", list(inverted["facts"]))
        self.assertEqual(self.value(inverted, "audio.tempo_estimate"), self.value(control, "audio.tempo_estimate"))
        self.assertTrue(any(abs(item["bpm"] - 120) < 1.5
                            for item in self.value(inverted, "audio.tempo_estimate")["candidates"]))

    def test_pitch_profile_preserves_independent_channels_under_each_polarity_flip(self):
        left, right = note(60, 3), note(66, 3)
        control = self.analyze(np.stack([left, right], axis=1))
        expected = self.value(control, "audio.pitch_class_profile")
        self.assertGreater(expected["C"], 0.4)
        self.assertGreater(expected["F#"], 0.4)
        for signs in ((-1, 1), (1, -1), (-1, -1)):
            with self.subTest(signs=signs):
                flipped = self.analyze(np.stack([left * signs[0], right * signs[1]], axis=1))
                self.assertEqual(self.value(flipped, "audio.pitch_class_profile"), expected)
                self.assertEqual(self.value(flipped, "audio.key_estimate"), self.value(control, "audio.key_estimate"))

    def test_music_features_are_invariant_to_decimator_chunk_boundaries(self):
        from music_guide.analysis import MusicFeatures
        rate = 192000
        n = np.arange(rate * 9)
        left = 0.2 * np.sin(2 * np.pi * 261.625565 * n / rate)
        right = 0.2 * np.sin(2 * np.pi * 369.994423 * n / rate)
        envelope = (n % (rate // 2) < rate // 10).astype(float)
        samples = np.stack([left * envelope, right * envelope], axis=1)
        whole = MusicFeatures(rate)
        whole.feed(samples)
        whole.finish()
        for chunk in (113, 65536):
            with self.subTest(chunk=chunk):
                streamed = MusicFeatures(rate)
                for start in range(0, len(samples), chunk):
                    streamed.feed(samples[start:start + chunk])
                streamed.finish()
                self.assertEqual(streamed.seconds, whole.seconds)
                self.assertEqual(streamed.pitch_class_profile(), whole.pitch_class_profile())
                self.assertEqual(streamed.key(), whole.key())
                self.assertEqual(streamed.tempo(), whole.tempo())
                self.assertIsNotNone(streamed.tempo())

    def test_decodes_beyond_wav_and_flags_lossy_codecs(self):
        signal = stereo(tone(-20, 5))
        formats = [("mix.flac", "FLAC", "PCM_24", False), ("mix.aiff", "AIFF", "PCM_16", False),
                   ("mix.caf", "CAF", "ALAC_24", False), ("mix.w64", "W64", "FLOAT", False),
                   ("mix.rf64", "RF64", "PCM_24", False), ("mix.ogg", "OGG", "VORBIS", True),
                   ("mix.opus", "OGG", "OPUS", True), ("mix.mp3", "MP3", "MPEG_LAYER_III", True)]
        for name, file_format, subtype, lossy in formats:
            with self.subTest(file_format=file_format, subtype=subtype):
                rate = 48000
                try:
                    sf.write(self.dir / name, signal, rate, format=file_format, subtype=subtype)
                except (sf.LibsndfileError, ValueError, TypeError) as error:
                    self.skipTest(f"libsndfile cannot write {file_format}/{subtype}: {error}")
                data = self.analyze_path(self.dir / name)
                self.assertEqual(self.value(data, "audio.format"), file_format)
                self.assertEqual(self.value(data, "audio.subtype"), subtype)
                loudness = data["facts"]["audio.integrated_loudness"]
                self.assertAlmostEqual(loudness["value"], -20.0, delta=0.5 if lossy else 0.1)
                self.assertEqual("lossy_decode" in data["codes"], lossy)
                self.assertEqual(loudness["confidence"], "medium" if lossy else "high")
                self.assertEqual(data["facts"]["audio.sample_rate"]["confidence"], "high")

    def test_undecodable_audio_is_a_clear_error(self):
        for name in ("mix.wav", "mix.mp3", "mix.flac"):
            with self.subTest(name=name):
                path = self.dir / name
                path.write_text("not audio at all\n" * 20)
                result = run_cli("--mixdown", path, cwd=self.dir)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
                self.assertIn("not a decodable audio file", result.stderr)

    def test_multichannel_weights_are_flagged(self):
        data = self.analyze(np.tile(tone(-20, 5)[:, None], (1, 6)))
        self.assertIn("channel_weighting", data["codes"])
        loudness = data["facts"]["audio.integrated_loudness"]
        # Six channels at G = 1.0: -20 - 3.01 + 10*log10(6) = -15.23 LUFS.
        self.assertAlmostEqual(loudness["value"], -15.23, delta=0.1)
        self.assertEqual(loudness["confidence"], "medium")
        self.assertIn("no channel map", loudness["confidenceReason"])
        self.assertNotIn("audio.stereo_correlation", data["facts"])

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    def test_matches_ffmpeg_ebur128_reference(self):
        rng = np.random.default_rng(1)
        rate = 44100
        noise = rng.standard_normal((rate * 30, 2)) * 0.1
        noise[:, 1] = 0.6 * noise[:, 0] + 0.4 * noise[:, 1]
        samples = noise * np.repeat(rng.uniform(0.2, 1.5, 30), rate)[:, None]
        data = self.analyze(samples, rate)
        report = subprocess.run(["ffmpeg", "-nostats", "-hide_banner", "-i", str(self.dir / "mix.wav"),
                                 "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True)
        summary = report.stderr[report.stderr.rindex("Summary"):]
        integrated = float(re.search(r"I:\s+(-?[\d.]+) LUFS", summary).group(1))
        loudness_range = float(re.search(r"LRA:\s+(-?[\d.]+) LU", summary).group(1))
        true_peak = float(re.search(r"Peak:\s+(-?[\d.]+) dBFS", summary).group(1))
        self.assertAlmostEqual(self.value(data, "audio.integrated_loudness"), integrated, delta=0.1)
        self.assertAlmostEqual(self.value(data, "audio.loudness_range"), loudness_range, delta=0.2)
        self.assertAlmostEqual(self.value(data, "audio.true_peak"), true_peak, delta=0.3)


if __name__ == "__main__":
    unittest.main()
