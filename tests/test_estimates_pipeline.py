from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import jsonschema
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from signals import click_track, progression, record  # noqa: E402
from support import SKILL, run_cli  # noqa: E402

from music_guide.estimates import beats, store  # noqa: E402

SCHEMA = json.loads((SKILL / "schemas" / "evidence.schema.json").read_text())
RATE = 48000
LABELS = ("pitch class profile", "tempo estimate", "key estimate")
CHORDS = [(62, 65, 69), (65, 69, 72), (58, 62, 65), (57, 61, 64)]


def music(seconds: float) -> np.ndarray:
    clicks, _, _ = click_track(120, 4, seconds)
    chords = progression(CHORDS, 4, 120, repeats=int(seconds // 8) + 1)[:len(clicks)]
    return clicks + chords


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def evidence(self, samples: np.ndarray, *flags: str, rate: int = RATE, name: str = "mix.wav", **write) -> dict:
        path = self.dir / name
        sf.write(path, samples, rate, **({"subtype": "FLOAT"} | write))
        result = run_cli("--mixdown", path, *flags, cwd=self.dir)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        data = json.loads(result.stdout)
        jsonschema.validate(data, SCHEMA)
        data["facts"] = {item["property"]: item for item in data["observations"]}
        return data

    def messages(self, data: dict, code: str) -> list[str]:
        return [item["message"] for item in data["diagnostics"] if item["code"] == code]

    def test_store_is_invariant_to_chunk_size(self):
        rate = 192000
        n = np.arange(rate * 9)
        envelope = (n % (rate // 2) < rate // 10).astype(float)
        samples = np.stack([0.2 * np.sin(2 * np.pi * 261.6 * n / rate) * envelope,
                            0.2 * np.sin(2 * np.pi * 370.0 * n / rate) * envelope], axis=1)
        small, large = record(samples, rate, chunk=113), record(samples, rate, chunk=65536)
        self.assertEqual(small.seconds, large.seconds)
        np.testing.assert_array_equal(small.onset, large.onset)
        np.testing.assert_array_equal(small.pcp, large.pcp)
        self.assertEqual(small.pcp.shape, (90, 12))
        self.assertEqual(small.onset.shape[1], 3)
        self.assertFalse(small.pcp.flags.writeable)

    def test_projected_bytes_at_six_hours(self):
        self.assertLess(store.projected_bytes(6 * 3600, 22050), 400 * 2 ** 20)

    def test_failed_analyzer_keeps_loudness(self):
        def broken(*_):
            raise ZeroDivisionError("boom")
        with mock.patch.object(beats, "tempo", broken):
            data = self.evidence(np.stack([music(20)] * 2, axis=1))
        self.assertEqual(self.messages(data, "estimate_failed"),
                         ["No tempo estimate: the analyzer failed with ZeroDivisionError: boom"])
        self.assertEqual(next(item["severity"] for item in data["diagnostics"] if item["code"] == "estimate_failed"),
                         "warning")
        self.assertIn("audio.integrated_loudness", data["facts"])
        self.assertIn("audio.key_estimate", data["facts"])

    def test_failed_report_is_isolated(self):
        def broken(*_):
            raise KeyError("shape")
        with mock.patch.object(beats.TempoCandidates, "report", broken):
            data = self.evidence(np.stack([music(20)] * 2, axis=1))
        self.assertEqual(self.messages(data, "estimate_failed"),
                         ["No tempo estimate: reporting failed with KeyError: 'shape'"])
        self.assertNotIn("audio.tempo_estimate", data["facts"])
        for kept in ("audio.integrated_loudness", "audio.key_estimate", "audio.pitch_class_profile"):
            self.assertIn(kept, data["facts"])

    def test_silence_and_short_files_skip_without_failure(self):
        for seconds in (0.3, 5, 30):
            with self.subTest(seconds=seconds):
                data = self.evidence(np.zeros((int(seconds * RATE), 2)))
                self.assertEqual(self.messages(data, "estimate_failed"), [])
                skipped = self.messages(data, "estimate_skipped")
                self.assertEqual([message.split(":")[0] for message in skipped], [f"No {label}" for label in LABELS])
                self.assertTrue(all(len(message.split(": ", 1)[1]) > 10 for message in skipped))
                self.assertFalse([name for name in data["facts"] if name.endswith(("_estimate", "_timeline",
                                                                                   "_profile"))
                                  and name != "audio.short_term_loudness_timeline"])
                self.assertIn("audio.integrated_loudness", data["facts"])

    def test_every_estimate_is_inferred_low_with_method_reason(self):
        samples = np.stack([music(40)] * 2, axis=1)
        summary, full = self.evidence(samples), self.evidence(samples, "--detail", "full", name="full.wav")
        self.assertEqual(self.messages(summary, "estimate_failed") + self.messages(full, "estimate_failed"), [])
        estimates = {"audio.tempo_estimate", "audio.key_estimate", "audio.key_timeline"}
        self.assertLessEqual(estimates, set(summary["facts"]))
        for name in estimates:
            for data in (summary, full):
                fact = data["facts"][name]
                self.assertEqual((fact["provenance"], fact["confidence"]), ("inferred", "low"))
                self.assertTrue(fact["confidenceReason"].startswith("Estimate, not a fact: "))
                self.assertTrue(fact["method"])
            brief, detailed = summary["facts"][name]["value"], full["facts"][name]["value"]
            self.assertEqual({key: detailed[key] for key in brief}, brief)
        profile = summary["facts"]["audio.pitch_class_profile"]
        self.assertEqual((profile["provenance"], profile["confidence"]), ("measured", "medium"))

    def test_lossy_and_fold_notes_reach_every_estimate_reason(self):
        mono = music(32)
        wide = np.zeros((len(mono), 12))
        wide[:, 0], wide[:, 1] = mono, mono
        folded = self.evidence(wide)
        try:
            lossy = self.evidence(np.stack([mono] * 2, axis=1), name="mix.mp3", format="MP3",
                                  subtype="MPEG_LAYER_III")
        except (sf.LibsndfileError, ValueError, TypeError) as error:
            self.skipTest(f"libsndfile cannot write MP3: {error}")
        names = ("audio.pitch_class_profile", "audio.tempo_estimate", "audio.key_estimate", "audio.key_timeline")
        for name in names:
            with self.subTest(name=name):
                self.assertIn("12 channels were folded into 8 groups", folded["facts"][name]["confidenceReason"])
                self.assertIn("Lossy decode", lossy["facts"][name]["confidenceReason"])


if __name__ == "__main__":
    unittest.main()
