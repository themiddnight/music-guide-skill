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
from support import ROOT, SKILL, run_cli  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))

import evaluate  # noqa: E402

from music_guide import estimates  # noqa: E402
from music_guide.estimates import beats, harmony, store  # noqa: E402
from music_guide.estimates.common import ReportContext  # noqa: E402

SCHEMA = json.loads((SKILL / "schemas" / "evidence.schema.json").read_text())
RATE = 48000
LABELS = ("pitch class profile", "tempo estimate", "key estimate", "beat grid", "meter estimate", "swing estimate")
CHORDS = [(62, 65, 69), (65, 69, 72), (58, 62, 65), (57, 61, 64)]


def music(seconds: float) -> np.ndarray:
    clicks, _, _ = click_track(120, 4, seconds, offbeats=True)
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

    def test_failed_analyzer_keeps_loudness_and_names_dependents(self):
        def broken(*_):
            raise ZeroDivisionError("boom")
        with mock.patch.dict(estimates.BACKENDS, {"builtin": estimates.Backend("builtin", beat_grid=broken)}):
            data = self.evidence(np.stack([music(20)] * 2, axis=1))
        self.assertEqual(self.messages(data, "estimate_failed"),
                         ["No beat grid: the analyzer failed with ZeroDivisionError: boom"])
        self.assertEqual(next(item["severity"] for item in data["diagnostics"] if item["code"] == "estimate_failed"),
                         "warning")
        skipped = self.messages(data, "estimate_skipped")
        for label in ("meter estimate", "swing estimate"):
            self.assertIn(f"No {label}: needs the beat grid, which was not produced: the analyzer failed with"
                          " ZeroDivisionError: boom", skipped)
        for kept in ("audio.integrated_loudness", "audio.tempo_estimate", "audio.key_estimate"):
            self.assertIn(kept, data["facts"])

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
        names = {"audio.tempo_estimate", "audio.key_estimate", "audio.key_timeline", "audio.beat_grid",
                 "audio.meter_estimate", "audio.swing_ratio"}
        self.assertLessEqual(names, set(summary["facts"]))
        for name in names:
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
        names = ("audio.pitch_class_profile", "audio.tempo_estimate", "audio.key_estimate", "audio.key_timeline",
                 "audio.beat_grid", "audio.meter_estimate", "audio.swing_ratio")
        for name in names:
            with self.subTest(name=name):
                self.assertIn("12 channels were folded into 8 groups", folded["facts"][name]["confidenceReason"])
                self.assertIn("Lossy decode", lossy["facts"][name]["confidenceReason"])



class TruncationTest(unittest.TestCase):
    def test_truncated_lists_name_their_property_for_the_evaluator(self):
        samples, _, _ = click_track(120, 4, 30, offbeats=True)
        recorded = record(samples, RATE, chunk=65536)
        with mock.patch.object(beats, "FULL_LIMIT", 5):
            facts, diagnostics = estimates.render(estimates.analyze(recorded, estimates.BACKENDS["builtin"], "mixdown"),
                                                  ReportContext("full", ""))
        values = {fact.property: fact.value for fact in facts}
        self.assertEqual(len(values["audio.beat_grid"]["beatTimes"]), 5)
        self.assertEqual(len(values["audio.meter_estimate"]["downbeatTimes"]), 5)
        report = {"observations": [{"property": fact.property, "value": fact.value, "scope": "mixdown"}
                                   for fact in facts],
                  "diagnostics": [{"code": item.code, "message": item.message} for item in diagnostics]}
        observations = evaluate.Observations(report)
        self.assertTrue(observations.truncated("audio.beat_grid"))
        self.assertTrue(observations.truncated("audio.meter_estimate"))
        self.assertFalse(observations.truncated("audio.key_timeline"))
        self.assertTrue(all(item.severity == "warning" for item in diagnostics if item.code == "estimate_truncated"))

    def test_key_timeline_summary_keeps_64_segments(self):
        segments = tuple(harmony.KeySegment(5.0 * index, 5.0 * index + 5, np.eye(24)[index % 24])
                         for index in range(70))
        result = harmony.KeyResult(np.eye(24)[0], segments)
        summary, full = result.report(ReportContext("summary", "")), result.report(ReportContext("full", ""))
        timeline = {fact.property: fact.value for fact in summary.facts}["audio.key_timeline"]
        self.assertEqual(len(timeline["segments"]), 64)
        self.assertEqual([limit.property for limit in summary.limits], ["audio.key_timeline"])
        self.assertIn("Only the first 64 of 70 key segments", summary.facts[1].reason)
        self.assertEqual(len({fact.property: fact.value for fact in full.facts}["audio.key_timeline"]["segments"]), 70)
        self.assertEqual(full.limits, ())


if __name__ == "__main__":
    unittest.main()
