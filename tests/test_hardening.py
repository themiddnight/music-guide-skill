"""Regression tests for hostile or malformed input. Each test drives the CLI the way a user does."""
from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "music-guide"
sys.path.insert(0, str(SKILL / "scripts"))


def smf(track: bytes, ticks: int = 480) -> bytes:
    return b"MThd" + struct.pack(">IHHH", 6, 0, 1, ticks) + b"MTrk" + struct.pack(">I", len(track)) + track


END_OF_TRACK = b"\x00\xff\x2f\x00"


class CliCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.mix = self.dir / "mix.wav"
        sf.write(self.mix, np.array([0.5, -0.5, 0.25, -0.25], dtype="float32"), 48000, subtype="FLOAT")

    def run_cli(self, *args, mixdown=True):
        prefix = ["--mixdown", str(self.mix)] if mixdown else []
        return subprocess.run([sys.executable, str(SKILL / "scripts" / "analyze.py"), *prefix, *map(str, args)],
                              cwd=self.dir, text=True, capture_output=True)

    def evidence(self, *args, mixdown=True):
        result = self.run_cli(*args, mixdown=mixdown)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        return json.loads(result.stdout)

    def write(self, name: str, data: bytes | str) -> Path:
        path = self.dir / name
        path.write_bytes(data if isinstance(data, bytes) else data.encode())
        return path

    def assert_optional_error(self, data, *, role_count=2):
        self.assertTrue(any(item["code"] == "optional_source_error" for item in data["diagnostics"]), data["diagnostics"])
        self.assertTrue(any(item["property"] == "audio.sample_peak" for item in data["observations"]))
        self.assertEqual(data["sources"][-1]["status"], "error")


class ErrorBoundaryTest(CliCase):
    def test_malformed_midi_meta_events_become_source_errors(self):
        bad_key = self.write("key.mid", smf(b"\x00\xff\x59\x02\x00\x02" + END_OF_TRACK))
        bad_time = self.write("time.mid", smf(b"\x00\xff\x58\x02\x04\x02" + END_OF_TRACK))
        for path in (bad_key, bad_time):
            with self.subTest(path=path.name):
                self.assert_optional_error(self.evidence("--midi", path))

    def test_unexpected_exception_in_optional_source_is_isolated(self):
        import dataclasses
        from music_guide import cli, formats

        def explode(*_):
            raise ZeroDivisionError("boom")
        patched = dataclasses.replace(formats.BY_NAME["solna"], adapter=explode)
        with mock.patch.dict(formats.BY_NAME, {"solna": patched}):
            evidence = cli.build(self.mix, [("solna", self.write("a.solna", "{}"))], None)
        diagnostics = [item for item in evidence.diagnostics if item.code == "optional_source_error"]
        self.assertEqual(len(diagnostics), 1)
        self.assertIn("ZeroDivisionError", diagnostics[0].message)
        self.assertTrue(any(item.property == "audio.sample_peak" for item in evidence.observations))
        with mock.patch.object(cli.audio, "audio", side_effect=ZeroDivisionError("boom")):
            with self.assertRaisesRegex(ValueError, "required mixdown"):
                cli.build(self.mix, [], None)

    def test_huge_note_durations_do_not_crash(self):
        from music_guide.notes import Note, summarize_notes
        with self.assertRaises(ValueError):
            Note(60, 64, 0.0, float("inf"))
        with self.assertRaises(ValueError):
            Note(60, 64, float("nan"), 1.0)
        huge = [Note(60, 64, 0.0, 1.7e308), Note(60, 64, 0.0, 1.7e308)]
        with self.assertRaises(ValueError):
            summarize_notes(huge)

    def test_unknown_xml_encoding_is_a_source_error(self):
        bogus = self.write("a.musicxml", '<?xml version="1.0" encoding="BOGUS"?><score-partwise/>')
        self.assert_optional_error(self.evidence("--project", bogus))


class PolyphonyTest(CliCase):
    def track_summary(self, track: bytes):
        data = self.evidence("--midi", self.write("n.mid", smf(track + END_OF_TRACK)))
        return next(item["value"] for item in data["observations"] if item["property"] == "midi.track_summary")

    def test_zero_length_notes_still_count_as_sounding(self):
        one = b"\x00\x90\x3c\x40\x00\x80\x3c\x00"
        self.assertEqual(self.track_summary(one)["maxPolyphony"], 1)
        three = b"".join(b"\x00\x90" + bytes([p, 0x40]) for p in (60, 64, 67)) + \
            b"".join(b"\x00\x80" + bytes([p, 0]) for p in (60, 64, 67))
        self.assertEqual(self.track_summary(three)["maxPolyphony"], 3)

    def test_touching_notes_do_not_overlap(self):
        track = b"\x00\x90\x3c\x40\x60\x80\x3c\x00\x00\x90\x40\x40\x60\x80\x40\x00"
        self.assertEqual(self.track_summary(track)["maxPolyphony"], 1)


class FiniteValueTest(CliCase):
    def test_evidence_rejects_nonfinite_values_recursively(self):
        from music_guide.model import Evidence
        evidence = Evidence()
        for value in (float("nan"), {"a": [1, {"b": float("inf")}]}, [float("-inf")], 10 ** 400):
            with self.subTest(value=str(value)[:20]), self.assertRaises(ValueError):
                evidence.add("s1", "p", value, "u", "project", "static", "l", "m", "extracted", "high", "r")
        self.assertEqual(evidence.observations, [])

    def test_nonfinite_adapter_value_becomes_a_source_error(self):
        import dataclasses
        from music_guide import cli, formats

        def adapter(path, source_id, evidence, detail):
            evidence.add(source_id, "ok", 1, "u", "project", "static", "l", "m", "extracted", "high", "r")
            evidence.add(source_id, "bad", {"x": float("nan")}, "u", "project", "static", "l", "m", "extracted", "high", "r")
        patched = dataclasses.replace(formats.BY_NAME["solna"], adapter=adapter)
        with mock.patch.dict(formats.BY_NAME, {"solna": patched}):
            result = cli.build(self.mix, [("solna", self.write("a.solna", "{}"))], None)
        self.assertEqual([item.property for item in result.observations if item.sourceIds == ["s2"]], [])
        self.assertTrue(any(item.code == "optional_source_error" for item in result.diagnostics))
        json.dumps(result.as_json(), allow_nan=False)

    def test_extreme_audio_amplitude_fails_the_source_not_the_report(self):
        huge = self.dir / "huge.wav"
        sf.write(huge, np.full(4096, 1e149), 48000, subtype="DOUBLE")
        data = self.evidence("--stem", huge)
        self.assert_optional_error(data)
        loud = self.dir / "loud.wav"
        sf.write(loud, np.full(4096, 2e6), 48000, subtype="DOUBLE")
        self.assert_optional_error(self.evidence("--stem", loud))
        result = self.run_cli("--mixdown", huge, mixdown=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("magnitude", result.stderr)


if __name__ == "__main__":
    unittest.main()
