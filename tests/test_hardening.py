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


def sparse_wav(path: Path, frames: int, rate: int) -> None:
    """An 8-bit mono WAV whose header and length agree but whose body is a hole in the file."""
    header = (b"RIFF" + struct.pack("<I", 36 + frames) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate, 1, 8) +
              b"data" + struct.pack("<I", frames))
    with path.open("wb") as stream:
        stream.write(header)
        stream.truncate(len(header) + frames)


class AudioBoundsTest(CliCase):
    def test_overlong_audio_is_rejected_before_decoding(self):
        rate = 8000
        long = self.dir / "long.wav"
        sparse_wav(long, 6 * 3600 * rate + 1, rate)
        self.assertTrue(sf.info(long).frames > 6 * 3600 * rate)
        data = self.evidence("--stem", long)
        self.assert_optional_error(data)
        self.assertIn("6 hours", next(item["message"] for item in data["diagnostics"] if item["code"] == "optional_source_error"))

    def test_frame_count_is_enforced_while_decoding_too(self):
        from music_guide.adapters import audio

        class Liar:
            frames, samplerate, channels = 0, 8000, 1

            def blocks(self, **_):
                for _ in range(3):
                    yield np.full((8000, 1), 0.1)
        with mock.patch.object(audio, "MAX_AUDIO_SECONDS", 2):
            with self.assertRaisesRegex(ValueError, "hours|longer"):
                audio._measure(Liar())

    def test_block_size_shrinks_as_channels_grow(self):
        from music_guide.adapters.audio import block_frames
        self.assertEqual(block_frames(2), 65536)
        self.assertLess(block_frames(256), block_frames(2))
        self.assertLessEqual(block_frames(1024) * 1024, 2 ** 20)

    def test_many_channel_audio_stays_within_memory(self):
        for channels, frames in ((256, 70000), (1024, 5000)):
            with self.subTest(channels=channels):
                wide = self.dir / f"wide{channels}.wav"
                sf.write(wide, np.zeros((frames, channels), dtype="int16") + 1000, 48000, subtype="PCM_16")
                script = ("import resource, sys, platform\n"
                          f"sys.path.insert(0, {str(SKILL / 'scripts')!r})\n"
                          "from pathlib import Path\n"
                          "from music_guide.adapters import audio\n"
                          "from music_guide.model import Evidence\n"
                          f"audio.audio(Path({str(wide)!r}), 's1', Evidence())\n"
                          "peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
                          "print(peak if platform.system() == 'Darwin' else peak * 1024)\n")
                result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertLess(int(result.stdout), 600 * 1024 * 1024)


class AudioEstimateReasonTest(CliCase):
    def facts(self, samples, rate=48000):
        path = self.dir / "short.wav"
        sf.write(path, samples, rate, subtype="FLOAT")
        data = self.evidence("--mixdown", path, mixdown=False)
        return {item["property"]: item for item in data["observations"]}

    def test_one_sample_file_is_not_called_silence(self):
        facts = self.facts(np.array([0.5], dtype="float32"))
        self.assertIsNone(facts["audio.band_energy"]["value"])
        self.assertNotIn("silence", facts["audio.band_energy"]["confidenceReason"])
        self.assertIn("too short", facts["audio.band_energy"]["confidenceReason"])
        self.assertNotIn("silence", facts["audio.spectral_centroid"]["confidenceReason"])

    def test_real_silence_keeps_its_reason(self):
        facts = self.facts(np.zeros(8, dtype="float32"))
        self.assertIn("digital silence", facts["audio.band_energy"]["confidenceReason"])

    def test_stereo_correlation_survives_extreme_scales(self):
        from music_guide.analysis import StereoMeter
        for scale in (1e-155, 1e-100, 1.0, 1e140):
            with self.subTest(scale=scale):
                meter = StereoMeter()
                block = np.full((8, 2), scale)
                meter.feed(block)
                correlation, _ = meter.result()
                self.assertAlmostEqual(correlation, 1.0, places=9)
        meter = StereoMeter()
        meter.feed(np.array([[1e-155, -1e-155]] * 4))
        self.assertAlmostEqual(meter.result()[0], -1.0, places=9)

    def test_tiny_stereo_file_reports_correlation_through_the_cli(self):
        path = self.dir / "tiny.wav"
        sf.write(path, np.full((64, 2), 1e-155), 48000, subtype="DOUBLE")
        facts = {item["property"]: item for item in self.evidence("--mixdown", path, mixdown=False)["observations"]}
        self.assertAlmostEqual(facts["audio.stereo_correlation"]["value"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
