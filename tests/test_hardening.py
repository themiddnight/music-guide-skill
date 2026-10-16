"""Regression tests for hostile or malformed input. Each test drives the CLI the way a user does."""
from __future__ import annotations

import json
import re
import struct
import zipfile
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import ROOT, SKILL, run_cli  # noqa: E402


def smf(track: bytes, ticks: int = 480) -> bytes:
    return b"MThd" + struct.pack(">IHHH", 6, 0, 1, ticks) + b"MTrk" + struct.pack(">I", len(track)) + track


END_OF_TRACK = b"\x00\xff\x2f\x00"


FIXTURE_LMMS = """<?xml version="1.0"?>
<!DOCTYPE lmms-project>
<lmms-project version="1.0" type="song" creator="LMMS" creatorversion="1.2.2">
  <head bpm="%d" timesig_numerator="%d" timesig_denominator="%d"/>
  <song><trackcontainer type="song">
    <track type="0" name="Lead" muted="0">
      <instrumenttrack basenote="57" vol="100" pitch="0" pan="0"><instrument name="tripleoscillator"/></instrumenttrack>
      <pattern pos="0" len="192" type="1" steps="16"><note key="57" len="48" pos="0" vol="100" pan="0"/></pattern>
    </track>
  </trackcontainer></song>
</lmms-project>
"""

FIXTURE_REAPER = """<REAPER_PROJECT 0.1 "7.0/x64" 1
 TEMPO %d %d %d
 MARKER 1 5 "a" 0 0 1 R {X} 0
>
"""

FIXTURE_MUSICXML = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0"><part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
<part id="P1"><measure number="1"><attributes><divisions>1</divisions><time><beats>%d</beats><beat-type>%d</beat-type></time></attributes>
<direction><sound tempo="%d"/></direction>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration></note></measure></part></score-partwise>
"""

FIXTURE_DAWPROJECT = """<Project version="1.0"><Application name="Bitwig Studio" version="5.1"/>
<Transport><Tempo unit="bpm" value="%d" id="t"/><TimeSignature numerator="%s" denominator="%d" id="ts"/></Transport>
<Structure><Track contentType="audio" id="tr2" name="Vox"><Channel role="regular" id="c2"/></Track></Structure>
<Arrangement id="a"/></Project>
"""


class CliCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.mix = self.dir / "mix.wav"
        sf.write(self.mix, np.array([0.5, -0.5, 0.25, -0.25], dtype="float32"), 48000, subtype="FLOAT")

    def run_cli(self, *args, mixdown=True):
        prefix = ["--mixdown", str(self.mix)] if mixdown else []
        return run_cli(*prefix, *args, cwd=self.dir)

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
        from music_guide.adapters import audio
        with mock.patch.object(audio, "audio", side_effect=ZeroDivisionError("boom")):
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


class MidiBoundsTest(CliCase):
    def test_oversized_midi_is_refused(self):
        big = self.write("big.mid", smf(END_OF_TRACK))
        with big.open("ab") as stream:
            stream.truncate(4 * 1024 * 1024 + 1)
        data = self.evidence("--midi", big)
        self.assert_optional_error(data)
        self.assertIn("byte limit", next(item["message"] for item in data["diagnostics"] if item["code"] == "optional_source_error"))

    def test_unmatched_note_offs_collapse_into_one_diagnostic_per_track(self):
        track = b"\x00\x80\x3c\x00" + b"\x00\x3c\x00" * 50000 + b"\x00\xff\x06\x01x"
        data = self.evidence("--midi", self.write("offs.mid", smf(track + END_OF_TRACK)))
        offs = [item for item in data["diagnostics"] if item["code"] == "unmatched_note_off"]
        self.assertEqual(len(offs), 1)
        self.assertIn("50001", offs[0]["message"])
        self.assertIn("track 0", offs[0]["message"])

    def test_unmatched_note_ons_collapse_into_one_diagnostic_per_track(self):
        track = b"".join(b"\x00\x90" + bytes([p, 0x40]) for p in range(40, 60)) + END_OF_TRACK
        data = self.evidence("--midi", self.write("ons.mid", smf(track)))
        ons = [item for item in data["diagnostics"] if item["code"] == "unmatched_note_on"]
        self.assertEqual(len(ons), 1)
        self.assertIn("20 MIDI note(s) in track 0", ons[0]["message"])

    def test_full_detail_caps_listed_notes_and_keeps_the_summary(self):
        from music_guide import cli
        from music_guide.adapters import midi
        pairs = b"".join(b"\x00\x90" + bytes([60 + n % 20, 0x40]) + b"\x10\x80" + bytes([60 + n % 20, 0]) for n in range(30))
        path = self.write("many.mid", smf(pairs + END_OF_TRACK))
        with mock.patch.object(midi, "MAX_NOTE_LISTING", 10):
            result = cli.build(None, [("midi", path)], None, "full")
        notes = [item for item in result.observations if item.property == "midi.note"]
        self.assertEqual(len(notes), 10)
        self.assertEqual(sum(item.code == "project_list_truncated" for item in result.diagnostics), 1)
        summary = next(item for item in result.observations if item.property == "midi.track_summary")
        self.assertEqual(summary.value["noteCount"], 30)

    def test_tempo_names_and_meter_lists_are_capped(self):
        tempo = b"\x00\xff\x51\x03\x07\xa1\x20" * 6000
        names = b"\x00\xff\x03\x01a" * 6000
        data = self.evidence("--midi", self.write("lists.mid", smf(tempo + names + END_OF_TRACK)))
        tempo_map = next(item["value"] for item in data["observations"] if item["property"] == "midi.tempo_map")
        self.assertEqual(len(tempo_map), 5000)
        self.assertEqual(sum(item["property"] == "midi.track_name" for item in data["observations"]), 5000)
        self.assertEqual(sum(item["code"] == "project_list_truncated" for item in data["diagnostics"]), 2)

    def test_overlapping_notes_pair_first_in_first_out(self):
        track = (b"\x00\x90\x3c\x40\x10\x90\x3c\x50\x10\x80\x3c\x00\x10\x80\x3c\x00" + END_OF_TRACK)
        data = self.evidence("--midi", self.write("fifo.mid", smf(track)), "--detail", "full")
        notes = [item["value"] for item in data["observations"] if item["property"] == "midi.note"]
        self.assertEqual([(n["velocity"], n["startTick"], n["endTick"]) for n in notes], [(64, 0, 32), (80, 16, 48)])


class InterpreterVersionTest(CliCase):
    def run_as(self, version: tuple, args: list[str]):
        script = ("import runpy, sys\n"
                  f"sys.version_info = {version!r}\n"
                  f"sys.argv = ['analyze.py', *{args!r}]\n"
                  f"runpy.run_path({str(SKILL / 'scripts' / 'analyze.py')!r}, run_name='__main__')\n")
        return subprocess.run([sys.executable, "-I", "-c", script], cwd=self.dir, text=True, capture_output=True)

    def test_old_python_gets_one_actionable_line(self):
        result = self.run_as((3, 9, 6, "final", 0), ["--mixdown", str(self.mix)])
        self.assertEqual(result.returncode, 3)
        self.assertEqual(result.stdout, "")
        self.assertNotIn("Traceback", result.stderr)
        lines = result.stderr.strip().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertIn("Python 3.10 or newer", lines[0])
        self.assertIn("3.9.6", lines[0])
        self.assertIn("uv run", lines[0])
        self.assertIn("python3.12 -m venv", lines[0])

    @unittest.skipUnless(Path("/usr/bin/python3").exists(), "no system interpreter")
    def test_real_system_interpreter_below_3_10_is_caught_before_any_import(self):
        probe = subprocess.run(["/usr/bin/python3", "-c", "import sys; print(sys.version_info >= (3, 10))"],
                               capture_output=True, text=True)
        if probe.stdout.strip() != "False":
            self.skipTest("system interpreter is 3.10 or newer")
        result = subprocess.run(["/usr/bin/python3", "-I", str(SKILL / "scripts" / "analyze.py"), "--mixdown", str(self.mix)],
                                cwd=self.dir, text=True, capture_output=True)
        self.assertEqual(result.returncode, 3)
        self.assertIn("Python 3.10 or newer", result.stderr)


class DuplicateKeyTest(CliCase):
    def test_declarations_reject_repeated_keys_like_solna(self):
        repeated = self.write("dup.json", '[{"property": "tempo", "value": 90, "value": 120, "unit": "BPM", "scope": "project"}]')
        data = self.evidence("--declarations", repeated)
        self.assertEqual(data["sources"][-1]["status"], "error")
        message = next(item["message"] for item in data["diagnostics"] if item["code"] == "optional_source_error")
        self.assertIn("duplicate JSON key: value", message)
        self.assertFalse([item for item in data["observations"] if item["provenance"] == "user_declared"])

    def test_declarations_still_accept_unique_keys_and_reject_nonfinite_literals(self):
        good = self.write("ok.json", '[{"property": "tempo", "value": 90, "unit": "BPM", "scope": "project"}]')
        data = self.evidence("--declarations", good)
        self.assertEqual([item["value"] for item in data["observations"] if item["provenance"] == "user_declared"], [90])
        bad = self.write("nan.json", '[{"property": "tempo", "value": NaN, "unit": "BPM", "scope": "project"}]')
        data = self.evidence("--declarations", bad)
        self.assertIn("nonfinite", next(item["message"] for item in data["diagnostics"] if item["code"] == "optional_source_error"))


class MessageTest(CliCase):
    def optional_message(self, *args):
        data = self.evidence(*args)
        return next(item["message"] for item in data["diagnostics"] if item["code"] == "optional_source_error")

    def test_mismatch_message_has_no_doubled_noun(self):
        message = self.optional_message("--project", self.write("fake.mid", b"not midi"))
        self.assertIn("does not match the Standard MIDI file it is named as", message)
        self.assertNotIn("file file", message)

    def test_audio_given_as_a_project_points_to_mixdown_and_stem(self):
        for name in ("take.wav", "song.FLAC", "bounce.mp3", "loop.aiff"):
            with self.subTest(name=name):
                message = self.optional_message("--project", self.write(name, b"RIFF"))
                self.assertIn("--mixdown", message)
                self.assertIn("--stem", message)

    def test_test_only_dependency_is_not_a_skill_requirement(self):
        skill = (SKILL / "requirements.txt").read_text().lower()
        self.assertNotIn("jsonschema", skill)
        dev = (ROOT / "requirements-dev.txt").read_text()
        self.assertIn("-r skills/music-guide/requirements.txt", dev)
        self.assertIn("jsonschema", dev)


class ZipLimitTest(CliCase):
    def test_embedded_audio_does_not_count_against_the_read_member(self):
        from music_guide.adapters.common import _read_zip_member
        many = self.dir / "many.zip"
        with zipfile.ZipFile(many, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("project.xml", "<Project/>")
            for index in range(2000):
                archive.writestr(f"audio/{index}.wav", b"RIFF" + bytes(64))
        self.assertEqual(_read_zip_member(many, "project.xml", 1024), b"<Project/>")
        big = self.dir / "big.zip"
        with zipfile.ZipFile(big, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("project.xml", "<Project/>")
            with archive.open("audio/take.wav", "w", force_zip64=True) as stream:
                for _ in range(70):
                    stream.write(bytes(1024 * 1024))
        self.assertLess(big.stat().st_size, 1024 * 1024)
        self.assertEqual(_read_zip_member(big, "project.xml", 1024), b"<Project/>")

    def test_dawproject_with_many_embedded_samples_is_parsed(self):
        project = self.dir / "samples.dawproject"
        with zipfile.ZipFile(project, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("project.xml", '<Project version="1"><Application name="Bitwig Studio" version="5.2"/>'
                             '<Transport><Tempo value="120" unit="bpm"/></Transport><Structure/></Project>')
            for index in range(600):
                archive.writestr(f"audio/sample-{index}.wav", b"RIFF" + bytes(32))
        data = self.evidence("--dawproject", project)
        self.assertEqual(data["sources"][-1]["status"], "ok")
        self.assertEqual([item["value"] for item in data["observations"] if item["property"] == "tempo"], [120])

    def test_entry_count_cap_still_applies(self):
        from music_guide.adapters.common import MAX_ZIP_ENTRIES, _read_zip_member
        crowded = self.dir / "crowded.zip"
        with zipfile.ZipFile(crowded, "w") as archive:
            for index in range(MAX_ZIP_ENTRIES + 1):
                archive.writestr(f"f{index}", "")
            archive.writestr("project.xml", "<a/>")
        with self.assertRaisesRegex(ValueError, "entries"):
            _read_zip_member(crowded, "project.xml", 1024)

    def test_oversized_central_directory_is_refused_before_it_is_parsed(self):
        from music_guide.adapters.common import _read_zip_member
        record = b"PK\x01\x02" + struct.pack("<HHHHHHIIIHHHHHII", 20, 20, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0) + b"a"
        directory = record * 400000
        spoof = self.dir / "spoof.zip"
        spoof.write_bytes(directory + b"PK\x05\x06" + struct.pack("<HHHHIIH", 0, 0, 1, 1, len(directory), 0, 0))
        with mock.patch.object(zipfile, "ZipFile", side_effect=AssertionError("parsed the directory")):
            with self.assertRaisesRegex(ValueError, "central directory"):
                _read_zip_member(spoof, "a", 1024)

    def test_zip64_end_record_is_read_for_the_directory_size(self):
        import io
        from music_guide.adapters.common import MAX_CENTRAL_DIRECTORY_BYTES, _central_directory_size, _read_zip_member
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("project.xml", "<a/>")
        plain = buffer.getvalue()
        directory_offset = struct.unpack_from("<I", plain, len(plain) - 6)[0]
        directory = plain[directory_offset:-22]

        def zip64(size: int) -> Path:
            record_offset = directory_offset + len(directory)
            record = b"PK\x06\x06" + struct.pack("<QHHIIQQQQ", 44, 45, 45, 0, 0, 1, 1, size, directory_offset)
            locator = b"PK\x06\x07" + struct.pack("<IQI", 0, record_offset, 1)
            end = b"PK\x05\x06" + struct.pack("<HHHHIIH", 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0)
            path = self.dir / f"z64-{size}.zip"
            path.write_bytes(plain[:directory_offset] + directory + record + locator + end)
            return path
        honest = zip64(len(directory))
        self.assertEqual(_central_directory_size(honest), len(directory))
        self.assertEqual(_read_zip_member(honest, "project.xml", 1024), b"<a/>")
        with self.assertRaisesRegex(ValueError, "central directory"):
            _read_zip_member(zip64(MAX_CENTRAL_DIRECTORY_BYTES + 1), "project.xml", 1024)


class ObservationCapTest(CliCase):
    def test_project_facts_cap_each_repeating_property_with_one_warning(self):
        from music_guide.model import Evidence
        from music_guide.project import ProjectFacts
        cases = {
            "project.plugin": lambda facts, n: facts.plugin(f"P{n}", None, "loc"),
            "project.marker": lambda facts, n: facts.marker(f"M{n}", float(n), "beats", "loc"),
            "project.track": lambda facts, n: facts.track(n, f"T{n}", "audio", "loc"),
            "project.media_reference": lambda facts, n: facts.media_references([f"/x/{n}.wav"], "loc"),
        }
        for property, add in cases.items():
            with self.subTest(property=property):
                evidence = Evidence()
                facts = ProjectFacts(evidence, "s1", "test", "summary")
                for number in range(5200):
                    add(facts, number)
                self.assertEqual(sum(item.property == property for item in evidence.observations), 5000)
                warnings = [item for item in evidence.diagnostics if item.code == "project_list_truncated"]
                self.assertEqual(len(warnings), 1)
                self.assertIn(property.split(".")[1].replace("_", " "), warnings[0].message)

    def test_cap_counts_per_source_across_every_facts_instance(self):
        from music_guide.adapters.graded import GradedFacts
        from music_guide.model import Evidence
        from music_guide.project import ProjectFacts
        evidence = Evidence()
        plain = ProjectFacts(evidence, "s1", "test", "summary")
        graded = GradedFacts(evidence, "s1", "test", "summary", grade="low", caveat="x")
        other = ProjectFacts(evidence, "s2", "test", "summary")
        for number in range(3000):
            plain.plugin(f"A{number}", None, "loc")
            graded.plugin(f"B{number}", None, "loc")
        other.plugin("C", None, "loc")
        counts = {source: sum(item.property == "project.plugin" and item.sourceIds == [source] for item in evidence.observations)
                  for source in ("s1", "s2")}
        self.assertEqual(counts, {"s1": 5000, "s2": 1})
        warnings = [item for item in evidence.diagnostics if item.code == "project_list_truncated"]
        self.assertEqual([item.sourceId for item in warnings], ["s1"])

    def test_full_detail_note_listing_is_capped_on_project_facts(self):
        from music_guide import project
        from music_guide.model import Evidence
        from music_guide.notes import Note
        evidence = Evidence()
        facts = project.ProjectFacts(evidence, "s1", "test", "full")
        notes = [Note(60 + n % 12, 64, float(n), n + 1.0) for n in range(30)]
        with mock.patch.object(project, "MAX_NOTE_LISTING", 12):
            facts.track_notes(0, notes, "loc")
        self.assertEqual(sum(item.property == "project.note" for item in evidence.observations), 12)
        summary = next(item for item in evidence.observations if item.property == "project.track_notes")
        self.assertEqual(summary.value["noteCount"], 30)
        self.assertEqual(sum(item.code == "project_list_truncated" for item in evidence.diagnostics), 1)


class XmlShapeTest(CliCase):
    def test_deep_nesting_is_refused(self):
        from music_guide.adapters.common import MAX_XML_DEPTH, _parse_xml
        shallow = b"<a>" * (MAX_XML_DEPTH - 1) + b"</a>" * (MAX_XML_DEPTH - 1)
        self.assertEqual(_parse_xml(shallow).tag, "a")
        with self.assertRaisesRegex(ValueError, "nested"):
            _parse_xml(b"<a>" * (MAX_XML_DEPTH + 1) + b"</a>" * (MAX_XML_DEPTH + 1))

    def test_element_count_is_refused_before_the_tree_grows(self):
        from music_guide.adapters import common
        with mock.patch.object(common, "MAX_XML_ELEMENTS", 100):
            self.assertEqual(len(common._parse_xml(b"<r>" + b"<a/>" * 98 + b"</r>")), 98)
            with self.assertRaisesRegex(ValueError, "elements"):
                common._parse_xml(b"<r>" + b"<a/>" * 100 + b"</r>")

    def test_parser_still_rejects_doctype_entities_and_truncation(self):
        from music_guide.adapters.common import _parse_xml
        for raw in (b'<!DOCTYPE a [<!ENTITY x "y">]><a>&x;</a>', b"<a><b></a>", b"", b"<a>"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _parse_xml(raw)

    def test_deeply_nested_project_becomes_a_source_error(self):
        nested = self.write("deep.musicxml", "<score-partwise>" + "<a>" * 400 + "</a>" * 400 + "</score-partwise>")
        self.assert_optional_error(self.evidence("--project", nested))


class PlausibleValueTest(CliCase):
    def optional_error(self, path: Path, flag="--project") -> str:
        data = self.evidence(flag, path)
        self.assert_optional_error(data)
        return next(item["message"] for item in data["diagnostics"] if item["code"] == "optional_source_error")

    def values(self, data, prop):
        return [item["value"] for item in data["observations"] if item["property"] == prop]

    def warnings_for(self, data, prop):
        return [item for item in data["diagnostics"]
                if item["code"] == "implausible_value" and re.match(rf"Skipped (\d+ of \d+ )?{re.escape(prop)} ", item["message"])]

    def test_project_facts_skip_implausible_values_with_one_warning_each(self):
        from music_guide.model import Evidence
        from music_guide.project import ProjectFacts
        evidence = Evidence()
        facts = ProjectFacts(evidence, "s1", "test", "summary")
        facts.tempo(1000, "loc")
        facts.tempo(0.5, "loc")
        for numerator, denominator in ((4, 4), (999, 128), (1, 1), (7, 8), (6, 16), (4, 3), (7, 5), (3.0, 8.0)):
            facts.meter(numerator, denominator, "loc")
        accepted = len(evidence.observations)
        self.assertEqual(accepted, 10)
        self.assertEqual(evidence.diagnostics, [])
        bad_tempos = (1000.001, 999999, 1e300, 10 ** 400, 0, -1, float("nan"), float("inf"), True, "120", None)
        bad_meters = ((1000, 4), (0, 4), (4, 0), (4, 129), (-4, 4), (10 ** 300, 4), (1e300, 4), (2.5, 4), (4, 2.5), (4, True),
                      (None, 4), ("4", 4), (float("nan"), 4))
        for bpm in bad_tempos:
            with self.subTest(bpm=bpm):
                before = len(evidence.diagnostics)
                facts.tempo(bpm, "tempo-locator")
                self.assertEqual(len(evidence.diagnostics), before + 1)
        for numerator, denominator in bad_meters:
            with self.subTest(meter=(numerator, denominator)):
                before = len(evidence.diagnostics)
                facts.meter(numerator, denominator, "meter-locator")
                self.assertEqual(len(evidence.diagnostics), before + 1)
        self.assertEqual(len(evidence.observations), accepted)
        for item in evidence.diagnostics:
            self.assertEqual((item.code, item.severity, item.sourceId), ("implausible_value", "warning", "s1"))
            self.assertLess(len(item.message), 200)
        self.assertIn("tempo", evidence.diagnostics[0].message)
        self.assertIn("tempo-locator", evidence.diagnostics[0].message)
        self.assertIn("1000.001", evidence.diagnostics[0].message)
        self.assertIn("meter", evidence.diagnostics[len(bad_tempos)].message)
        self.assertIn("meter-locator", evidence.diagnostics[len(bad_tempos)].message)

    def test_tempo_map_drops_only_implausible_points_and_warns_once(self):
        from music_guide.model import Evidence
        from music_guide.project import ProjectFacts
        evidence = Evidence()
        facts = ProjectFacts(evidence, "s1", "test", "summary")
        facts.tempo_map([{"beat": 0, "bpm": 120}, {"beat": 4, "bpm": 5000}, {"beat": 8, "bpm": 90},
                         {"beat": 12, "bpm": -3}, {"beat": float("nan"), "bpm": 100}], "map")
        self.assertEqual([item.value for item in evidence.observations],
                         [[{"beat": 0.0, "bpm": 120.0}, {"beat": 8.0, "bpm": 90.0}]])
        warnings = [item for item in evidence.diagnostics if item.code == "implausible_value"]
        self.assertEqual(len(warnings), 1)
        self.assertIn("3 of 5", warnings[0].message)
        self.assertIn("project.tempo_map", warnings[0].message)
        facts.tempo_map([{"beat": 0, "bpm": 5000}], "all-bad")
        self.assertEqual(len(evidence.observations), 1)
        self.assertEqual(sum(item.code == "implausible_value" for item in evidence.diagnostics), 2)

    def test_plausibility_predicates_use_the_exported_limits(self):
        from music_guide import project
        from music_guide.project import plausible_meter, plausible_tempo
        self.assertEqual((project.MAX_TEMPO_BPM, project.MAX_METER_NUMERATOR, project.MAX_METER_DENOMINATOR),
                         (1000.0, 999, 128))
        self.assertTrue(plausible_tempo(120) and plausible_tempo(1000.0) and plausible_tempo(10 ** 3))
        self.assertFalse(any(plausible_tempo(value) for value in (0, -3, 1001, 10 ** 400, float("inf"), float("nan"), True, "120")))
        self.assertTrue(all(plausible_meter(*pair) for pair in ((3, 4), (3.0, 8.0), (4, 3), (7, 5), (999, 128), (1, 1))))
        self.assertFalse(any(plausible_meter(*pair) for pair in (
            (0, 4), (1000, 4), (3, 129), (3, 0), (-3, 4), (2.5, 4), (True, 4), (3, None), (10 ** 300, 4), (1e300, 4))))

    def test_unusual_but_real_values_keep_every_project_in_every_format(self):
        lmms = (self.dir / "a.mmp")
        lmms.write_text((FIXTURE_LMMS % (140, 3, 3)))
        reaper = self.write("a.rpp", FIXTURE_REAPER % (120, 7, 5))
        cases = {
            "lmms 3/3": (lmms, "3/3", "project.track"),
            "reaper 7/5": (reaper, "7/5", "project.marker"),
            "musicxml 4/3": (self.write("a.musicxml", FIXTURE_MUSICXML % (4, 3, 120)), "4/3", "project.track_notes"),
            "musicxml 100/4": (self.write("b.musicxml", FIXTURE_MUSICXML % (100, 4, 120)), "100/4", "project.track_notes"),
            "dawproject 7/5": (self.dawproject(7, 5, 120), "7/5", "project.track"),
        }
        for name, (path, meter, kept) in cases.items():
            with self.subTest(name=name):
                data = self.evidence("--dawproject" if path.suffix == ".dawproject" else "--project", path)
                self.assertEqual(data["sources"][-1]["status"], "ok", data["diagnostics"])
                self.assertEqual(self.values(data, "meter"), [meter])
                self.assertTrue(self.values(data, kept))
                self.assertEqual([item for item in data["diagnostics"] if item["code"] == "implausible_value"], [])

    def test_implausible_tempo_or_meter_is_skipped_and_the_rest_of_the_project_survives(self):
        cases = {
            "lmms tempo": (self.write("t.mmp", FIXTURE_LMMS % (5000, 3, 4)), "tempo", "project.track"),
            "lmms meter": (self.write("m.mmp", FIXTURE_LMMS % (140, 1000, 4)), "meter", "project.track"),
            "reaper tempo": (self.write("t.rpp", FIXTURE_REAPER % (5000, 4, 4)), "tempo", "project.marker"),
            "reaper meter": (self.write("m.rpp", FIXTURE_REAPER % (120, 4, 200)), "meter", "project.marker"),
            "musicxml tempo": (self.write("t.musicxml", FIXTURE_MUSICXML % (4, 4, 1200)), "tempo", "project.track_notes"),
            "musicxml meter": (self.write("m.musicxml", FIXTURE_MUSICXML % (1000, 4, 120)), "meter", "project.track_notes"),
            "musicxml beat type": (self.write("b.musicxml", FIXTURE_MUSICXML % (4, 129, 120)), "meter", "project.track_notes"),
            "dawproject tempo": (self.dawproject(4, 4, 5000), "tempo", "project.track"),
            "dawproject meter": (self.dawproject(4, 129, 120), "meter", "project.track"),
            "dawproject zero numerator": (self.dawproject(0, 4, 120), "meter", "project.track"),
            "dawproject fractional": (self.dawproject(2.5, 4, 120), "meter", "project.track"),
            "lmms fractional": (self.write("f.mmp", FIXTURE_LMMS.replace('timesig_numerator="%d"', 'timesig_numerator="2.5"')
                                           % (140, 4)), "meter", "project.track"),
        }
        for name, (path, prop, kept) in cases.items():
            with self.subTest(name=name):
                data = self.evidence("--dawproject" if path.suffix == ".dawproject" else "--project", path)
                self.assertEqual(data["sources"][-1]["status"], "ok", data["diagnostics"])
                self.assertEqual(self.values(data, prop), [])
                self.assertTrue(self.values(data, kept))
                self.assertEqual(len(self.warnings_for(data, prop)), 1, data["diagnostics"])

    def test_implausible_tempo_points_are_dropped_from_the_tempo_map_not_the_project(self):
        score = FIXTURE_MUSICXML % (4, 4, 120)
        note = "<note><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration></note>"
        score = score.replace("</measure>", f'</measure><measure number="2"><direction><sound tempo="1200"/></direction>{note}</measure>'
                              f'<measure number="3"><direction><sound tempo="90"/></direction>{note}</measure>')
        data = self.evidence("--project", self.write("map.musicxml", score))
        self.assertEqual(data["sources"][-1]["status"], "ok", data["diagnostics"])
        self.assertEqual([[point["bpm"] for point in value] for value in self.values(data, "project.tempo_map")], [[120.0, 90.0]])
        self.assertEqual(len(self.warnings_for(data, "project.tempo_map")), 1)

    def dawproject(self, numerator, denominator, tempo) -> Path:
        path = self.dir / f"t{numerator}-{denominator}-{tempo}.dawproject"
        numerator = f"{numerator:g}"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("project.xml", FIXTURE_DAWPROJECT % (tempo, numerator, denominator))
        return path

    def test_flp_with_implausible_meter_keeps_the_rest_of_the_project(self):
        def event(event_id: int, payload: bytes) -> bytes:
            return bytes([event_id]) + payload
        events = [bytes([199, 12]) + b"20.8.3.2304\0", event(156, struct.pack("<I", 128000)), event(17, bytes([0])),
                  event(18, bytes([4]))]
        body = b"".join(events)
        path = self.write("a.flp", b"FLhd" + struct.pack("<IhHH", 6, 0, 2, 96) + b"FLdt" + struct.pack("<I", len(body)) + body)
        data = self.evidence("--project", path)
        self.assertEqual(data["sources"][-1]["status"], "ok", data["diagnostics"])
        self.assertEqual(self.values(data, "tempo"), [128.0])
        self.assertEqual(self.values(data, "meter"), [])
        self.assertEqual(self.values(data, "flstudio.time_signature"), [{"numerator": 0, "beat": 4}])
        self.assertEqual(len(self.warnings_for(data, "meter")), 1, data["diagnostics"])

    def test_studio_one_skips_implausible_tempo_and_meter_segments(self):
        song = ('<?xml version="1.0" encoding="UTF-8"?><Song xmlns:x="urn:presonus"><Attributes x:id="Root">'
                '<Attributes x:id="timeContext" sampleRate="44100">'
                '<TempoMap x:id="tempoMap"><TempoMapSegment curveType="0" start="0" tempo="0.0001"/></TempoMap>'
                '<TimeSignatureMap x:id="timeSignatureMap"><TimeSignatureMapSegment start="0" numerator="3" denominator="256"/>'
                '</TimeSignatureMap></Attributes></Attributes></Song>')
        path = self.dir / "a.song"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("Song/song.xml", song)
            archive.writestr("metainfo.xml", "<MetaInformation/>")
        data = self.evidence("--project", path)
        self.assertEqual(data["sources"][-1]["status"], "ok", data["diagnostics"])
        self.assertEqual(self.values(data, "tempo") + self.values(data, "meter"), [])


class StartupCostTest(CliCase):
    def loaded(self, *args) -> set[str]:
        script = ("import sys\n"
                  f"sys.path.insert(0, {str(SKILL / 'scripts')!r})\n"
                  "from music_guide.cli import main\n"
                  f"main({[*map(str, args)]!r})\n"
                  "print('LOADED:' + ','.join(name for name in ('scipy', 'soundfile', 'numpy') if name in sys.modules), file=sys.stderr)\n")
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=self.dir)
        self.assertEqual(result.returncode, 0, result.stderr)
        line = next(item for item in result.stderr.splitlines() if item.startswith("LOADED:"))
        return set(filter(None, line.removeprefix("LOADED:").split(",")))

    def test_midi_only_runs_skip_the_audio_stack(self):
        midi = self.write("a.mid", smf(b"\x00\x90\x3c\x40\x10\x80\x3c\x00" + END_OF_TRACK))
        loaded = self.loaded("--midi", midi)
        self.assertNotIn("scipy", loaded)
        self.assertNotIn("soundfile", loaded)

    def test_audio_runs_still_load_it(self):
        self.assertIn("scipy", self.loaded("--mixdown", self.mix))

    def test_missing_audio_dependency_is_still_reported_before_any_work(self):
        script = ("import sys, importlib.util\n"
                  "real = importlib.util.find_spec\n"
                  "importlib.util.find_spec = lambda name, *a: None if name == 'scipy' else real(name, *a)\n"
                  "import runpy\n"
                  f"sys.argv = ['analyze.py', '--midi', {str(self.dir / 'a.mid')!r}]\n"
                  f"runpy.run_path({str(SKILL / 'scripts' / 'analyze.py')!r}, run_name='__main__')\n")
        self.write("a.mid", smf(END_OF_TRACK))
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 3)
        self.assertIn("missing Python module 'scipy'", result.stderr)
        self.assertEqual(len(result.stderr.strip().splitlines()), 1)

    def test_pyaaf2_is_required_only_for_aaf_files(self):
        script = ("import sys, importlib.util\n"
                  "real = importlib.util.find_spec\n"
                  "importlib.util.find_spec = lambda name, *a: None if name == 'aaf2' else real(name, *a)\n"
                  "import runpy\n"
                  f"sys.argv = ['analyze.py', '--midi', {str(self.dir / 'a.mid')!r}]\n"
                  f"runpy.run_path({str(SKILL / 'scripts' / 'analyze.py')!r}, run_name='__main__')\n")
        self.write("a.mid", smf(END_OF_TRACK))
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("missing Python module", result.stderr)


if __name__ == "__main__":
    unittest.main()
