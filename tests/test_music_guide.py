from __future__ import annotations

import gzip
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import jsonschema
import mido
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "music-guide"
sys.path.insert(0, str(SKILL / "scripts"))

from music_guide.adapters.common import _gunzip_bounded, _media_basename, _read_zip_member  # noqa: E402
from music_guide.model import Evidence, detect_conflicts  # noqa: E402
from music_guide.notes import Note  # noqa: E402
from music_guide.project import ProjectFacts  # noqa: E402

SCHEMA = json.loads((SKILL / "schemas" / "evidence.schema.json").read_text())
DECLARATIONS_SCHEMA = json.loads((SKILL / "schemas" / "declarations.schema.json").read_text())


class MusicGuideTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.mix = self.dir / "mix.wav"
        sf.write(self.mix, np.array([0.5, -0.5, 0.25, -0.25], dtype="float32"), 48000, subtype="FLOAT")

    def run_cli(self, *args, script=None, mixdown=True):
        prefix = ["--mixdown", str(self.mix)] if mixdown else []
        return subprocess.run([sys.executable, str(script or SKILL / "scripts" / "analyze.py"),
                               *prefix, *map(str, args)], cwd=self.dir,
                              text=True, capture_output=True)

    def evidence(self, *args, mixdown=True):
        result = self.run_cli(*args, mixdown=mixdown)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        data = json.loads(result.stdout)
        jsonschema.validate(data, SCHEMA)
        source_ids = {item["id"] for item in data["sources"]}
        observation_ids = {item["id"] for item in data["observations"]}
        self.assertTrue(all(set(item["sourceIds"]) <= source_ids for item in data["observations"]))
        self.assertTrue(all(set(item["observationIds"]) <= observation_ids for item in data["conflicts"]))
        return data

    def test_audio_metrics_silence_and_stem_scope(self):
        data = self.evidence()
        facts = {item["property"]: item for item in data["observations"]}
        self.assertAlmostEqual(facts["audio.sample_peak"]["value"], -6.0205999, places=5)
        self.assertEqual(facts["audio.sample_peak"]["unit"], "dBFS")
        self.assertEqual(facts["audio.rms"]["provenance"], "measured")
        self.assertEqual(facts["audio.sample_rate"]["value"], 48000)
        stem = self.dir / "stem.wav"
        sf.write(stem, np.zeros(4), 48000, subtype="PCM_16")
        data = self.evidence("--stem", stem)
        stem_facts = [item for item in data["observations"] if item["scope"].startswith("stem:")]
        self.assertEqual(len(stem_facts), 6)
        self.assertIsNone(next(item["value"] for item in stem_facts if item["property"] == "audio.rms"))
        self.assertTrue(any(item["code"] == "digital_silence" for item in data["diagnostics"]))
        self.assertFalse(data["conflicts"])
        sf.write(self.mix, np.array([1.2, -0.2], dtype="float32"), 48000, subtype="FLOAT")
        data = self.evidence()
        self.assertEqual(next(item["value"] for item in data["observations"]
                              if item["property"] == "audio.samples_at_or_above_full_scale"), 1)
        self.assertTrue(any(item["code"] == "possible_clipping" for item in data["diagnostics"]))

    def test_midi_events_notes_and_unsupported_type(self):
        midi_path = self.dir / "notes.mid"
        song = mido.MidiFile(type=1, ticks_per_beat=480)
        track = mido.MidiTrack([mido.MetaMessage("track_name", name="Piano"),
                                mido.MetaMessage("set_tempo", tempo=500000),
                                mido.Message("note_on", note=60, velocity=90),
                                mido.Message("note_off", note=60, velocity=0, time=480),
                                mido.MetaMessage("set_tempo", tempo=600000)])
        song.tracks.append(track)
        song.save(midi_path)
        data = self.evidence("--midi", midi_path, "--detail", "full")
        notes = [item for item in data["observations"] if item["property"] == "midi.note"]
        self.assertEqual(notes[0]["value"]["endBeat"], 1)
        tempo_map = [item for item in data["observations"] if item["property"] == "midi.tempo_map"]
        self.assertEqual([item["value"] for item in tempo_map],
                         [[{"tick": 0, "beat": 0, "bpm": 120}, {"tick": 480, "beat": 1, "bpm": 100}]])
        self.assertFalse(any(item["property"] == "tempo" for item in data["observations"]))
        self.assertFalse(data["conflicts"])
        mido.MidiFile(type=2, ticks_per_beat=480, tracks=[track]).save(midi_path)
        data = self.evidence("--midi", midi_path)
        self.assertEqual(data["sources"][1]["status"], "error")
        self.assertFalse(any(item["sourceIds"] == ["s2"] for item in data["observations"]))

    def test_distinct_midi_parts_do_not_conflict(self):
        paths = []
        for index, ppq in enumerate((480, 960)):
            path = self.dir / f"part{index}.mid"
            song = mido.MidiFile(type=1, ticks_per_beat=ppq)
            track = mido.MidiTrack([mido.MetaMessage("track_name", name=f"Part {index}")])
            for note in range(index + 1):
                track.extend([mido.Message("note_on", note=60 + note, velocity=80),
                              mido.Message("note_off", note=60 + note, velocity=0, time=ppq)])
            song.tracks.append(track)
            song.save(path)
            paths.append(path)
        data = self.evidence("--midi", paths[0], "--midi", paths[1])
        self.assertFalse(data["conflicts"])

    def test_solna_conflict_and_invalid_optional_is_atomic(self):
        path = self.dir / "song.solna"
        path.write_text(json.dumps({"formatVersion": 12, "content": {"bpm": 120, "meterId": "4/4",
            "loops": [{"scaleRoot": "C", "scaleType": "major", "chords": [
                {"root": "C", "quality": "maj", "bars": 1, "bassNote": None}]}]}}))
        declarations = self.dir / "declarations.json"
        declarations.write_text(json.dumps([{"property": "tempo", "value": 125, "unit": "BPM", "scope": "project"}]))
        jsonschema.validate(json.loads(declarations.read_text()), DECLARATIONS_SCHEMA)
        data = self.evidence("--solna", path, "--declarations", declarations)
        self.assertEqual(data["conflicts"][0]["property"], "tempo")
        self.assertEqual(len(data["conflicts"][0]["observationIds"]), 2)
        chord = next(item for item in data["observations"] if item["property"] == "solna.chord")
        self.assertIsNone(chord["value"]["bassNote"])
        self.assertEqual(chord["provenance"], "extracted")
        declarations.write_text(json.dumps([{"property": "tempo", "value": 120, "unit": "BPM", "scope": "project"}]))
        self.assertFalse(self.evidence("--solna", path, "--declarations", declarations)["conflicts"])
        path.write_text('{"formatVersion":13,"content":{}}')
        data = self.evidence("--solna", path)
        self.assertEqual(data["sources"][1]["status"], "error")
        self.assertFalse(any(item["sourceIds"] == ["s2"] for item in data["observations"]))
        path.write_text('{"formatVersion":12,"content":{"bpm":NaN}}')
        self.assertEqual(self.evidence("--solna", path)["sources"][1]["status"], "error")
        declarations.write_text(json.dumps([{"property": "tempo", "value": None, "unit": "BPM", "scope": "project"}]))
        self.assertEqual(self.evidence("--declarations", declarations)["sources"][1]["status"], "error")

    def test_dawproject_transport_only_and_xml_safety(self):
        project = self.dir / "song.dawproject"
        with zipfile.ZipFile(project, "w") as archive:
            archive.writestr("project.xml", '<Project version="1"><Application name="Bitwig Studio" version="5.2"/><Transport><Tempo value="120" unit="bpm"/><TimeSignature numerator="4" denominator="4"/></Transport><Structure><Track name="Drums" contentType="audio"><Channel role="regular"/></Track><Track name="Bass" contentType="notes"/><Track name="FX" contentType="audio"><Channel role="effect"/></Track><Track contentType="audio notes"><Channel role="master"/></Track></Structure><Arrangement><Tempo value="180"/></Arrangement></Project>')
        data = self.evidence("--dawproject", project)
        tempos = [item for item in data["observations"] if item["property"] == "tempo"]
        self.assertEqual([item["value"] for item in tempos], [120])
        self.assertEqual(next(item["value"] for item in data["observations"] if item["property"] == "project.track_count"), 4)
        self.assertEqual(next(item["value"] for item in data["observations"] if item["property"] == "project.application"), "Bitwig Studio 5.2")
        tracks = {item["scope"]: item["value"] for item in data["observations"] if item["property"] == "project.track"}
        self.assertEqual(tracks, {
            "project_track:0": {"name": "Drums", "kind": "audio", "index": 0},
            "project_track:1": {"name": "Bass", "kind": "instrument", "index": 1},
            "project_track:2": {"name": "FX", "kind": "return", "index": 2},
            "project_track:3": {"name": None, "kind": "master", "index": 3},
        })
        with zipfile.ZipFile(project, "w") as archive:
            archive.writestr("project.xml", '<?xml version="1.0" encoding="utf-16"?><!DOCTYPE Project [<!ENTITY x "bad">]><Project/>'.encode("utf-16"))
        data = self.evidence("--dawproject", project)
        self.assertEqual(data["sources"][1]["status"], "error")

    def test_required_error_unsupported_directory_and_installed_payload(self):
        missing = self.dir / "missing.wav"
        self.assertEqual(self.evidence("--midi", missing)["sources"][1]["status"], "error")
        native = self.dir / "song.logicx"
        native.mkdir()
        data = self.evidence("--project", native)
        self.assertEqual(data["sources"][1]["status"], "error")
        self.assertIsNone(data["sources"][1]["sha256"])
        self.assertTrue(any("Alternatives/NNN/MetaData.plist" in item["message"] for item in data["diagnostics"]))
        self.assertEqual(self.evidence("--project", self.dir / "missing.logicx")["sources"][1]["status"], "error")
        target = self.dir / "outside" / "music-guide"
        install = subprocess.run([sys.executable, str(ROOT / "install.py"), "--dest", str(target)],
                                 text=True, capture_output=True)
        self.assertEqual(install.returncode, 0, install.stderr)
        self.assertEqual(subprocess.run([sys.executable, str(ROOT / "install.py"), "--dest", str(target)],
                                        capture_output=True).returncode, 0)
        data = self.run_cli(script=target / "scripts" / "analyze.py")
        self.assertEqual(data.returncode, 0, data.stderr)
        self.assertEqual(json.loads(data.stdout)["schemaVersion"], "1.1.0")
        self.assertEqual(subprocess.run([sys.executable, str(ROOT / "install.py"), "--dest", str(target)],
                                        capture_output=True).returncode, 0)
        (target / "SKILL.md").write_text("changed")
        collision = subprocess.run([sys.executable, str(ROOT / "install.py"), "--dest", str(target)], capture_output=True)
        self.assertEqual(collision.returncode, 2)
        self.assertEqual((target / "SKILL.md").read_text(), "changed")
        self.mix.unlink()
        self.assertEqual(self.run_cli().returncode, 2)

    def test_skill_routes_all_intents_and_domains(self):
        skill = (SKILL / "SKILL.md").read_text().lower()
        intents = (SKILL / "references" / "intents.md").read_text().lower()
        domains = (SKILL / "references" / "domains.md").read_text().lower()
        for word in ("analyze", "explain", "critique", "recommend", "compare", "create"):
            self.assertIn(word, skill)
            self.assertIn(word, intents)
        for word in ("theory", "composition", "arrangement", "production", "mixing", "mastering", "mood", "genre", "artwork"):
            self.assertIn(word, skill)
            self.assertIn(word, domains)

    def test_unreadable_optional_payloads_preserve_mixdown(self):
        project = self.dir / "unsupported-compression.dawproject"
        with zipfile.ZipFile(project, "w") as archive:
            archive.writestr("project.xml", '<Project version="1.0"/>')
        raw = bytearray(project.read_bytes())
        raw[8:10] = (99).to_bytes(2, "little")
        central = raw.index(b"PK\x01\x02")
        raw[central + 10:central + 12] = (99).to_bytes(2, "little")
        project.write_bytes(raw)
        corrupt = self.dir / "corrupt-deflate.dawproject"
        with zipfile.ZipFile(corrupt, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("project.xml", '<Project version="1.0"/>')
        raw = bytearray(corrupt.read_bytes())
        offset = 30 + int.from_bytes(raw[26:28], "little") + int.from_bytes(raw[28:30], "little")
        raw[offset] = 0xff
        corrupt.write_bytes(raw)
        for flag, path in [("--dawproject", project), ("--dawproject", corrupt),
                           ("--solna", self.dir / "large-number.solna"),
                           ("--solna", self.dir / "deep.solna"),
                           ("--declarations", self.dir / "deep-declarations.json")]:
            if path.name == "large-number.solna":
                path.write_text('{"formatVersion":12,"content":{"bpm":' + "9" * 400 + ',"meterId":"4/4","loops":[]}}')
            elif path.name.startswith("deep"):
                path.write_text("[" * 1500 + "0" + "]" * 1500)
            with self.subTest(path=path.name):
                data = self.evidence(flag, path)
                self.assertEqual(data["sources"][1]["status"], "error")
                self.assertEqual(len(data["observations"]), 6)

    def test_dawproject_missing_structure_and_tempo_units(self):
        project = self.dir / "partial.dawproject"
        for xml, expected in [
            ('<Project version="1.0"/>', "missing"),
            ('<Project version="1.0"><Structure/></Project>', "empty"),
            ('<Project version="1.0"><Transport><Tempo value="0.5" unit="seconds"/></Transport></Project>', "bad_unit"),
            ('<Project version="1.0"><Transport><Tempo value="120"/></Transport></Project>', "missing_unit"),
        ]:
            with zipfile.ZipFile(project, "w") as archive:
                archive.writestr("project.xml", xml)
            with self.subTest(expected=expected):
                data = self.evidence("--dawproject", project)
                counts = [item for item in data["observations"] if item["property"] == "project.track_count"]
                if expected == "missing":
                    self.assertEqual(counts, [])
                elif expected == "empty":
                    self.assertEqual([item["value"] for item in counts], [0])
                else:
                    self.assertFalse(any(item["property"] == "tempo" for item in data["observations"]))
                    self.assertEqual(data["sources"][1]["status"], "error")


    def test_audio_precision_and_diagnostics(self):
        for subtype in ("PCM_U8", "PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"):
            with self.subTest(subtype=subtype):
                sf.write(self.mix, np.full((32, 2), 0.5), 48000, subtype=subtype)
                data = self.evidence()
                facts = {item["property"]: item["value"] for item in data["observations"]}
                self.assertEqual(facts["audio.channels"], 2)
                self.assertAlmostEqual(facts["audio.rms"], -6.020599913, places=7)
        sf.write(self.mix, np.array([-1.0, 1.0]), 48000, subtype="FLOAT")
        data = self.evidence()
        self.assertEqual(next(item["value"] for item in data["observations"]
                              if item["property"] == "audio.samples_at_or_above_full_scale"), 2)
        self.assertTrue(any(item["code"] == "possible_clipping" for item in data["diagnostics"]))
        for values in (np.array([float("nan")]), np.array([1e-300])):
            sf.write(self.mix, values, 48000, subtype="DOUBLE")
            result = self.run_cli()
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")

    def test_installer_dry_run_and_symlink_collision(self):
        target = self.dir / "not-created" / "music-guide"
        result = subprocess.run([sys.executable, str(ROOT / "install.py"), "--dest", str(target), "--dry-run"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(target.parent.exists())
        target.parent.mkdir()
        target.symlink_to(SKILL, target_is_directory=True)
        result = subprocess.run([sys.executable, str(ROOT / "install.py"), "--dest", str(target)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertTrue(target.is_symlink())

    def test_wave_extensible_pcm_and_float(self):
        for subtype in ("PCM_24", "FLOAT"):
            with self.subTest(subtype=subtype):
                sf.write(self.mix, np.full((32, 2), 0.5), 48000, format="WAVEX", subtype=subtype)
                data = self.evidence()
                facts = {item["property"]: item["value"] for item in data["observations"]}
                self.assertEqual(facts["audio.channels"], 2)
                self.assertAlmostEqual(facts["audio.rms"], -6.020599913, places=7)

    def write_midi(self, name, *tracks, ticks_per_beat=480):
        path = self.dir / name
        song = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
        song.tracks.extend(mido.MidiTrack(track) for track in tracks)
        song.save(path)
        return path

    def test_mixdown_is_optional_but_some_input_is_required(self):
        declarations = self.dir / "declarations.json"
        declarations.write_text(json.dumps([{"property": "tempo", "value": 120, "unit": "BPM", "scope": "project"}]))
        for args in ((), ("--declarations", declarations)):
            with self.subTest(args=args):
                result = self.run_cli(*args, mixdown=False)
                self.assertEqual(result.returncode, 2)
                self.assertIn("usage:", result.stderr)
                self.assertEqual(result.stdout, "")
        midi_path = self.write_midi("only.mid", [mido.Message("note_on", note=60, velocity=80),
                                                 mido.Message("note_off", note=60, time=480)])
        data = self.evidence("--midi", midi_path, mixdown=False)
        self.assertEqual([item["role"] for item in data["sources"]], ["midi"])
        self.assertFalse(any(item["property"].startswith("audio.") for item in data["observations"]))
        notice = [item for item in data["diagnostics"] if item["code"] == "no_mixdown"]
        self.assertEqual(len(notice), 1)
        self.assertEqual(notice[0]["severity"], "info")
        self.assertIn("rendered", notice[0]["message"])
        with_mix = self.evidence("--midi", midi_path)
        self.assertFalse(any(item["code"] == "no_mixdown" for item in with_mix["diagnostics"]))
        self.mix.unlink()
        self.assertEqual(self.run_cli("--midi", midi_path).returncode, 2)

    def test_midi_summary_describes_each_track(self):
        keys = [mido.MetaMessage("track_name", name="Keys"),
                mido.Message("note_on", note=50, velocity=60),
                mido.Message("note_on", note=54, velocity=100),
                mido.Message("note_off", note=50, time=480),
                mido.Message("note_off", note=54, time=480),
                mido.Message("note_on", note=57, velocity=80),
                mido.Message("note_off", note=57, time=480)]
        drums = [mido.Message("note_on", channel=9, note=36, velocity=127, time=960),
                 mido.Message("note_off", channel=9, note=36, time=240)]
        path = self.write_midi("summary.mid", [mido.MetaMessage("track_name", name="Conductor")], keys, drums)
        data = self.evidence("--midi", path, mixdown=False)
        self.assertFalse(any(item["property"] == "midi.note" for item in data["observations"]))
        summaries = {item["scope"]: item for item in data["observations"] if item["property"] == "midi.track_summary"}
        self.assertEqual(sorted(summaries), ["midi_track:1", "midi_track:2"])
        zeros = dict.fromkeys(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"], 0)
        self.assertEqual(summaries["midi_track:1"]["value"], {
            "noteCount": 3,
            "pitchMin": {"midi": 50, "name": "D3"}, "pitchMax": {"midi": 57, "name": "A3"},
            "pitchClassCounts": {**zeros, "D": 1, "F#": 1, "A": 1},
            "pitchClassBeats": {**zeros, "D": 1, "F#": 2, "A": 1},
            "velocityMin": 60, "velocityMax": 100, "velocityMean": 80,
            "startBeat": 0, "endBeat": 3, "maxPolyphony": 2, "channels": [0],
        })
        drum = summaries["midi_track:2"]["value"]
        self.assertEqual((drum["channels"], drum["startBeat"], drum["endBeat"], drum["pitchMin"]["name"]),
                         ([9], 2, 2.5, "C2"))
        self.assertEqual(next(item["value"] for item in data["observations"] if item["property"] == "midi.note_count"), 4)
        full = self.evidence("--midi", path, "--detail", "full", mixdown=False)
        self.assertEqual(len([item for item in full["observations"] if item["property"] == "midi.note"]), 4)
        self.assertEqual(len([item for item in full["observations"] if item["property"] == "midi.track_summary"]), 2)

    def test_summary_detail_keeps_large_midi_output_small(self):
        track = []
        for index in range(3000):
            track.extend([mido.Message("note_on", note=48 + index % 24, velocity=64 + index % 32),
                          mido.Message("note_off", note=48 + index % 24, time=120)])
        path = self.write_midi("large.mid", track)
        summary = self.run_cli("--midi", path, mixdown=False)
        self.assertEqual(summary.returncode, 0, summary.stderr)
        self.assertLess(len(summary.stdout.encode()), 30 * 1024)
        self.assertEqual(next(item["value"] for item in json.loads(summary.stdout)["observations"]
                              if item["property"] == "midi.note_count"), 3000)
        full = self.evidence("--midi", path, "--detail", "full", mixdown=False)
        notes = [item["value"] for item in full["observations"] if item["property"] == "midi.note"]
        self.assertEqual(len(notes), 3000)
        self.assertEqual(notes[-1]["endBeat"], 750)
        self.assertEqual(self.run_cli("--midi", path, "--detail", "everything", mixdown=False).returncode, 2)

    def declare(self, *facts):
        path = self.dir / "declarations.json"
        path.write_text(json.dumps([{"property": prop, "value": value, "unit": unit, "scope": "project"}
                                    for prop, value, unit in facts]))
        return path

    def test_single_midi_tempo_and_meter_become_static_claims(self):
        conductor = [mido.MetaMessage("set_tempo", tempo=652174),
                     mido.MetaMessage("time_signature", numerator=3, denominator=4)]
        path = self.write_midi("constant.mid", conductor, [mido.Message("note_on", note=60, velocity=80),
                                                           mido.Message("note_off", note=60, time=480)])
        data = self.evidence("--midi", path, mixdown=False)
        tempo = next(item for item in data["observations"] if item["property"] == "tempo")
        self.assertAlmostEqual(tempo["value"], 92, places=3)
        self.assertEqual((tempo["timeDomain"], tempo["scope"], tempo["method"]), ("static", "project", "midi-constant-tempo"))
        self.assertIn("tick=0", tempo["locator"])
        self.assertIn("only", tempo["confidenceReason"])
        meter = next(item for item in data["observations"] if item["property"] == "meter")
        self.assertEqual((meter["value"], meter["timeDomain"], meter["method"]), ("3/4", "static", "midi-constant-meter"))
        agree = self.evidence("--midi", path, "--declarations", self.declare(("tempo", 92, "BPM"), ("meter", "3/4", "signature")), mixdown=False)
        self.assertEqual(agree["conflicts"], [])
        disagree = self.evidence("--midi", path, "--declarations", self.declare(("tempo", 95, "BPM"), ("meter", "4/4", "signature")), mixdown=False)
        self.assertEqual(sorted(item["property"] for item in disagree["conflicts"]), ["meter", "tempo"])
        self.assertIn(tempo["id"], next(item for item in disagree["conflicts"] if item["property"] == "tempo")["observationIds"])

    def test_changing_or_late_midi_tempo_makes_no_static_claim(self):
        cases = {
            "late": ([mido.MetaMessage("set_tempo", tempo=500000, time=480)], []),
            "two_tracks": ([mido.MetaMessage("set_tempo", tempo=500000)], [mido.MetaMessage("set_tempo", tempo=600000, time=960)]),
        }
        for name, tracks in cases.items():
            with self.subTest(name=name):
                path = self.write_midi(f"{name}.mid", *tracks)
                data = self.evidence("--midi", path, "--declarations", self.declare(("tempo", 125, "BPM")), mixdown=False)
                self.assertFalse(any(item["property"] == "tempo" and item["provenance"] == "extracted"
                                     for item in data["observations"]))
                self.assertEqual(data["conflicts"], [])
                tempo_map = next(item["value"] for item in data["observations"] if item["property"] == "midi.tempo_map")
                self.assertEqual(tempo_map[-1]["bpm"], 100 if name == "two_tracks" else 120)
        path = self.write_midi("meters.mid", [mido.MetaMessage("time_signature", numerator=4, denominator=4),
                                              mido.MetaMessage("time_signature", numerator=7, denominator=8, time=1920)])
        data = self.evidence("--midi", path, mixdown=False)
        self.assertFalse(any(item["property"] == "meter" for item in data["observations"]))
        self.assertEqual(next(item["value"] for item in data["observations"] if item["property"] == "midi.meter_map"),
                         [{"tick": 0, "beat": 0, "meter": "4/4"}, {"tick": 1920, "beat": 4, "meter": "7/8"}])

    def test_conflicts_use_tempo_tolerance_and_ignore_inferred_values(self):
        def tempo_conflicts(*claims):
            evidence = Evidence()
            for provenance, value in claims:
                evidence.add("s1", "tempo", value, "BPM", "project", "static", "here", "test", provenance, "low", "test")
            detect_conflicts(evidence)
            return evidence.conflicts
        self.assertEqual(tempo_conflicts(("extracted", 120.0), ("user_declared", 120.009)), [])
        self.assertEqual(len(tempo_conflicts(("extracted", 120.0), ("user_declared", 120.02))), 1)
        self.assertEqual(tempo_conflicts(("inferred", 96.0), ("extracted", 120.0), ("user_declared", 120)), [])
        self.assertEqual(tempo_conflicts(("inferred", 96.0), ("measured", 120.0)), [])
        self.assertEqual(len(tempo_conflicts(("inferred", 96.0), ("measured", 120.0), ("extracted", 100))[0].observationIds), 2)

    def test_shared_archive_helpers_refuse_bombs_and_never_keep_paths(self):
        payload = b"<Project/>" * 1000
        self.assertEqual(_gunzip_bounded(gzip.compress(payload), len(payload)), payload)
        bomb = gzip.compress(b"\0" * (4 * 1024 * 1024))
        self.assertLess(len(bomb), 16 * 1024)
        for raw in (bomb, gzip.compress(payload)[:-12], b"not gzip"):
            with self.assertRaises(ValueError):
                _gunzip_bounded(raw, 1024 * 1024)
        archive_path = self.dir / "many.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            for index in range(300):
                archive.writestr(f"f{index}.txt", "x")
        with self.assertRaises(ValueError):
            _read_zip_member(archive_path, "f1.txt", 1024)
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("song.xml", "<a/>")
        self.assertEqual(_read_zip_member(archive_path, "song.xml", 1024), b"<a/>")
        for member, limit in (("missing.xml", 1024), ("song.xml", 2)):
            with self.assertRaises(ValueError):
                _read_zip_member(archive_path, member, limit)
        self.assertEqual([_media_basename(raw) for raw in (
            "/Users/me/Audio Files/kick.wav", "C:\\Samples\\snare 01.wav", "file:///Users/me/My%20Pad.aif",
            "../../../etc/passwd", "Samples/loop.wav/", "")],
            ["kick.wav", "snare 01.wav", "My Pad.aif", "passwd", "loop.wav", ""])

    def test_project_vocabulary_is_uniform_and_detail_aware(self):
        def record(detail):
            evidence = Evidence()
            facts = ProjectFacts(evidence, "s1", "test-xml", detail)
            facts.application("Example DAW 1.0", "app")
            facts.tempo(128, "tempo")
            facts.meter(6, 8, "meter")
            facts.tempo_map([{"beat": 0, "bpm": 128}, {"beat": 64, "bpm": 140}], "tempo map")
            facts.sample_rate(48000, "rate")
            facts.key("F# minor", "key")
            facts.track_count(2, "tracks")
            facts.track(0, "Lead", "instrument", "track 0")
            facts.track(1, "Vox", "audio", "track 1")
            facts.marker("Chorus", 32, "beats", "marker")
            facts.plugin("Reverb", 1, "plugin")
            facts.media_references(["/a/b/vox.wav", "D:\\other\\vox.wav", "/x/kick.wav"], "media")
            facts.clip_count(3, "clips")
            facts.track_notes(0, [Note(62, 90, 0, 1, None), Note(66, 70, 1, 1.5, None)], "notes")
            facts.track_notes(1, [], "notes")
            payload = evidence.as_json()
            jsonschema.validate(payload, SCHEMA)
            return payload["observations"]
        summary = record("summary")
        by_property = {}
        for item in summary:
            by_property.setdefault(item["property"], []).append(item)
        self.assertEqual(sorted(by_property), sorted([
            "project.application", "tempo", "meter", "project.tempo_map", "project.sample_rate", "project.key",
            "project.track_count", "project.track", "project.marker", "project.plugin",
            "project.media_reference", "project.clip_count", "project.track_notes"]))
        self.assertEqual(by_property["meter"][0]["value"], "6/8")
        self.assertEqual(by_property["project.plugin"][0]["scope"], "project_track:1")
        self.assertEqual([item["value"] for item in by_property["project.media_reference"]],
                         [{"basename": "kick.wav", "count": 1}, {"basename": "vox.wav", "count": 2}])
        notes = by_property["project.track_notes"]
        self.assertEqual([(item["scope"], item["value"]["noteCount"], item["value"]["pitchMax"]["name"]) for item in notes],
                         [("project_track:0", 2, "F#4")])
        self.assertTrue(all(item["provenance"] == "extracted" and item["method"] == "test-xml" for item in summary))
        full = record("full")
        self.assertEqual(len([item for item in full if item["property"] == "project.note"]), 2)
        with self.assertRaises(ValueError):
            ProjectFacts(Evidence(), "s1", "test", "summary").track(0, "x", "bus", "here")

    def test_project_flag_detects_formats_and_gives_specific_guidance(self):
        zip_bytes = self.dir / "empty.zip"
        with zipfile.ZipFile(zip_bytes, "w") as archive:
            archive.writestr("content.xml", "<a/>")
        zipped = zip_bytes.read_bytes()
        inventory = [
            ("song.ptx", b"\x03\x00anything", "pro-tools-session", "AAF"),
            ("song.PTF", b"x", "pro-tools-session", "AAF"),
            ("song.bwproject", b"BtWg", "bitwig-project", "File > Export > DAWproject"),
            ("song.reason", b"x", "reason-song", "MIDI"),
            ("song.cpr", b"RIFF", "cubase-project", "Cubase 14"),
            ("song.npr", b"RIFF", "cubase-project", "DAWproject"),
            ("song.als", gzip.compress(b"<Ableton/>"), "ableton-als", "Export Audio/Video"),
            ("song.rpp", b"\xef\xbb\xbf<REAPER_PROJECT 0.1", "reaper-rpp", "MIDI"),
            ("song.flp", b"FLhd\x06\x00", "fl-studio-flp", "MIDI file"),
            ("song.song", zipped, "studio-one-song", "DAWproject"),
            ("song.aaf", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "aaf", "mixdown"),
            ("song.omf", b"x", "omf", "mixdown"),
            ("score.musicxml", b"<score-partwise/>", "musicxml", "MIDI"),
            ("score.mxl", zipped, "musicxml", "MIDI"),
            ("score.mscz", zipped, "musescore-mscz", "MIDI"),
        ]
        for name, content, adapter, hint in inventory:
            with self.subTest(name=name):
                path = self.dir / name
                path.write_bytes(content)
                data = self.evidence("--project", path, mixdown=False)
                source = data["sources"][0]
                self.assertEqual((source["role"], source["adapter"], source["status"]), ("project", adapter, "unsupported"))
                self.assertRegex(source["sha256"], "^[0-9a-f]{64}$")
                warning = next(item for item in data["diagnostics"] if item["code"] == "unsupported_project")
                self.assertEqual(warning["severity"], "warning")
                self.assertIn(hint, warning["message"])
        unknown = self.dir / "song.xyz"
        unknown.write_bytes(b"x")
        data = self.evidence("--project", unknown, mixdown=False)
        self.assertEqual((data["sources"][0]["adapter"], data["sources"][0]["status"]), ("unsupported", "unsupported"))
        self.assertIn("DAWproject", next(item["message"] for item in data["diagnostics"] if item["code"] == "unsupported_project"))

    def test_project_flag_parses_supported_formats_and_rejects_mismatches(self):
        project = self.dir / "song.dawproject"
        with zipfile.ZipFile(project, "w") as archive:
            archive.writestr("project.xml", '<Project version="1"><Transport><Tempo value="97" unit="bpm"/></Transport></Project>')
        midi_path = self.write_midi("part.MID", [mido.Message("note_on", note=60, velocity=80),
                                                 mido.Message("note_off", note=60, time=480)])
        data = self.evidence("--project", project, "--project", midi_path, mixdown=False)
        self.assertEqual([(item["role"], item["adapter"], item["status"]) for item in data["sources"]],
                         [("project", "dawproject", "ok"), ("project", "midi", "ok")])
        self.assertEqual(next(item["value"] for item in data["observations"] if item["property"] == "tempo"), 97)
        self.assertTrue(any(item["property"] == "midi.track_summary" for item in data["observations"]))
        alias = self.evidence("--dawproject", project, mixdown=False)["sources"][0]
        self.assertEqual((alias["role"], alias["adapter"], alias["status"]), ("dawproject", "dawproject", "ok"))
        fake = self.dir / "fake.dawproject"
        fake.write_text("not a zip")
        packaged_file = self.dir / "flat.logicx"
        packaged_file.write_bytes(b"x")
        fake_als = self.dir / "plain.als"
        fake_als.write_text("<Ableton/>")
        for path in (fake, packaged_file, fake_als, self.dir / "missing.ptx"):
            with self.subTest(path=path.name):
                data = self.evidence("--project", path, mixdown=False)
                self.assertEqual(data["sources"][0]["status"], "error")
                self.assertEqual(data["observations"], [])
                self.assertTrue(any(item["code"] == "optional_source_error" for item in data["diagnostics"]))

    def test_missing_dependencies_print_one_actionable_line(self):
        bare = Path("/usr/bin/python3")
        if not bare.exists():
            self.skipTest("no system interpreter")
        if subprocess.run([str(bare), "-I", "-c", "import numpy, soundfile, mido, defusedxml"],
                          capture_output=True).returncode == 0:
            self.skipTest("system interpreter already has the dependencies")
        result = subprocess.run([str(bare), "-I", str(SKILL / "scripts" / "analyze.py"), "--mixdown", str(self.mix)],
                                cwd=self.dir, text=True, capture_output=True)
        self.assertEqual(result.returncode, 3)
        self.assertEqual(result.stdout, "")
        self.assertNotIn("Traceback", result.stderr)
        lines = result.stderr.strip().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertRegex(lines[0], r"missing Python module '(numpy|soundfile|mido|defusedxml)'")
        self.assertIn(str((SKILL / "requirements.txt").resolve()), lines[0])
        self.assertIn("venv", lines[0])


if __name__ == "__main__":
    unittest.main()
