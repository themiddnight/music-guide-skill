"""Behavior tests over synthetic, local source files and an installed skill copy."""

from __future__ import annotations

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
SCHEMA = json.loads((SKILL / "schemas" / "evidence.schema.json").read_text())
DECLARATIONS_SCHEMA = json.loads((SKILL / "schemas" / "declarations.schema.json").read_text())


class MusicGuideTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.mix = self.dir / "mix.wav"
        sf.write(self.mix, np.array([0.5, -0.5, 0.25, -0.25], dtype="float32"), 48000, subtype="FLOAT")

    def run_cli(self, *args, script=None):
        return subprocess.run([sys.executable, str(script or SKILL / "scripts" / "analyze.py"),
                               "--mixdown", str(self.mix), *map(str, args)], cwd=self.dir,
                              text=True, capture_output=True)

    def evidence(self, *args):
        result = self.run_cli(*args)
        self.assertEqual(result.returncode, 0, result.stderr)
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
        data = self.evidence("--midi", midi_path)
        notes = [item for item in data["observations"] if item["property"] == "midi.note"]
        self.assertEqual(notes[0]["value"]["endBeat"], 1)
        tempos = [item for item in data["observations"] if item["property"] == "tempo"]
        self.assertEqual(len(tempos), 2)
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
            archive.writestr("project.xml", '<Project version="1"><Transport><Tempo value="120"/><TimeSignature numerator="4" denominator="4"/></Transport><Structure><Track name="Drums"/><Track name="Bass"/></Structure><Arrangement><Tempo value="180"/></Arrangement></Project>')
        data = self.evidence("--dawproject", project)
        tempos = [item for item in data["observations"] if item["property"] == "tempo"]
        self.assertEqual([item["value"] for item in tempos], [120])
        self.assertEqual(next(item["value"] for item in data["observations"] if item["property"] == "dawproject.track_count"), 2)
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
        self.assertEqual(data["sources"][1]["status"], "unsupported")
        self.assertIsNone(data["sources"][1]["sha256"])
        target = self.dir / "outside" / "music-guide"
        install = subprocess.run([sys.executable, str(ROOT / "install.py"), "--dest", str(target)],
                                 text=True, capture_output=True)
        self.assertEqual(install.returncode, 0, install.stderr)
        self.assertEqual(subprocess.run([sys.executable, str(ROOT / "install.py"), "--dest", str(target)],
                                        capture_output=True).returncode, 0)
        data = self.run_cli(script=target / "scripts" / "analyze.py")
        self.assertEqual(data.returncode, 0, data.stderr)
        self.assertEqual(json.loads(data.stdout)["schemaVersion"], "1.0.0")
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


if __name__ == "__main__":
    unittest.main()
