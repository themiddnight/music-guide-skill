from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import mido

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import SKILL, run_cli  # noqa: E402
from test_fl_s1_cubase import fl_events, flp  # noqa: E402
from music_guide.model import Evidence  # noqa: E402
from music_guide.notes import Note  # noqa: E402
from music_guide.project import ProjectFacts, attach_track_rhythm  # noqa: E402


class StoredRhythmTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def midi(self, starts: list[int], meter: bool = True) -> Path:
        song = mido.MidiFile(type=1, ticks_per_beat=480)
        meta = mido.MidiTrack()
        if meter:
            meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
        song.tracks.append(meta)
        track = mido.MidiTrack()
        last = 0
        for tick in starts:
            track.append(mido.Message("note_on", note=60, velocity=90, time=tick - last))
            track.append(mido.Message("note_off", note=60, time=120))
            last = tick + 120
        song.tracks.append(track)
        path = self.root / "notes.mid"
        song.save(path)
        return path

    def report(self, path: Path) -> dict:
        result = run_cli("--midi", path, cwd=self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        return json.loads(result.stdout)

    @staticmethod
    def one(data: dict, property: str) -> dict:
        return next(item for item in data["observations"] if item["property"] == property)

    def test_midi_straight_and_swung_eighths(self):
        straight = self.report(self.midi([0, 240, 480, 720, 960, 1200]))
        swung = self.report(self.midi([0, 320, 480, 800, 960, 1280]))
        first = self.one(straight, "project.track_rhythm")
        second = self.one(swung, "project.track_rhythm")
        self.assertAlmostEqual(first["value"]["swingRatio"], 1.0)
        self.assertAlmostEqual(second["value"]["swingRatio"], 2.0)
        self.assertEqual(first["value"]["meterSource"], "midi.meter_map")
        self.assertEqual(first["provenance"], "extracted")
        self.assertEqual(first["confidence"], "high")

    def test_meter_after_notes_and_meter_map_changes_bar_grid(self):
        evidence = Evidence()
        facts = ProjectFacts(evidence, "s1", "musicxml", "summary")
        facts.track_notes(0, [Note(60, 90, 0, 1), Note(60, 90, 3, 4), Note(60, 90, 4, 5),
                              Note(60, 90, 7, 8)], "score notes")
        evidence.add("s1", "musicxml.meter_map", [{"beat": 0, "meter": "4/4"}, {"beat": 4, "meter": "3/4"}],
                     "signature", "project", "project_beats", "meter marks", "musicxml", "extracted", "high",
                     "Stored meter changes.")
        attach_track_rhythm(evidence, "s1")
        value = self.one(evidence.as_json(), "project.track_rhythm")["value"]
        self.assertEqual(value["meterSource"], "musicxml.meter_map")
        self.assertEqual(value["onsetsPerBar"], 1.333)
        self.assertIsNotNone(value["syncopation"])

    def test_musicxml_adapter_scores_notes_after_meter_emission(self):
        xml = ("<score-partwise version='4.0'><part-list><score-part id='P1'>"
               "<part-name>Piano</part-name></score-part></part-list><part id='P1'>"
               "<measure number='1'><attributes><divisions>1</divisions><time><beats>4</beats>"
               "<beat-type>4</beat-type></time></attributes><note><pitch><step>C</step><octave>4</octave>"
               "</pitch><duration>1</duration><type>quarter</type></note></measure>"
               "<measure number='2'><attributes><time><beats>3</beats><beat-type>4</beat-type>"
               "</time></attributes><note><pitch><step>D</step><octave>4</octave></pitch>"
               "<duration>1</duration><type>quarter</type></note></measure></part></score-partwise>")
        path = self.root / "score.musicxml"
        path.write_text(xml)
        result = run_cli("--project", path, cwd=self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(self.one(data, "project.track_rhythm")["value"]["meterSource"], "musicxml.meter_map")
        self.assertFalse(any(item["code"] == "rhythm_meter_unknown" for item in data["diagnostics"]))

    def test_no_meter_gives_null_and_diagnostic(self):
        data = self.report(self.midi([0, 240, 480], meter=False))
        value = self.one(data, "project.track_rhythm")["value"]
        self.assertIsNone(value["onsetsPerBar"])
        self.assertIsNone(value["syncopation"])
        self.assertIsNone(value["meterSource"])
        self.assertTrue(any(item["code"] == "rhythm_meter_unknown" for item in data["diagnostics"]))

    def test_partial_meter_map_does_not_invent_earlier_meter(self):
        evidence = Evidence()
        facts = ProjectFacts(evidence, "s1", "score", "summary")
        facts.track_notes(0, [Note(60, 80, 0, 1), Note(60, 80, 5, 6)], "notes")
        evidence.add("s1", "musicxml.meter_map", [{"beat": 4, "meter": "4/4"}], "signature",
                     "project", "project_beats", "meter", "score", "extracted", "high", "Stored meter.")
        attach_track_rhythm(evidence, "s1")
        self.assertIsNone(self.one(evidence.as_json(), "project.track_rhythm")["value"]["onsetsPerBar"])
        self.assertTrue(any(item.code == "rhythm_meter_unknown" for item in evidence.diagnostics))

    def test_repeated_equal_meter_marks_share_one_bar_origin(self):
        evidence = Evidence()
        facts = ProjectFacts(evidence, "s1", "mido", "summary")
        facts.track_notes(0, [Note(60, 90, 0, 1), Note(60, 90, 1, 2)], "notes")
        evidence.add("s1", "midi.meter_map", [{"beat": 0, "meter": "4/4"}, {"beat": 0, "meter": "4/4"}],
                     "signature", "project", "midi_ticks", "meters", "mido", "extracted", "high", "Stored marks")
        attach_track_rhythm(evidence, "s1")
        self.assertEqual(self.one(evidence.as_json(), "project.track_rhythm")["value"]["onsetsPerBar"], 2)

    def test_existing_note_summary_unchanged(self):
        data = self.report(self.midi([0, 480]))
        self.assertEqual(self.one(data, "midi.track_summary")["value"]["noteCount"], 2)
        self.assertEqual(self.one(data, "midi.track_summary")["value"]["startBeat"], 0)
        self.assertEqual(self.one(data, "midi.track_summary")["value"]["endBeat"], 1.25)

    def test_graded_fl_pattern_keeps_note_grade(self):
        path = self.root / "pattern.flp"
        path.write_bytes(flp(fl_events()))
        result = run_cli("--project", path, cwd=self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        original = self.one(data, "flstudio.pattern_notes")
        derived = self.one(data, "project.track_rhythm")
        self.assertEqual((derived["scope"], derived["confidence"], derived["provenance"]),
                         (original["scope"], "low", "extracted"))
        self.assertEqual(original["confidence"], "medium")
        self.assertIn("inside the pattern", derived["confidenceReason"])
        self.assertIn("meter is low confidence", derived["confidenceReason"])

    def test_unsupported_meter_keeps_density_and_nulls_syncopation(self):
        evidence = Evidence()
        facts = ProjectFacts(evidence, "s1", "score", "summary")
        facts.track_notes(0, [Note(60, 90, beat, beat + 0.5) for beat in range(6)], "notes")
        facts.meter(5, 4, "meter")
        attach_track_rhythm(evidence, "s1")
        value = self.one(evidence.as_json(), "project.track_rhythm")["value"]
        self.assertEqual(value["onsetsPerBar"], 3)
        self.assertIsNone(value["syncopation"])
        self.assertTrue(any(item.code == "rhythm_meter_unknown" for item in evidence.diagnostics))

    def test_straight_sixteenths_have_one_to_one_swing(self):
        evidence = Evidence()
        facts = ProjectFacts(evidence, "s1", "score", "summary")
        facts.track_notes(0, [Note(60, 90, beat / 4, beat / 4 + 0.1) for beat in range(16)], "notes")
        facts.meter(4, 4, "meter")
        attach_track_rhythm(evidence, "s1")
        self.assertEqual(self.one(evidence.as_json(), "project.track_rhythm")["value"]["swingRatio"], 1.0)

    def test_midi_only_run_does_not_import_numpy(self):
        code = ("import sys; from music_guide.cli import main; main(['--midi',sys.argv[1]]); "
                "assert 'numpy' not in sys.modules")
        result = subprocess.run([sys.executable, "-c", code, str(self.midi([0]))],
                                env={"PYTHONPATH": str(SKILL / "scripts")}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
