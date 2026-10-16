from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import signal
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import aaf2
import jsonschema
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import SKILL, assert_matches_subprocess, run_cli, run_subprocess  # noqa: E402

from music_guide.adapters import aaf as aaf_adapter  # noqa: E402
from music_guide.model import Evidence  # noqa: E402
SCHEMA = json.loads((SKILL / "schemas" / "evidence.schema.json").read_text())

DOCTYPE = ('<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
           '"http://www.musicxml.org/dtds/partwise.dtd">')


def note(step, octave, duration, extra="", alter=None):
    shift = f"<alter>{alter}</alter>" if alter is not None else ""
    return (f"<note>{'<chord/>' if 'chord' in extra else ''}<pitch><step>{step}</step>{shift}"
            f"<octave>{octave}</octave></pitch><duration>{duration}</duration>"
            f"{extra.replace('chord', '')}</note>")


PIANO_MEASURES = (
    '<measure number="1"><attributes><divisions>2</divisions><key><fifths>2</fifths><mode>major</mode></key>'
    '<time><beats>3</beats><beat-type>4</beat-type></time></attributes>'
    '<direction><direction-type><rehearsal>A</rehearsal></direction-type><sound tempo="96"/></direction>'
    + note("D", 4, 2, '<tie type="start"/>') + note("F", 4, 2, "chord", alter=1) + note("E", 4, 2)
    + '<backup><duration>4</duration></backup>' + note("D", 2, 4) + '<note><rest/><duration>2</duration></note></measure>'
    '<measure number="2">' + note("D", 4, 6, '<tie type="stop"/>') + '</measure>')
CLARINET_MEASURES = ('<measure number="1"><attributes><divisions>1</divisions><key><fifths>4</fifths>'
                     '<mode>major</mode></key><transpose><chromatic>-2</chromatic></transpose></attributes>'
                     + note("C", 5, 3) + '</measure>')
PART_LIST = ('<part-list><score-part id="P1"><part-name>Piano</part-name><score-instrument id="I1">'
             '<instrument-name>Acoustic Piano</instrument-name></score-instrument></score-part>'
             '<score-part id="P2"><part-name>Clarinet in Bb</part-name></score-part></part-list>')
HEAD = ('<work><work-title>Etude</work-title></work><identification><creator type="composer">A. Writer</creator>'
        '<encoding><software>Example Notation 3.1</software></encoding></identification>')


def partwise(piano=PIANO_MEASURES, clarinet=CLARINET_MEASURES, doctype=DOCTYPE):
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n{doctype}\n<score-partwise version="4.0">{HEAD}{PART_LIST}'
            f'<part id="P1">{piano}</part><part id="P2">{clarinet}</part></score-partwise>')


def timewise():
    first = PIANO_MEASURES.split('<measure number="2">')
    second = CLARINET_MEASURES
    return ('<?xml version="1.0"?><score-timewise version="4.0">' + HEAD + PART_LIST
            + '<measure number="1"><part id="P1">' + first[0].split(">", 1)[1].rsplit("</measure>", 1)[0]
            + '</part><part id="P2">' + second.split(">", 1)[1].rsplit("</measure>", 1)[0] + '</part></measure>'
            + '<measure number="2"><part id="P1">' + first[1].rsplit("</measure>", 1)[0]
            + '</part></measure></score-timewise>')


class ScoreCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.mix = self.dir / "mix.wav"
        sf.write(self.mix, np.array([0.5, -0.5, 0.25, -0.25], dtype="float32"), 48000, subtype="FLOAT")

    def run_cli(self, *args, mixdown=True):
        prefix = ["--mixdown", str(self.mix)] if mixdown else []
        return run_cli(*prefix, *args, cwd=self.dir)

    def evidence(self, *args, mixdown=False):
        result = self.run_cli(*args, mixdown=mixdown)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        data = json.loads(result.stdout)
        jsonschema.validate(data, SCHEMA)
        return data

    @staticmethod
    def values(data, prop, source="s1"):
        return [item["value"] for item in data["observations"] if item["property"] == prop
                and item["sourceIds"] == [source]]

    @staticmethod
    def by_scope(data, prop):
        return {item["scope"]: item["value"] for item in data["observations"] if item["property"] == prop}

    def assert_rejected(self, data, fragment, source="s1"):
        entry = next(item for item in data["sources"] if item["id"] == source)
        self.assertEqual(entry["status"], "error")
        self.assertEqual([item for item in data["observations"] if item["sourceIds"] == [source]], [])
        warning = next(item for item in data["diagnostics"] if item["code"] == "optional_source_error")
        self.assertIn(fragment, warning["message"])

    def write(self, name, content):
        path = self.dir / name
        path.write_text(content) if isinstance(content, str) else path.write_bytes(content)
        return path

    def zip(self, name, members):
        path = self.dir / name
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for member, content in members.items():
                archive.writestr(member, content)
        return path


class MusicXmlTest(ScoreCase):
    def assert_etude(self, data, detail_full=False):
        self.assertEqual(data["sources"][0]["adapter"], "musicxml")
        self.assertEqual(data["sources"][0]["status"], "ok")
        self.assertEqual(self.values(data, "project.application"), ["Example Notation 3.1"])
        self.assertEqual(self.values(data, "musicxml.title"), ["Etude"])
        self.assertEqual(self.values(data, "musicxml.creator"), [{"type": "composer", "name": "A. Writer"}])
        self.assertEqual(self.values(data, "project.track_count"), [2])
        self.assertEqual([item["name"] for item in self.values(data, "project.track")], ["Piano", "Clarinet in Bb"])
        self.assertEqual(self.values(data, "tempo"), [96])
        self.assertEqual(self.values(data, "project.tempo_map"), [[{"beat": 0, "bpm": 96}]])
        self.assertEqual(self.values(data, "meter"), ["3/4"])
        self.assertEqual(self.values(data, "project.key"), ["D major"])
        self.assertEqual(self.values(data, "project.marker"), [{"name": "A", "position": 0, "unit": "beats"}])
        self.assertEqual(self.values(data, "musicxml.measure_count"), [2])
        self.assertEqual(self.values(data, "musicxml.length"), [6])
        notes = self.by_scope(data, "project.track_notes")
        piano = notes["project_track:0"]
        # D4 tied across the barline is one note, the F#4 chord member shares its start, backup places D2 at 0.
        self.assertEqual((piano["noteCount"], piano["pitchMin"]["midi"], piano["pitchMax"]["midi"]), (4, 38, 66))
        self.assertEqual((piano["startBeat"], piano["endBeat"], piano["maxPolyphony"]), (0, 6, 3))
        self.assertEqual(notes["project_track:1"]["pitchMin"]["midi"], 70)
        full = [item for item in data["observations"] if item["property"] == "project.note"]
        self.assertEqual(len(full), 5 if detail_full else 0)
        if detail_full:
            tied = next(item for item in full if item["scope"] == "project_track:0" and item["value"]["pitch"] == 62)
            self.assertEqual((tied["value"]["startBeat"], tied["value"]["endBeat"]), (0, 6))

    def test_partwise_values_summary_and_full(self):
        path = self.write("etude.musicxml", partwise())
        self.assert_etude(self.evidence("--project", path))
        self.assert_etude(self.evidence("--project", path, "--detail", "full"), detail_full=True)
        data = self.evidence("--project", path)
        self.assertTrue(any(item["code"] == "score_velocity_placeholder" for item in data["diagnostics"]))

    def test_timewise_matches_partwise(self):
        self.assert_etude(self.evidence("--project", self.write("etude.musicxml", timewise())))

    def test_compressed_mxl_uses_container_rootfile(self):
        container = ('<container><rootfiles><rootfile full-path="OEBPS/score.xml" '
                     'media-type="application/vnd.recordare.musicxml+xml"/></rootfiles></container>')
        path = self.zip("etude.mxl", {"mimetype": "application/vnd.recordare.musicxml",
                                      "META-INF/container.xml": container, "OEBPS/score.xml": partwise()})
        self.assert_etude(self.evidence("--project", path))
        broken = self.zip("no-container.mxl", {"score.xml": partwise()})
        self.assert_rejected(self.evidence("--project", broken, mixdown=True), "container.xml", "s2")
        empty = self.zip("no-root.mxl", {"META-INF/container.xml": "<container><rootfiles/></container>"})
        self.assert_rejected(self.evidence("--project", empty, mixdown=True), "rootfile", "s2")

    def test_metronome_tempo_and_changing_tempo_make_no_static_claim(self):
        metronome = ('<measure number="1"><attributes><divisions>1</divisions></attributes><direction>'
                     '<direction-type><metronome><beat-unit>quarter</beat-unit><beat-unit-dot/>'
                     '<per-minute>80</per-minute></metronome></direction-type></direction>'
                     + note("C", 4, 4) + '</measure>')
        data = self.evidence("--project", self.write("m.musicxml", partwise(metronome, "<measure number='1'/>")))
        self.assertEqual(self.values(data, "tempo"), [120])
        self.assertTrue(any(item["code"] == "musicxml_tempo_from_metronome" for item in data["diagnostics"]))
        changing = ('<measure number="1"><attributes><divisions>1</divisions></attributes><sound tempo="100"/>'
                    + note("C", 4, 4) + '<sound tempo="90"/>' + note("D", 4, 4) + '</measure>')
        data = self.evidence("--project", self.write("c.musicxml", partwise(changing, "<measure number='1'/>")))
        self.assertEqual(self.values(data, "tempo"), [])
        self.assertEqual(self.values(data, "project.tempo_map"), [[{"beat": 0, "bpm": 100}, {"beat": 4, "bpm": 90}]])

    def test_tempo_conflicts_with_declaration(self):
        path = self.write("etude.musicxml", partwise())
        declarations = self.write("declarations.json", json.dumps(
            [{"property": "tempo", "value": 120, "unit": "BPM", "scope": "project"}]))
        data = self.evidence("--project", path, "--declarations", declarations)
        self.assertEqual([(item["property"], item["scope"]) for item in data["conflicts"]], [("tempo", "project")])
        agreeing = self.write("agree.json", json.dumps(
            [{"property": "tempo", "value": 96, "unit": "BPM", "scope": "project"}]))
        self.assertEqual(self.evidence("--project", path, "--declarations", agreeing)["conflicts"], [])

    def test_unsafe_and_malformed_scores_are_rejected_atomically(self):
        cases = {
            "internal-subset.musicxml": (partwise(doctype='<!DOCTYPE score-partwise [<!ENTITY x "y">]>'), "XML"),
            "foreign-doctype.musicxml": (partwise(doctype='<!DOCTYPE score-partwise SYSTEM "http://evil.example/x.dtd">'), "XML"),
            "wrong-root.musicxml": ("<opus/>", "root"),
            "garbage.musicxml": ("<score-partwise><part", "XML"),
            "no-divisions.musicxml": (partwise('<measure number="1">' + note("C", 4, 4) + '</measure>'), "divisions"),
            "bad-duration.musicxml": (partwise('<measure number="1"><attributes><divisions>1</divisions></attributes>'
                                               + note("C", 4, "x") + '</measure>'), "duration"),
            "bad-pitch.musicxml": (partwise('<measure number="1"><attributes><divisions>1</divisions></attributes>'
                                            + note("C", 12, 1) + '</measure>'), "MIDI range"),
            "no-parts.musicxml": ("<score-partwise><part-list/></score-partwise>", "no parts"),
        }
        for name, (content, fragment) in cases.items():
            with self.subTest(name=name):
                self.assert_rejected(self.evidence("--project", self.write(name, content), mixdown=True), fragment, "s2")

    def test_wrong_extension_content_is_a_source_error(self):
        data = self.evidence("--project", self.write("binary.musicxml", b"\x00\x01binary"), mixdown=True)
        self.assertEqual(data["sources"][1]["status"], "error")

    def one_part(self, body, attributes="<divisions>1</divisions>", name="one.musicxml"):
        measure = f'<measure number="1"><attributes>{attributes}</attributes>{body}</measure>'
        return self.write(name, partwise(measure, "<measure number='1'/>"))

    def test_subprocess_smoke_matches_the_in_process_runner(self):
        result = assert_matches_subprocess(self, "--project", self.one_part(note("C", 4, 1)), cwd=self.dir)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["sources"][0]["status"], "ok")

    def test_overflowing_and_exponent_numbers_are_rejected_quickly(self):
        cases = {
            "duration-exp.musicxml": (note("C", 4, "1e400"), "duration"),
            "duration-huge-exp.musicxml": (note("C", 4, "1e10000000"), "duration"),
            "duration-digits.musicxml": (note("C", 4, "9" * 400), "duration"),
            "alter-exp.musicxml": (note("C", 4, 1, alter="1e400"), "alter"),
            "divisions-exp.musicxml": (None, "divisions"),
        }
        for name, (body, fragment) in cases.items():
            with self.subTest(name=name):
                if body is None:
                    path = self.one_part(note("C", 4, 1), "<divisions>1e400</divisions>", name)
                else:
                    path = self.one_part(body, name=name)
                self.assert_rejected(self.evidence("--project", path, mixdown=True), fragment, "s2")

    def test_unreadable_note_dynamics_fall_back_to_the_placeholder(self):
        for text in ("inf", "nan", "1e400", "-5", "loud"):
            with self.subTest(dynamics=text):
                data = self.evidence("--project", self.one_part(
                    note("C", 4, 1).replace("<note>", f'<note dynamics="{text}">')))
                summary = self.by_scope(data, "project.track_notes")["project_track:0"]
                self.assertEqual((summary["velocityMin"], summary["velocityMax"]), (90, 90))

    def test_sound_dynamics_set_the_default_velocity_until_changed_and_note_dynamics_win(self):
        body = ('<sound dynamics="50"/>' + note("C", 4, 1) + note("D", 4, 1).replace("<note>", '<note dynamics="100">')
                + '<direction><sound dynamics="80"/></direction>' + note("E", 4, 1))
        data = self.evidence("--project", self.one_part(body), "--detail", "full")
        notes = {item["value"]["pitch"]: item["value"]["velocity"] for item in data["observations"]
                 if item["property"] == "project.note"}
        self.assertEqual(notes, {60: 45, 62: 90, 64: 72})
        summary = next(item for item in data["observations"] if item["property"] == "project.track_notes")
        self.assertEqual(summary["confidence"], "high")
        self.assertFalse(any(item["code"] == "score_velocity_placeholder" for item in data["diagnostics"]))

    def test_placeholder_velocities_lower_confidence_and_are_counted(self):
        body = '<sound dynamics="50"/>' + note("C", 4, 1) + note("D", 4, 1)
        data = self.evidence("--project", self.one_part(note("A", 3, 1) + body))
        summary = next(item for item in data["observations"] if item["property"] == "project.track_notes")
        self.assertEqual(summary["confidence"], "medium")
        self.assertIn("1 of 3 velocities are placeholders", summary["confidenceReason"])
        counts = self.values(data, "musicxml.velocity_placeholders")
        self.assertEqual(counts, [{"placeholder": 1, "total": 3, "velocity": 90}])
        diagnostic = next(item for item in data["diagnostics"] if item["code"] == "score_velocity_placeholder")
        self.assertIn("1 of 3", diagnostic["message"])

    def test_composite_meter_microtones_octave_transposition_and_early_backup_are_flagged(self):
        signature = '<time><beats>2</beats><beat-type>4</beat-type><beats>3</beats><beat-type>8</beat-type></time>'
        data = self.evidence("--project", self.one_part(note("C", 4, 1), "<divisions>1</divisions>" + signature))
        self.assertEqual(self.values(data, "meter"), [])
        self.assertTrue(any(item["code"] == "musicxml_composite_meter" for item in data["diagnostics"]))
        data = self.evidence("--project", self.one_part(note("C", 4, 1, alter="0.5") + note("D", 4, 1, alter="-0.5")))
        flag = next(item for item in data["diagnostics"] if item["code"] == "musicxml_microtonal")
        self.assertIn("2 note", flag["message"])
        octave = ("<divisions>1</divisions><key><fifths>2</fifths><mode>major</mode></key>"
                  "<transpose><chromatic>0</chromatic><octave-change>-1</octave-change></transpose>")
        data = self.evidence("--project", self.one_part(note("C", 4, 1), octave))
        self.assertEqual(self.values(data, "project.key"), ["D major"])
        pitch = self.by_scope(data, "project.track_notes")["project_track:0"]["pitchMin"]["midi"]
        self.assertEqual(pitch, 48)
        early = note("C", 4, 1) + '<backup><duration>4</duration></backup>' + note("D", 4, 1)
        data = self.evidence("--project", self.one_part(early))
        self.assertEqual(data["sources"][0]["status"], "ok")
        self.assertEqual(self.by_scope(data, "project.track_notes")["project_track:0"]["noteCount"], 2)
        self.assertTrue(any(item["code"] == "musicxml_backup_clamped" for item in data["diagnostics"]))

    def test_many_repeated_key_and_meter_marks_stay_linear(self):
        measures = "".join(f'<measure number="{n}"><attributes><key><fifths>{n % 3}</fifths></key>'
                           f'<time><beats>{n % 5 + 1}</beats><beat-type>4</beat-type></time></attributes>'
                           + note("C", 4, 1) + '</measure>' for n in range(1, 55001))
        measures = measures.replace('<measure number="1">', '<measure number="1"><attributes><divisions>1</divisions></attributes>', 1)
        started = time.monotonic()
        data = self.evidence("--project", self.write("many.musicxml", partwise(measures, "<measure number='1'/>")))
        self.assertEqual(data["sources"][0]["status"], "ok")
        self.assertLess(time.monotonic() - started, 12)


class MuseScoreTest(ScoreCase):
    SCORE = (
        '<?xml version="1.0" encoding="UTF-8"?><museScore version="3.02"><programVersion>3.6.2</programVersion>'
        '<Score><Division>480</Division><metaTag name="workTitle">Etude</metaTag>'
        '<metaTag name="composer">A. Writer</metaTag>'
        '<Part><Staff id="1"/><trackName>Piano</trackName><Instrument id="piano"><trackName>Piano</trackName>'
        '</Instrument></Part>'
        '<Part><Staff id="2"/><Instrument id="clarinet"><trackName>Clarinet</trackName>'
        '<transposeChromatic>-2</transposeChromatic></Instrument></Part>'
        '<Staff id="1"><Measure><voice><KeySig><accidental>2</accidental><mode>major</mode></KeySig>'
        '<TimeSig><sigN>3</sigN><sigD>4</sigD></TimeSig>'
        '<Tempo><tempo>1.6</tempo><followText>1</followText><text>Allegro</text></Tempo>'
        '<RehearsalMark><text>A</text></RehearsalMark>'
        '<Chord><durationType>quarter</durationType><Note><Spanner type="Tie"><Tie/><next><location>'
        '<fractions>3/4</fractions></location></next></Spanner><pitch>62</pitch></Note>'
        '<Note><pitch>66</pitch></Note></Chord>'
        '<Chord><durationType>quarter</durationType><dots>1</dots><Note><pitch>64</pitch></Note></Chord></voice>'
        '<voice><Rest><durationType>quarter</durationType></Rest>'
        '<Chord><durationType>half</durationType><Note><pitch>38</pitch></Note></Chord></voice></Measure>'
        '<Measure><voice><Tuplet><normalNotes>2</normalNotes><actualNotes>3</actualNotes></Tuplet>'
        '<Chord><durationType>eighth</durationType><Note><Spanner type="Tie"><prev><location>'
        '<fractions>-3/4</fractions></location></prev></Spanner><pitch>62</pitch></Note></Chord>'
        '<Chord><durationType>eighth</durationType><Note><pitch>60</pitch></Note></Chord>'
        '<Chord><durationType>eighth</durationType><Note><pitch>61</pitch></Note></Chord><endTuplet/>'
        '<Rest><durationType>measure</durationType><duration>1/2</duration></Rest></voice></Measure></Staff>'
        '<Staff id="2"><Measure><voice><KeySig><accidental>4</accidental></KeySig>'
        '<TimeSig><sigN>3</sigN><sigD>4</sigD></TimeSig>'
        '<Chord><durationType>half</durationType><dots>1</dots><Note><pitch>70</pitch></Note></Chord></voice>'
        '</Measure></Staff></Score></museScore>')

    def package(self, name="etude.mscz", score=SCORE):
        container = ('<container><rootfiles><rootfile full-path="Etude.mscx"/></rootfiles></container>')
        return self.zip(name, {"META-INF/container.xml": container, "Etude.mscx": score, "Thumbnails/t.png": b"png"})

    def test_values_summary_and_full(self):
        for args, full in (([], False), (["--detail", "full"], True)):
            data = self.evidence("--project", self.package(), *args)
            self.assertEqual(data["sources"][0]["adapter"], "musescore-mscz")
            self.assertEqual(self.values(data, "project.application"), ["MuseScore 3.6.2"])
            self.assertEqual(self.values(data, "musescore.title"), ["Etude"])
            self.assertEqual(self.values(data, "tempo"), [96])
            self.assertEqual(self.values(data, "meter"), ["3/4"])
            self.assertEqual(self.values(data, "project.key"), ["D major"])
            self.assertEqual(self.values(data, "project.marker"), [{"name": "A", "position": 0, "unit": "beats"}])
            self.assertEqual([item["name"] for item in self.values(data, "project.track")], ["Piano", "Clarinet"])
            notes = self.by_scope(data, "project.track_notes")
            self.assertEqual(notes["project_track:0"]["noteCount"], 6)
            self.assertEqual(notes["project_track:0"]["endBeat"], 4)
            self.assertEqual(notes["project_track:1"]["pitchMax"]["midi"], 70)
            written = [item for item in data["observations"] if item["property"] == "project.note"]
            self.assertEqual(len(written), 7 if full else 0)
            if full:
                tuplet = [item["value"] for item in written if item["value"]["pitch"] == 60][0]
                self.assertAlmostEqual(tuplet["startBeat"], 3 + 1 / 3)
                self.assertAlmostEqual(tuplet["endBeat"], 3 + 2 / 3)

    def test_single_unnamed_mscx_is_found_without_container(self):
        path = self.zip("plain.mscz", {"score.mscx": self.SCORE})
        self.assertEqual(self.values(self.evidence("--project", path), "tempo"), [96])
        ambiguous = self.zip("two.mscz", {"a.mscx": self.SCORE, "b.mscx": self.SCORE})
        self.assert_rejected(self.evidence("--project", ambiguous, mixdown=True), "container.xml", "s2")

    def test_tempo_conflicts_with_declaration(self):
        declarations = self.write("declarations.json", json.dumps(
            [{"property": "tempo", "value": 120, "unit": "BPM", "scope": "project"}]))
        data = self.evidence("--project", self.package(), "--declarations", declarations)
        self.assertEqual([item["property"] for item in data["conflicts"]], ["tempo"])

    def test_malformed_scores_are_rejected_atomically(self):
        cases = {
            "wrong-root.mscz": self.zip("wrong-root.mscz", {"a.mscx": "<opus/>"}),
            "no-score.mscz": self.zip("no-score.mscz", {"a.mscx": '<museScore version="3.02"/>'}),
            "bad-duration.mscz": self.package("bad-duration.mscz", self.SCORE.replace("<durationType>half</durationType>",
                                                                                      "<durationType>nope</durationType>")),
            "bad-pitch.mscz": self.package("bad-pitch.mscz", self.SCORE.replace("<pitch>70</pitch>", "<pitch>300</pitch>")),
            "entity.mscz": self.zip("entity.mscz", {"a.mscx": '<!DOCTYPE x [<!ENTITY y "z">]><museScore/>'}),
            "not-zip.mscz": self.write("not-zip.mscz", b"PK\x03\x04broken"),
        }
        for name, path in cases.items():
            with self.subTest(name=name):
                data = self.evidence("--project", path, mixdown=True)
                self.assertEqual(data["sources"][1]["status"], "error")
                self.assertEqual([item for item in data["observations"] if item["sourceIds"] == ["s2"]], [])

    def located(self, fractions):
        return self.SCORE.replace("<voice><Rest>", f"<voice><location><fractions>{fractions}</fractions></location><Rest>", 1)

    def test_huge_fractions_and_dots_are_rejected_quickly(self):
        cases = {
            "fractions-exp.mscz": self.located("1e400"),
            "fractions-huge-exp.mscz": self.located("1e10000000"),
            "fractions-digits.mscz": self.located(f"{'9' * 400}/1"),
            "duration-exp.mscz": self.SCORE.replace("<duration>1/2</duration>", "<duration>1e100000000</duration>"),
            "dots.mscz": self.SCORE.replace("<dots>1</dots>", "<dots>5000000</dots>"),
            "tempo-inf.mscz": self.SCORE.replace("<tempo>1.6</tempo>", "<tempo>inf</tempo>"),
        }
        for name, score in cases.items():
            with self.subTest(name=name):
                data = self.evidence("--project", self.package(name, score), mixdown=True)
                self.assertEqual(data["sources"][1]["status"], "error")
                self.assertEqual([item for item in data["observations"] if item["sourceIds"] == ["s2"]], [])

    def test_all_velocities_are_placeholders_and_say_so(self):
        data = self.evidence("--project", self.package())
        summary = next(item for item in data["observations"] if item["property"] == "project.track_notes")
        self.assertEqual(summary["confidence"], "medium")
        self.assertIn("6 of 6 velocities are placeholders", summary["confidenceReason"])
        self.assertEqual(self.values(data, "musescore.velocity_placeholders")[0], {"placeholder": 6, "total": 6, "velocity": 80})


def build_aaf(path, *, top_level=True, track_rates=(48000, 48000), markers=True, unrated=False):
    """A composition with two sound slots, a picture slot, source clips and markers, written by pyaaf2."""
    with aaf2.open(str(path), "w") as f:
        ident = f.header["IdentificationList"].value[0]
        ident["ProductName"].value = "Example DAW"
        ident["CompanyName"].value = "Example Co"
        ident["ProductVersionString"].value = "9.1"
        source = f.create.SourceMob("Kick source")
        descriptor = f.create.WAVEDescriptor()
        descriptor["SampleRate"].value = 48000
        descriptor["Length"].value = 96000
        descriptor["ContainerFormat"].value = f.dictionary.lookup_containerdef("AAF")
        locator = f.create.NetworkLocator()
        locator["URLString"].value = "file:///Volumes/Secret%20Drive/Samples/Kick%20Drum.wav"
        descriptor["Locator"].append(locator)
        fmt = struct.pack("<HHIIHH", 1, 1, 48000, 96000, 2, 16)
        descriptor["Summary"].value = bytearray(b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt)) + b"WAVEfmt "
                                                + struct.pack("<I", len(fmt)) + fmt)
        source.descriptor = descriptor
        source.create_empty_slot(edit_rate=48000, media_kind="sound", slot_id=1)
        f.content.mobs.append(source)
        master = f.create.MasterMob("Kick")
        master_slot = master.create_timeline_slot(48000, slot_id=1)
        clip = f.create.SourceClip(media_kind="sound")
        clip.mob, clip.slot, clip.start, clip.length = source, source.slot_at(1), 0, 96000
        master_slot.segment = clip
        f.content.mobs.append(master)
        composition = f.create.CompositionMob("Song Mix")
        if top_level:
            composition.usage = "Usage_TopLevel"
        for number, (clips, rate) in enumerate(zip((2, 1), track_rates), 1):
            slot = composition.create_empty_sequence_slot(edit_rate=rate, media_kind="sound")
            slot.name = f"Track {number}"
            for _ in range(clips):
                slot.segment.components.append(master.create_source_clip(slot_id=1))
        composition.create_picture_slot(edit_rate=25).name = "Video"
        if unrated:
            slot = composition.create_empty_sequence_slot(edit_rate=25, media_kind="sound")
            slot.name = "No rate"
            slot["EditRate"].value = aaf2.rational.AAFRational(0, 1)
        if markers:
            event_slot = f.create.EventMobSlot()
            event_slot.slot_id, event_slot.name, event_slot.edit_rate = 9, "Markers", aaf2.rational.AAFRational(48000, 1)
            events = f.create.Sequence(media_kind="DescriptiveMetadata")
            events["Components"].value = []
            for position, text in ((48000, "Verse"), (288000, "Chorus")):
                marker = f.create.DescriptiveMarker()
                marker["Position"].value = position
                marker["Comment"].value = text
                events["Components"].append(marker)
            event_slot.segment = events
            composition.slots.append(event_slot)
        f.content.mobs.append(composition)


class AafTest(ScoreCase):
    # A drain thread that dies when the parent stops the child would only print a traceback, so collect them.
    thread_errors: list

    @classmethod
    def setUpClass(cls):
        cls.thread_errors = []
        cls.previous_hook = threading.excepthook
        threading.excepthook = lambda args: cls.thread_errors.append(args)

    @classmethod
    def tearDownClass(cls):
        threading.excepthook = cls.previous_hook
        errors = [args.exc_value for args in cls.thread_errors]
        assert not errors, f"exceptions in threads: {errors!r}"

    def make(self, name="mix.aaf", **options):
        path = self.dir / name
        build_aaf(path, **options)
        return path

    def test_composition_tracks_clips_markers_media_and_application(self):
        path = self.make()
        before = hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns
        data = self.evidence("--project", path)
        self.assertEqual(data["sources"][0]["adapter"], "aaf")
        self.assertEqual(data["sources"][0]["status"], "ok")
        self.assertEqual(self.values(data, "project.application"), ["Example DAW 9.1"])
        self.assertEqual(self.values(data, "aaf.company"), ["Example Co"])
        self.assertEqual(self.values(data, "aaf.mob_counts"), [{"composition": 1, "master": 1, "source": 1}])
        self.assertEqual(self.values(data, "aaf.composition"), [{"name": "Song Mix", "usage": "Usage_TopLevel", "slotCount": 4}])
        self.assertEqual(self.values(data, "project.track_count"), [3])
        tracks = self.by_scope(data, "project.track")
        self.assertEqual({scope: (item["name"], item["kind"]) for scope, item in tracks.items()}, {
            "project_track:0": ("Track 1", "audio"), "project_track:1": ("Track 2", "audio"),
            "project_track:2": ("Video", "unknown")})
        self.assertEqual(self.by_scope(data, "project.clip_count"),
                         {"project_track:0": 2, "project_track:1": 1, "project_track:2": 0})
        slots = self.by_scope(data, "aaf.slot")
        self.assertEqual((slots["project_track:0"]["mediaKind"], slots["project_track:0"]["editRate"],
                          slots["project_track:0"]["lengthSeconds"]), ("sound", 48000, 4))
        self.assertEqual((slots["project_track:2"]["mediaKind"], slots["project_track:2"]["editRate"]), ("picture", 25))
        self.assertEqual(self.values(data, "project.sample_rate"), [48000])
        self.assertEqual(self.values(data, "project.media_reference"), [{"basename": "Kick Drum.wav", "count": 1}])
        self.assertEqual(sorted((item["name"], item["position"], item["unit"]) for item in self.values(data, "project.marker")),
                         [("Chorus", 6, "seconds"), ("Verse", 1, "seconds")])
        self.assertNotIn("Secret", json.dumps(data))
        self.assertFalse(any(item["property"] in {"tempo", "meter"} for item in data["observations"]))
        full = self.evidence("--project", path, "--detail", "full")
        self.assertEqual([(item["property"], item["value"]) for item in full["observations"]],
                         [(item["property"], item["value"]) for item in data["observations"]])
        self.assertEqual((hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns), before)

    def test_disagreeing_sound_rates_make_no_single_sample_rate(self):
        data = self.evidence("--project", self.make(track_rates=(48000, 44100)))
        self.assertEqual(self.values(data, "project.sample_rate"), [])
        self.assertEqual(self.values(data, "aaf.sample_rates"), [[44100, 48000]])

    def test_file_without_top_level_composition_warns_and_reports_no_tracks(self):
        data = self.evidence("--project", self.make(top_level=False, markers=False))
        self.assertEqual(self.values(data, "project.track_count"), [])
        warning = next(item for item in data["diagnostics"] if item["code"] == "aaf_no_top_level_composition")
        self.assertEqual(warning["severity"], "warning")

    def test_container_warnings_become_diagnostics(self):
        raw = bytearray(self.make().read_bytes())
        struct.pack_into("<I", raw, 0x2C, 2)
        data = self.evidence("--project", self.write("odd-header.aaf", bytes(raw)))
        self.assertEqual(data["sources"][0]["status"], "ok")
        messages = [item["message"] for item in data["diagnostics"] if item["code"] == "aaf_container_warning"]
        self.assertEqual(len(messages), 1)
        self.assertIn("fat sector count", messages[0])

    def test_tempo_declaration_is_not_contradicted_by_a_file_without_tempo(self):
        declarations = self.write("declarations.json", json.dumps(
            [{"property": "tempo", "value": 120, "unit": "BPM", "scope": "project"}]))
        data = self.evidence("--project", self.make(), "--declarations", declarations)
        self.assertEqual(data["conflicts"], [])

    def test_installed_copy_runs_the_isolated_parser(self):
        installed = self.dir / "installed" / "music-guide"
        shutil.copytree(SKILL, installed, ignore=shutil.ignore_patterns("__pycache__"))
        result = run_subprocess("--project", self.make(), cwd=self.dir, script=installed / "scripts" / "analyze.py",
                                env={**os.environ, "PYTHONWARNINGS": "error"})
        self.assertEqual((result.returncode, result.stderr), (0, ""))
        self.assertEqual(json.loads(result.stdout)["sources"][0]["status"], "ok")

    def test_unsupported_and_corrupt_files_are_source_errors(self):
        good = self.make().read_bytes()
        cases = {
            "xml.aaf": (b'<?xml version="1.0"?><AAF/>', "XML"),
            "text.aaf": (b"\xd0\xcf\x11\xe0 not quite", "does not match"),
            "truncated.aaf": (good[:4096], "cannot read AAF"),
            "header.aaf": (good[:512], "cannot read AAF"),
            "zeroed.aaf": (good[:8] + bytes(len(good) - 8), "cannot read AAF"),
        }
        for name, (content, fragment) in cases.items():
            with self.subTest(name=name):
                data = self.evidence("--project", self.write(name, content), mixdown=True)
                self.assert_rejected(data, fragment, "s2")

    def make_bomb(self):
        """An index that claims 4 billion entries: pyaaf2 allocates memory in proportion to the claim."""
        path = self.dir / "bomb.aaf"
        with aaf2.open(str(path), "w"):
            pass
        with aaf2.open(str(path), "rw") as f:
            index = next(entry for entry in f.cfb.listdir_dict("/Header-2/Content-3b03").values()
                         if entry.name.endswith(" index"))
            index.open("rw").write(struct.pack("<I", 0xFFFFFFFF))
        return path

    def run_adapter(self, path, **limits):
        evidence = Evidence()
        aaf_adapter.aaf(path, "s1", evidence, "summary", **limits)
        return evidence

    def test_memory_limit_stops_a_parser_that_over_allocates(self):
        started = time.monotonic()
        with self.assertRaisesRegex(ValueError, r"AAF parser stopped: .*needed more than 256 MiB"):
            self.run_adapter(self.make_bomb(), memory_bytes=256 * 1024 * 1024)
        self.assertLess(time.monotonic() - started, 40)

    def test_time_limit_stops_the_parser(self):
        # The limit is checked once per poll, and a child that ends before the first poll is never checked. A 0.1 s
        # poll against a 0.05 s limit therefore failed on a quiet machine whose child finishes in under 0.1 s. A 10 ms
        # poll against a 5 ms limit cannot be beaten: the child cannot start, import pyaaf2 and parse in 10 ms.
        with mock.patch.object(aaf_adapter, "POLL_SECONDS", 0.01):
            with self.assertRaisesRegex(ValueError, r"AAF parser stopped: .*took longer than 0.005 seconds"):
                self.run_adapter(self.make(), timeout_seconds=0.005)
        self.assertEqual(self.run_adapter(self.make(), timeout_seconds=30).observations[0].property, "project.application")

    def test_child_ends_itself_at_its_own_deadline(self):
        config = {"paths": aaf_adapter.child_sys_path(), "path": str(self.make_bomb()), "sourceId": "s1",
                  "detail": "summary", "deadline": 1, "memory": 8 * 1024 ** 3}
        started = time.monotonic()
        child = subprocess.run([sys.executable, "-I", "-c", aaf_adapter.CHILD_CODE], input=json.dumps(config).encode(),
                               capture_output=True, timeout=60)
        self.assertLess(time.monotonic() - started, 30)
        deadline_status = (-signal.SIGALRM, aaf_adapter.DEADLINE_EXIT) if hasattr(signal, "SIGALRM") else (aaf_adapter.DEADLINE_EXIT,)
        self.assertIn(child.returncode, deadline_status)

    def test_child_sees_the_parents_import_path(self):
        shim = self.dir / "shim"
        (shim / "aaf2").mkdir(parents=True)
        (shim / "aaf2" / "__init__.py").write_text('raise ImportError("shim aaf2 was loaded")\n')
        result = run_subprocess("--mixdown", self.mix, "--project", self.make(), cwd=self.dir,
                                env={**os.environ, "PYTHONPATH": str(shim)})
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assert_rejected(data, "shim aaf2 was loaded", "s2")
        plain = self.evidence("--project", self.make("plain.aaf"))
        self.assertEqual(plain["sources"][0]["status"], "ok")

    def test_missing_pyaaf2_is_a_source_error_with_install_guidance(self):
        shim = self.dir / "shim"
        (shim / "aaf2").mkdir(parents=True)
        (shim / "aaf2" / "__init__.py").write_text('raise ImportError("No module named aaf2")\n')
        result = run_subprocess("--mixdown", self.mix, "--project", self.make(), cwd=self.dir,
                                env={**os.environ, "PYTHONPATH": str(shim)})
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assert_rejected(data, "pyaaf2 could not be imported", "s2")
        self.assertIn("requirements.txt", next(item["message"] for item in data["diagnostics"]
                                               if item["code"] == "optional_source_error"))
        self.assertTrue(any(item["property"] == "audio.sample_peak" for item in data["observations"]))

    def test_child_import_path_drops_the_working_directory_and_relative_entries(self):
        elsewhere = self.dir / "site"
        elsewhere.mkdir()
        original = sys.path[:]
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), original))
        sys.path[:] = ["", ".", "relative", str(self.dir), str(elsewhere), *original]
        previous = os.getcwd()
        os.chdir(self.dir)
        self.addCleanup(os.chdir, previous)
        chosen = aaf_adapter.child_sys_path()
        self.assertEqual(chosen[0], str(aaf_adapter.SCRIPTS_DIR))
        self.assertIn(str(elsewhere), chosen)
        for dropped in ("", ".", "relative", str(self.dir)):
            self.assertNotIn(dropped, chosen)

    def test_parent_does_not_import_pyaaf2(self):
        code = ("import sys; sys.path.insert(0, sys.argv[1]); import music_guide.cli, music_guide.adapters.aaf; "
                "print('aaf2' in sys.modules)")
        result = subprocess.run([sys.executable, "-c", code, str(SKILL / "scripts")], capture_output=True, text=True)
        self.assertEqual((result.stdout.strip(), result.returncode), ("False", 0), result.stderr)

    def test_child_output_is_read_with_a_bound(self):
        import io
        reader = aaf_adapter._Reader(io.BytesIO(b"x" * 100_000), 1000)
        reader.run()
        self.assertTrue(reader.overflow)
        self.assertEqual(len(reader.data), 1000)

    def test_stderr_is_drained_past_its_limit_so_a_chatty_child_finishes(self):
        chatty = ("import json, sys; config = json.loads(sys.stdin.buffer.read(1 << 20)); sys.path[:0] = config['paths']; "
                  "sys.stderr.write('x' * (1 << 20)); sys.stderr.flush(); "
                  "from music_guide.adapters.aaf_child import main; raise SystemExit(main(config))")
        started = time.monotonic()
        with mock.patch.object(aaf_adapter, "CHILD_CODE", chatty):
            evidence = self.run_adapter(self.make(), timeout_seconds=10)
        self.assertLess(time.monotonic() - started, 9)
        self.assertEqual(evidence.observations[0].property, "project.application")

    def test_a_missing_memory_monitor_is_reported(self):
        with mock.patch.object(aaf_adapter, "_memory_poller", return_value=None):
            evidence = self.run_adapter(self.make())
        notices = [item for item in evidence.diagnostics if item.code == "aaf_memory_cap_unavailable"]
        self.assertEqual(len(notices), 0 if sys.platform.startswith("linux") else 1)
        self.assertEqual(len([item for item in self.run_adapter(self.make()).diagnostics
                              if item.code == "aaf_memory_cap_unavailable"]), 0)

    def test_slots_without_an_edit_rate_are_counted(self):
        data = self.evidence("--project", self.make(unrated=True))
        warning = next(item for item in data["diagnostics"] if item["code"] == "aaf_slots_without_edit_rate")
        self.assertIn("1 slot", warning["message"])
        self.assertEqual(self.values(data, "project.track_count"), [3])


def daw_note(time, duration, key, vel="0.5", channel=0):
    return f'<Note time="{time}" duration="{duration}" channel="{channel}" key="{key}" vel="{vel}" rel="0.5"/>'


DAW_STRUCTURE = (
    '<Structure>'
    '<Track contentType="notes" id="t1" name="Bass"><Channel role="regular" id="c1"><Devices>'
    '<ClapPlugin deviceID="surge" deviceName="Surge XT" deviceRole="instrument"/>'
    '<Vst3Plugin deviceName="Hall Reverb" deviceRole="audioFX"/></Devices></Channel></Track>'
    '<Track contentType="audio" id="t2" name="Drums"><Channel role="regular" id="c2"/></Track>'
    '<Track contentType="audio notes" id="t3" name="Master"><Channel role="master" id="c3"><Devices>'
    '<BuiltinDevice deviceName="Limiter" deviceRole="audioFX"/></Devices></Channel></Track></Structure>')
DAW_ARRANGEMENT = (
    '<Arrangement><Lanes timeUnit="beats">'
    '<Lanes track="t1"><Clips>'
    '<Clip time="0" duration="8"><Notes>' + daw_note(0, 1, 60) + daw_note(0, 1, 64) + daw_note(1, 1, 67)
    + daw_note(4, 2, 72, "1.0") + '</Notes></Clip>'
    '<Clip time="16" duration="4" playStart="2"><Notes>' + daw_note(1, 1, 50) + daw_note(2, 1, 52)
    + daw_note(5, 4, 55) + daw_note(6, 1, 57) + '</Notes></Clip>'
    '<Clip time="32" duration="8" contentTimeUnit="seconds"><Notes>' + daw_note(0.5, 0.5, 40) + '</Notes></Clip>'
    '</Clips></Lanes>'
    '<Lanes track="t2"><Clips>'
    '<Clip time="0" duration="8" loopStart="0" loopEnd="2"><Clips><Clip time="0" duration="2" contentTimeUnit="beats">'
    '<Warps contentTimeUnit="seconds" timeUnit="beats"><Audio channels="2" sampleRate="48000" duration="1.0">'
    '<File path="C:\\Users\\secret\\audio\\Drum Loop.wav"/></Audio><Warp time="0" contentTime="0"/>'
    '<Warp time="2" contentTime="1"/></Warps></Clip></Clips></Clip>'
    '<Clip time="8" duration="4"/></Clips></Lanes>'
    '<Lanes track="t3"><Clips/></Lanes></Lanes>'
    '<Markers timeUnit="beats"><Marker time="0" name="Intro"/><Marker time="32" name="Drop"/></Markers>'
    '<TempoAutomation unit="bpm" timeUnit="beats"><Target/><RealPoint time="16" value="90" interpolation="hold"/>'
    '<RealPoint time="0" value="120" interpolation="hold"/></TempoAutomation>'
    '<TimeSignatureAutomation timeUnit="beats"><Target/><TimeSignaturePoint time="0" numerator="4" denominator="4"/>'
    '<TimeSignaturePoint time="32" numerator="3" denominator="4"/></TimeSignatureAutomation></Arrangement>')
DAW_HEAD = ('<Project version="1.0"><Application name="Bitwig Studio" version="5.2"/><Transport>'
            '<Tempo value="120" unit="bpm"/><TimeSignature numerator="4" denominator="4"/></Transport>')
DAW_METADATA = ('<MetaData><Title>Demo</Title><Artist>Someone</Artist><Genre>House</Genre><Comment> </Comment>'
                '<Bogus>ignored</Bogus></MetaData>')


class DawProjectTest(ScoreCase):
    def project(self, xml=None, metadata=DAW_METADATA, name="song.dawproject", extra=None):
        members = {"project.xml": xml or DAW_HEAD + DAW_STRUCTURE + DAW_ARRANGEMENT + "</Project>"}
        if metadata is not None:
            members["metadata.xml"] = metadata
        members.update(extra or {})
        return self.zip(name, members)

    def test_rich_project_values(self):
        data = self.evidence("--project", self.project())
        self.assertEqual(self.values(data, "dawproject.metadata"), [{"Title": "Demo", "Artist": "Someone", "Genre": "House"}])
        self.assertEqual(self.values(data, "tempo"), [120])
        self.assertEqual(self.values(data, "project.tempo_map"), [[{"beat": 0, "bpm": 120}, {"beat": 16, "bpm": 90}]])
        self.assertEqual(self.values(data, "meter"), ["4/4"])
        self.assertEqual(self.values(data, "dawproject.meter_map"),
                         [[{"beat": 0, "meter": "4/4"}, {"beat": 32, "meter": "3/4"}]])
        self.assertEqual([item["kind"] for item in self.values(data, "project.track")], ["instrument", "audio", "master"])
        plugins = sorted((item["scope"], item["value"]["name"]) for item in data["observations"]
                         if item["property"] == "project.plugin")
        self.assertEqual(plugins, [("project_track:0", "Hall Reverb"), ("project_track:0", "Surge XT"),
                                   ("project_track:2", "Limiter")])
        self.assertEqual(self.by_scope(data, "project.clip_count"),
                         {"project_track:0": 3, "project_track:1": 2, "project_track:2": 0})
        self.assertEqual(self.values(data, "project.media_reference"), [{"basename": "Drum Loop.wav", "count": 1}])
        self.assertNotIn("secret", json.dumps(data))
        self.assertEqual(sorted((item["name"], item["position"]) for item in self.values(data, "project.marker")),
                         [("Drop", 32), ("Intro", 0)])
        notes = self.by_scope(data, "project.track_notes")["project_track:0"]
        # First clip 4 notes; second clip keeps the notes at content 2 and 5 (the one at 1 is before playStart,
        # the one at 6 starts after the clip ends); the seconds clip adds one note.
        self.assertEqual(notes["noteCount"], 4 + 2 + 1)
        codes = {item["code"] for item in data["diagnostics"]}
        self.assertTrue({"dawproject_looped_clips", "dawproject_notes_outside_clip", "dawproject_note_timing"} <= codes)

    def test_clip_offsets_windows_and_seconds_conversion_with_tempo_automation(self):
        data = self.evidence("--project", self.project(), "--detail", "full")
        written = sorted((item["value"]["pitch"], item["value"]["startBeat"], item["value"]["endBeat"],
                          item["value"]["velocity"]) for item in data["observations"]
                         if item["property"] == "project.note")
        by_pitch = {pitch: (start, end, velocity) for pitch, start, end, velocity in written}
        self.assertEqual(by_pitch[52], (16, 17, 64))
        # The note at content 5 lasting 4 beats starts at 19 and is cut at the clip end (beat 20).
        self.assertEqual(by_pitch[55][:2], (19, 20))
        self.assertNotIn(50, by_pitch)
        self.assertNotIn(57, by_pitch)
        self.assertEqual(by_pitch[72], (4, 6, 127))
        # 32 beats is 18.667 s with 120 BPM for 16 beats and 90 BPM after; 0.5 s later is beat 32.75.
        start, end, _ = by_pitch[40]
        self.assertAlmostEqual(start, 32.75)
        self.assertAlmostEqual(end, 33.5)
        self.assertEqual(len(written), 7)

    def test_seconds_content_uses_constant_tempo_and_needs_one(self):
        arrangement = ('<Arrangement><Lanes timeUnit="beats"><Lanes track="t1"><Clips>'
                       '<Clip time="4" duration="8" contentTimeUnit="seconds"><Notes>' + daw_note(1, 0.5, 60)
                       + '</Notes></Clip></Clips></Lanes></Lanes></Arrangement>')
        constant = DAW_HEAD + DAW_STRUCTURE + arrangement + "</Project>"
        data = self.evidence("--project", self.project(constant), "--detail", "full")
        note = next(item["value"] for item in data["observations"] if item["property"] == "project.note")
        self.assertEqual((note["startBeat"], note["endBeat"]), (6, 7))
        untempoed = ('<Project version="1.0">' + DAW_STRUCTURE + arrangement + "</Project>")
        data = self.evidence("--project", self.project(untempoed))
        self.assertEqual(self.by_scope(data, "project.clip_count"), {"project_track:0": 1})
        self.assertEqual(self.by_scope(data, "project.track_notes"), {})
        warning = next(item for item in data["diagnostics"] if item["code"] == "dawproject_no_tempo_for_seconds")
        self.assertEqual(warning["severity"], "warning")

    def test_summary_and_full_differ_only_by_individual_notes(self):
        summary = self.evidence("--project", self.project())
        full = self.evidence("--project", self.project(), "--detail", "full")
        strip = lambda data: [(item["property"], item["value"]) for item in data["observations"]
                              if item["property"] != "project.note"]
        self.assertEqual(strip(summary), strip(full))
        self.assertEqual(len([item for item in full["observations"] if item["property"] == "project.note"]), 7)

    def test_tempo_conflicts_with_declaration(self):
        declarations = self.write("declarations.json", json.dumps(
            [{"property": "tempo", "value": 100, "unit": "BPM", "scope": "project"}]))
        data = self.evidence("--project", self.project(), "--declarations", declarations)
        self.assertEqual([item["property"] for item in data["conflicts"]], ["tempo"])

    def test_bad_optional_parts_keep_the_project_and_warn(self):
        bad_automation = DAW_ARRANGEMENT.replace('<TempoAutomation unit="bpm"', '<TempoAutomation unit="percent"')
        data = self.evidence("--project", self.project(DAW_HEAD + DAW_STRUCTURE + bad_automation + "</Project>"))
        self.assertEqual(self.values(data, "project.tempo_map"), [])
        self.assertEqual(self.values(data, "tempo"), [120])
        self.assertTrue(any(item["code"] == "dawproject_tempo_automation_unit" for item in data["diagnostics"]))
        for metadata in ("<MetaData><Title>unterminated", '<!DOCTYPE MetaData [<!ENTITY x "y">]><MetaData/>',
                         "<Other/>"):
            with self.subTest(metadata=metadata[:20]):
                data = self.evidence("--project", self.project(metadata=metadata))
                self.assertEqual(self.values(data, "dawproject.metadata"), [])
                self.assertEqual(self.values(data, "project.track_count"), [3])
                self.assertTrue(any(item["code"] == "dawproject_metadata_unreadable" for item in data["diagnostics"]))
        self.assertEqual(self.values(self.evidence("--project", self.project(metadata=None)), "project.track_count"), [3])

    def arrangement(self, body, tempo=""):
        return DAW_HEAD + DAW_STRUCTURE + f'<Arrangement>{tempo}{body}</Arrangement></Project>'

    def test_astronomical_numbers_are_errors_not_overflows(self):
        lane = '<Lanes track="t1"><Clips>{}</Clips></Lanes>'
        huge_note = lambda time, duration: (f'<Clip time="0" duration="8"><Notes><Note time="{time}" duration="{duration}" '
                                            'channel="0" key="60" vel="0.5"/></Notes></Clip>')
        ramp = ('<TempoAutomation unit="bpm" timeUnit="beats"><RealPoint time="0" value="{}" interpolation="hold"/>'
                '<RealPoint time="{}" value="60" interpolation="hold"/></TempoAutomation>')
        cases = {
            "note-time": (lane.format(huge_note("1e308", "1e308")), ""),
            "note-duration": (lane.format(huge_note("0", "1e308")), ""),
            "clip-time": (lane.format('<Clip time="1e308" duration="1e308"/>'), ""),
            "marker": ('<Markers timeUnit="beats"><Marker time="1e308" name="x"/></Markers>', ""),
            "tempo-time": (lane.format(huge_note("0", "1")), ramp.format("120", "1e308")),
            "tiny-tempo-seconds": (lane.format('<Clip time="1" duration="2" contentTimeUnit="seconds"><Notes>'
                                               + daw_note(1, 1, 60) + '</Notes></Clip>'), ramp.format("1e-320", "16")),
        }
        for name, (body, tempo) in cases.items():
            with self.subTest(name=name):
                data = self.evidence("--project", self.project(self.arrangement(body, tempo)), mixdown=True)
                self.assertEqual(data["sources"][1]["status"], "error")
                self.assertEqual([item for item in data["observations"] if item["sourceIds"] == ["s2"]], [])

    def test_placeholder_velocities_are_counted_and_lower_confidence(self):
        silent = '<Note time="2" duration="1" channel="0" key="62"/>'
        body = ('<Lanes track="t1"><Clips><Clip time="0" duration="8"><Notes>' + daw_note(0, 1, 60) + silent
                + '</Notes></Clip></Clips></Lanes>')
        data = self.evidence("--project", self.project(self.arrangement(body)))
        summary = next(item for item in data["observations"] if item["property"] == "project.track_notes")
        self.assertEqual(summary["confidence"], "medium")
        self.assertIn("1 of 2 velocities are placeholders", summary["confidenceReason"])
        self.assertEqual(self.values(data, "dawproject.velocity_placeholders"), [{"placeholder": 1, "total": 2, "velocity": 100}])
        self.assertIn("1 note", next(item for item in data["diagnostics"]
                                     if item["code"] == "dawproject_placeholder_velocity")["message"])
        stored = self.evidence("--project", self.project())
        self.assertEqual({item["confidence"] for item in stored["observations"] if item["property"] == "project.track_notes"}, {"high"})
        self.assertFalse(any(item["code"] == "dawproject_placeholder_velocity" for item in stored["diagnostics"]))

    def test_tempo_ramps_are_declared_even_without_notes_and_lower_note_confidence(self):
        ramp = ('<TempoAutomation unit="bpm" timeUnit="beats"><RealPoint time="0" value="120" interpolation="linear"/>'
                '<RealPoint time="16" value="60" interpolation="hold"/></TempoAutomation>')
        empty = self.evidence("--project", self.project(self.arrangement('<Lanes track="t1"><Clips/></Lanes>', ramp)))
        notice = next(item for item in empty["diagnostics"] if item["code"] == "dawproject_tempo_ramp_approximation")
        self.assertIn("mean tempo", notice["message"])
        body = ('<Lanes track="t1"><Clips><Clip time="0" duration="8"><Notes>' + daw_note(0, 1, 60) + '</Notes></Clip></Clips></Lanes>'
                '<Lanes track="t3"><Clips><Clip time="4" duration="8" contentTimeUnit="seconds"><Notes>' + daw_note(1, 1, 64)
                + '</Notes></Clip></Clips></Lanes>')
        data = self.evidence("--project", self.project(self.arrangement(body, ramp)))
        summaries = {item["scope"]: item for item in data["observations"] if item["property"] == "project.track_notes"}
        self.assertEqual(summaries["project_track:0"]["confidence"], "high")
        self.assertEqual(summaries["project_track:2"]["confidence"], "medium")
        self.assertIn("mean tempo", summaries["project_track:2"]["confidenceReason"])
        hold = ramp.replace('interpolation="linear"', 'interpolation="hold"')
        quiet = self.evidence("--project", self.project(self.arrangement(body, hold)))
        self.assertFalse(any(item["code"] == "dawproject_tempo_ramp_approximation" for item in quiet["diagnostics"]))

    def test_malformed_arrangements_are_rejected_atomically(self):
        def project(arrangement):
            return self.project(DAW_HEAD + DAW_STRUCTURE + arrangement + "</Project>")
        clip = '<Arrangement><Lanes track="t1"><Clips>{}</Clips></Lanes></Arrangement>'
        cases = {
            "key": clip.format('<Clip time="0" duration="1"><Notes>' + daw_note(0, 1, 200) + '</Notes></Clip>'),
            "time": clip.format('<Clip time="soon" duration="1"/>'),
            "unit": clip.format('<Clip time="0" contentTimeUnit="days"/>'),
            "missing": clip.format('<Clip duration="1"/>'),
            "negative": clip.format('<Clip time="0" duration="1"><Notes>' + daw_note(0, -1, 60) + '</Notes></Clip>'),
        }
        deep = "<Lanes>" * 40 + "</Lanes>" * 40
        cases["deep"] = f"<Arrangement>{deep}</Arrangement>"
        for name, arrangement in cases.items():
            with self.subTest(name=name):
                data = self.evidence("--project", project(arrangement), mixdown=True)
                self.assertEqual(data["sources"][1]["status"], "error")
                self.assertEqual([item for item in data["observations"] if item["sourceIds"] == ["s2"]], [])


class SignatureTest(ScoreCase):
    def test_pro_tools_signature_follows_ptformat(self):
        for name, content in (("a.ptx", b"\x03\x00rest"), ("b.ptf", b"\x00" + b"0010111100101011" + bytes(8)),
                              ("c.PTX", b"\x03")):
            with self.subTest(name=name):
                data = self.evidence("--project", self.write(name, content))
                self.assertEqual((data["sources"][0]["adapter"], data["sources"][0]["status"]),
                                 ("pro-tools-session", "unsupported"))
        for name, content in (("d.ptx", b"RIFF----"), ("e.ptf", b"0010111100101011"), ("f.ptx", b""),
                              ("g.ptf", bytes(1) + bytes(0x100) + b"0010111100101011")):
            with self.subTest(name=name):
                data = self.evidence("--project", self.write(name, content), mixdown=True)
                self.assert_rejected(data, "does not match the Pro Tools session", "s2")
        data = self.evidence("--project", self.write("template.pts", b"anything"))
        self.assertEqual(data["sources"][0]["status"], "unsupported")

    def test_bitwig_and_reason_have_no_confirmed_signature_and_stay_inventory_only(self):
        for name, adapter in (("a.bwproject", "bitwig-project"), ("a.reason", "reason-song")):
            with self.subTest(name=name):
                data = self.evidence("--project", self.write(name, b"whatever bytes"))
                self.assertEqual((data["sources"][0]["adapter"], data["sources"][0]["status"]), (adapter, "unsupported"))


if __name__ == "__main__":
    unittest.main()
