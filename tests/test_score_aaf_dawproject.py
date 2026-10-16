from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import jsonschema
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "music-guide"
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
        return subprocess.run([sys.executable, str(SKILL / "scripts" / "analyze.py"), *prefix, *map(str, args)],
                              cwd=self.dir, text=True, capture_output=True)

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


if __name__ == "__main__":
    unittest.main()
