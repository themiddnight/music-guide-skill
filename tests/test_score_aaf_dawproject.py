from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import aaf2
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


def build_aaf(path, *, top_level=True, track_rates=(48000, 48000), markers=True):
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
    def run_cli(self, *args, mixdown=True):
        # Zero tolerance for stray warnings from the AAF parser or its file handles.
        prefix = ["--mixdown", str(self.mix)] if mixdown else []
        return subprocess.run([sys.executable, str(SKILL / "scripts" / "analyze.py"), *prefix, *map(str, args)],
                              cwd=self.dir, text=True, capture_output=True,
                              env={**os.environ, "PYTHONWARNINGS": "error"})

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

    def test_unsupported_and_corrupt_files_are_source_errors(self):
        good = self.make().read_bytes()
        bomb = bytearray(good)
        with aaf2.open(str(self.dir / "bomb.aaf"), "w"):
            pass
        (self.dir / "bomb.aaf").write_bytes(bytes(bomb))
        with aaf2.open(str(self.dir / "bomb.aaf"), "rw") as f:
            index = next(entry for entry in f.cfb.listdir_dict("/Header-2/Content-3b03").values()
                         if entry.name.endswith(" index"))
            stream = index.open("rw")
            stream.write(struct.pack("<I", 0xFFFFFFFF))
        cases = {
            "xml.aaf": (b'<?xml version="1.0"?><AAF/>', "XML"),
            "text.aaf": (b"\xd0\xcf\x11\xe0 not quite", "does not match"),
            "truncated.aaf": (good[:4096], "cannot read AAF"),
            "header.aaf": (good[:512], "cannot read AAF"),
            "zeroed.aaf": (good[:8] + bytes(len(good) - 8), "cannot read AAF"),
            "bomb.aaf": ((self.dir / "bomb.aaf").read_bytes(), "AAF parser stopped"),
        }
        for name, (content, fragment) in cases.items():
            with self.subTest(name=name):
                data = self.evidence("--project", self.write(name, content), mixdown=True)
                self.assert_rejected(data, fragment, "s2")


if __name__ == "__main__":
    unittest.main()
