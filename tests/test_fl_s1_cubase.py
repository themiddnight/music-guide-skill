"""FL Studio, Studio One and Cubase adapters, exercised through the CLI with small synthetic projects."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import jsonschema

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import SKILL, assert_matches_subprocess, run_cli  # noqa: E402

SCHEMA = json.loads((SKILL / "schemas" / "evidence.schema.json").read_text())

from music_guide.adapters import cubase, flstudio, studioone  # noqa: E402


class ProjectCliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def run_cli(self, *args):
        return run_cli(*args, cwd=self.dir)

    def evidence(self, *args):
        result = self.run_cli(*args)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        data = json.loads(result.stdout)
        jsonschema.validate(data, SCHEMA)
        return data

    def project(self, path, *extra):
        return self.evidence("--project", path, *extra)

    @staticmethod
    def values(data, property):
        return [item["value"] for item in data["observations"] if item["property"] == property]

    @staticmethod
    def items(data, property):
        return [item for item in data["observations"] if item["property"] == property]

    def assert_rejected(self, path, *fragments):
        data = self.project(path)
        self.assertEqual(data["sources"][0]["status"], "error")
        self.assertEqual(data["observations"], [])
        for fragment in fragments:
            self.assertIn(fragment, self.error_message(data))

    @staticmethod
    def error_message(data):
        return next(item["message"] for item in data["diagnostics"] if item["code"] == "optional_source_error")

    def declare_tempo(self, bpm):
        path = self.dir / "declarations.json"
        path.write_text(json.dumps([{"property": "tempo", "value": bpm, "unit": "BPM", "scope": "project"}]))
        return path


def varint(value):
    out = bytearray()
    while True:
        byte, value = value & 0x7F, value >> 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def event(event_id, payload):
    if event_id < 64:
        assert len(payload) == 1
    elif event_id < 128:
        assert len(payload) == 2
    elif event_id < 192:
        assert len(payload) == 4
    else:
        return bytes([event_id]) + varint(len(payload)) + payload
    return bytes([event_id]) + payload


def byte(event_id, value):
    return event(event_id, bytes([value]))


def word(event_id, value):
    return event(event_id, struct.pack("<H", value))


def dword(event_id, value):
    return event(event_id, struct.pack("<I", value))


def text(event_id, value, utf16=True):
    return event(event_id, (value + "\0").encode("utf-16-le" if utf16 else "ascii"))


def note(position, length, key, velocity=100, rack=0):
    return struct.pack("<IHHIHHBBBBBBBB", position, 0, rack, length, key, 0, 120, 0, 64, 0, 64, velocity, 128, 128)


def playlist_item(position, length, item_index, track=499, size=32):
    item = struct.pack("<IHHIHH2sH4sff", position, 20480, item_index, length, track, 0, b"x\0", 64, b"@d\x80\x80", 0.0, -1.0)
    return item + bytes(size - len(item))


def flp(events, version="20.8.3.2304", ppq=96, channels=2, size_delta=0, header_size=6):
    body = b"".join(events)
    return (b"FLhd" + struct.pack("<IhHH", header_size, 0, channels, ppq) + b"FLdt"
            + struct.pack("<I", len(body) + size_delta) + body)


def fl_events(version="20.8.3.2304", playlist_size=32):
    utf16 = tuple(int(part) for part in version.split(".")[:2]) >= (11, 5)

    def label(event_id, value):
        return text(event_id, value, utf16)
    return [
        event(199, (version + "\0").encode("ascii")), dword(156, 128_000), byte(17, 4), byte(18, 4),
        label(194, "Test Song"), label(206, "Ambient"), label(207, "Ada"),
        word(64, 0), byte(21, 4), label(201, "3x Osc"), label(203, "Lead"),
        word(64, 1), byte(21, 0), label(192, "Kick"), label(196, "C:\\Samples\\Drums\\Kick One.wav"),
        word(65, 1), label(193, "Verse"),
        event(224, note(0, 96, 60) + note(96, 96, 64, 90, 1) + note(192, 0, 67)),
        word(99, 1), label(241, "Main"),
        event(233, playlist_item(0, 384, 20481, size=playlist_size) + playlist_item(384, 96, 1, 498, playlist_size)),
        word(98, 0), label(201, "Fruity Reeverb 2"), label(204, "Drums"),
    ]


class FlStudioTest(ProjectCliTest):
    def write(self, name="song.flp", **kwargs):
        path = self.dir / name
        path.write_bytes(flp(kwargs.pop("events", None) or fl_events(), **kwargs))
        return path

    def test_subprocess_smoke_matches_the_in_process_runner(self):
        result = assert_matches_subprocess(self, "--project", self.write(), cwd=self.dir)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["sources"][0]["adapter"], "fl-studio-flp")

    def test_project_facts_and_pattern_summary(self):
        data = self.project(self.write())
        source = data["sources"][0]
        self.assertEqual((source["adapter"], source["status"]), ("fl-studio-flp", "ok"))
        self.assertEqual(self.values(data, "project.application"), ["FL Studio 20.8.3.2304"])
        tempo = self.items(data, "tempo")[0]
        self.assertEqual((tempo["value"], tempo["unit"], tempo["confidence"]), (128.0, "BPM", "medium"))
        meter = self.items(data, "meter")[0]
        self.assertEqual((meter["value"], meter["confidence"]), ("4/4", "low"))
        self.assertEqual(self.values(data, "flstudio.header"), [{"format": 0, "channelCount": 2, "ppq": 96}])
        self.assertEqual(self.values(data, "flstudio.title"), ["Test Song"])
        self.assertEqual(self.values(data, "flstudio.genre"), ["Ambient"])
        self.assertEqual(self.values(data, "flstudio.artists"), ["Ada"])
        self.assertEqual(self.values(data, "project.track_count"), [2])
        self.assertEqual({item["scope"]: item["value"] for item in self.items(data, "project.track")}, {
            "project_track:0": {"name": "Lead", "kind": "instrument", "index": 0},
            "project_track:1": {"name": "Kick", "kind": "unknown", "index": 1}})
        self.assertEqual([(item["value"], item["scope"]) for item in self.items(data, "project.plugin")],
                         [({"name": "3x Osc", "track": 0}, "project_track:0"),
                          ({"name": "Fruity Reeverb 2", "track": None}, "project")])
        self.assertEqual(self.values(data, "project.media_reference"), [{"basename": "Kick One.wav", "count": 1}])
        self.assertEqual(self.values(data, "flstudio.mixer_insert_name"), ["Drums"])
        self.assertEqual(self.values(data, "flstudio.pattern"), [{"id": 1, "name": "Verse", "noteCount": 3}])
        summary = self.items(data, "flstudio.pattern_notes")[0]
        self.assertEqual((summary["scope"], summary["confidence"], summary["timeDomain"]),
                         ("flstudio_pattern:1", "medium", "static"))
        self.assertEqual((summary["value"]["noteCount"], summary["value"]["endBeat"], summary["value"]["rackChannelIds"]),
                         (3, 2, [0, 1]))
        self.assertEqual(self.values(data, "flstudio.arrangement"), [
            {"index": 0, "name": "Main", "itemCount": 2, "patternItemCount": 1, "channelItemCount": 1, "spanBeats": 5.0}])
        self.assertEqual(self.values(data, "project.clip_count"), [2])
        self.assertEqual(self.items(data, "flstudio.note"), [])
        self.assertTrue(any(item["code"] == "flp_zero_length_notes" for item in data["diagnostics"]))

    def test_full_detail_lists_every_note_in_pattern_time(self):
        notes = self.items(self.project(self.write(), "--detail", "full"), "flstudio.note")
        self.assertEqual([(item["value"]["pitch"], item["value"]["startBeat"], item["value"]["endBeat"]) for item in notes],
                         [(60, 0.0, 1.0), (64, 1.0, 2.0), (67, 2.0, 2.0)])
        self.assertEqual({(item["timeDomain"], item["scope"]) for item in notes}, {("pattern_beats", "flstudio_pattern:1")})

    def test_tempo_conflicts_with_declaration(self):
        data = self.project(self.write(), "--declarations", self.declare_tempo(140))
        self.assertEqual([item["property"] for item in data["conflicts"]], ["tempo"])
        self.assertEqual(self.project(self.write(), "--declarations", self.declare_tempo(128))["conflicts"], [])

    def test_new_playlist_item_size_and_legacy_tempo_and_ascii_strings(self):
        data = self.project(self.write(events=fl_events("21.0.3.3517", playlist_size=60), name="new.flp"))
        self.assertEqual(self.values(data, "flstudio.arrangement")[0]["itemCount"], 2)
        old = [event(199, b"9.1.0\0"), word(66, 120), text(194, "Old Song", utf16=False), word(64, 0), byte(21, 2),
               text(196, "kick.wav", utf16=False)]
        data = self.project(self.write(events=old, name="old.flp", channels=1))
        self.assertEqual(self.values(data, "tempo"), [120.0])
        self.assertIn("event 66", self.items(data, "tempo")[0]["locator"])
        self.assertEqual(self.values(data, "flstudio.title"), ["Old Song"])
        self.assertEqual(self.values(data, "project.track"), [{"name": None, "kind": "audio", "index": 0}])
        self.assertEqual(self.values(data, "meter"), [])

    def test_rejects_malformed_and_oversized_input(self):
        complete = flp(fl_events())
        cases = {
            "cut.flp": (complete[:-3], "data chunk declares"),
            "cut-event.flp": (flp(fl_events()[:-1] + [fl_events()[-1][:-2]]), "truncated or corrupt"),
            "oversized-chunk.flp": (flp(fl_events(), size_delta=10_000), "data chunk declares"),
            "oversized-text.flp": (flp([event(199, b"20.8.3.2304\0"), bytes([194]) + varint(2**31) + b"x"]), "declares"),
            "endless-varint.flp": (flp([event(199, b"20.8.3.2304\0"), bytes([194]) + b"\xff" * 8]), "invalid length"),
            "mid-varint.flp": (flp([event(199, b"20.8.3.2304\0"), bytes([194, 0x80])]), "ends inside its length"),
            "no-version.flp": (flp([dword(156, 128_000)]), "no FL Studio version"),
            "bad-version.flp": (flp([event(199, b"beta\0")]), "not in major.minor.patch"),
            "bad-header.flp": (flp(fl_events(), header_size=7), "header is malformed"),
            "zero-ppq.flp": (flp(fl_events(), ppq=0), "invalid PPQ"),
        }
        for name, (content, fragment) in cases.items():
            with self.subTest(name=name):
                (self.dir / name).write_bytes(content)
                self.assert_rejected(self.dir / name, fragment)
        (self.dir / "fake.flp").write_bytes(b"FLhd\x06\x00")
        data = self.project(self.dir / "fake.flp")
        self.assertEqual(data["sources"][0]["status"], "error")
        self.assertIn("does not match", self.error_message(data))

    def test_skips_bad_notes_without_losing_the_project(self):
        events = fl_events()
        events[events.index(event(224, note(0, 96, 60) + note(96, 96, 64, 90, 1) + note(192, 0, 67)))] = event(
            224, note(0, 96, 60) + note(96, 96, 200) + b"\0")
        data = self.project(self.write(events=events))
        self.assertEqual(data["sources"][0]["status"], "ok")
        self.assertEqual(self.items(data, "flstudio.pattern_notes"), [])
        self.assertTrue(any(item["code"] == "flp_event_skipped" for item in data["diagnostics"]))
        with_key = fl_events()
        with_key[with_key.index(event(224, note(0, 96, 60) + note(96, 96, 64, 90, 1) + note(192, 0, 67)))] = event(
            224, note(0, 96, 60) + note(96, 96, 200))
        data = self.project(self.write(events=with_key, name="key.flp"))
        self.assertEqual(self.values(data, "flstudio.pattern"), [{"id": 1, "name": "Verse", "noteCount": 1}])
        self.assertTrue(any("key 200" in item["message"] for item in data["diagnostics"]))

    def test_detect_requires_both_chunk_tags(self):
        path = self.write()
        self.assertTrue(flstudio.detect(path))
        path.write_bytes(path.read_bytes()[:14] + b"XXXX" + path.read_bytes()[18:])
        self.assertFalse(flstudio.detect(path))


    def test_wrapped_vst_is_named_by_its_display_name_at_low_confidence(self):
        events = [event(199, b"20.8.3.2304\0"), word(64, 0), byte(21, 4), text(201, "Fruity Wrapper"),
                  text(203, "Serum"), word(64, 1), byte(21, 4), text(201, "Fruity Wrapper"),
                  word(98, 0), text(201, "Fruity Wrapper"), text(203, "Valhalla Room"),
                  word(98, 1), text(201, "Fruity Wrapper"), word(98, 2), text(201, "Fruity Limiter"), text(203, "Master Limiter")]
        data = self.project(self.write(events=events, channels=2))
        plugins = {(item["value"]["name"], item["scope"]): item for item in self.items(data, "project.plugin")}
        self.assertEqual(sorted(plugins), [("Fruity Limiter", "project"), ("Fruity Wrapper", "project"),
                                           ("Fruity Wrapper", "project_track:1"), ("Serum", "project_track:0"),
                                           ("Valhalla Room", "project")])
        for key in (("Serum", "project_track:0"), ("Valhalla Room", "project"), ("Fruity Wrapper", "project_track:1"),
                    ("Fruity Wrapper", "project")):
            self.assertEqual(plugins[key]["confidence"], "low", key)
            self.assertIn("Fruity Wrapper", plugins[key]["confidenceReason"], key)
        self.assertEqual(plugins[("Fruity Limiter", "project")]["confidence"], "medium")
        self.assertEqual(self.values(data, "flstudio.channel")[0]["plugin"], "Fruity Wrapper")

    def test_channel_kinds_follow_the_stored_type_and_sample(self):
        events = [event(199, b"20.8.3.2304\0"), word(64, 0), byte(21, 0), word(64, 1), byte(21, 4), text(196, "C:\\a\\hit.wav"),
                  word(64, 2), byte(21, 2), text(196, "C:\\a\\clip.wav"), word(64, 3), byte(21, 4), text(201, "Fruity Wrapper"),
                  text(196, "C:\\a\\x.wav"), word(64, 4), byte(21, 99)]
        data = self.project(self.write(events=events, channels=5))
        kinds = [item["value"]["kind"] for item in sorted(self.items(data, "project.track"), key=lambda item: item["value"]["index"])]
        self.assertEqual(kinds, ["unknown", "audio", "audio", "instrument", "unknown"])

    def test_layout_dependent_facts_are_not_high_confidence(self):
        data = self.project(self.write())
        for property in ("flstudio.time_signature", "flstudio.arrangement", "flstudio.channel", "project.track",
                         "project.plugin", "project.track_count", "project.clip_count", "project.media_reference"):
            items = self.items(data, property)
            self.assertTrue(items, property)
            for item in items:
                self.assertNotEqual(item["confidence"], "high", property)
        self.assertEqual(self.items(data, "project.application")[0]["confidence"], "high")

    def test_playlist_item_size_follows_the_version_and_never_guesses(self):
        def arrangement(version, count, size):
            items = b"".join(playlist_item(index * 96, 96, 20481, size=size) for index in range(count))
            return [event(199, version.encode() + b"\0"), word(99, 1), text(241, "Main"), event(233, items)]
        old = self.project(self.write("old15.flp", events=arrangement("20.8.3.2304", 15, 32)))
        self.assertEqual(self.values(old, "flstudio.arrangement")[0]["itemCount"], 15)
        self.assertEqual(self.values(old, "project.clip_count"), [15])
        new = self.project(self.write("new8.flp", events=arrangement("21.0.3.3517", 8, 60)))
        self.assertEqual(self.values(new, "flstudio.arrangement")[0]["itemCount"], 8)
        wrong = self.project(self.write("wrong.flp", events=arrangement("20.8.3.2304", 3, 60)))
        self.assertEqual(wrong["sources"][0]["status"], "ok")
        self.assertEqual(self.values(wrong, "flstudio.arrangement"), [])
        self.assertTrue(any(item["code"] == "flp_event_skipped" and "32 byte" in item["message"] for item in wrong["diagnostics"]))

    def test_rejects_a_project_with_too_many_notes(self):
        count = flstudio.MAX_NOTES + 1
        events = [event(199, b"20.8.3.2304\0"), word(65, 1), event(224, note(0, 96, 60) * count)]
        self.assert_rejected(self.write("many.flp", events=events), "notes")

    def test_empty_plugin_name_event_keeps_the_earlier_channel_name(self):
        events = [event(199, b"20.8.3.2304\0"), word(64, 0), byte(21, 2), text(203, "Lead"), text(203, ""), text(201, "3x Osc")]
        data = self.project(self.write(events=events, channels=1))
        self.assertEqual(self.values(data, "project.track")[0]["name"], "Lead")


SONG_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Song{ns}>
  <Attributes x:id="Root" defaultTimeFormat="2" length="316">
    <Attributes x:id="timeContext" sampleRate="44100">
      <TempoMap x:id="tempoMap">{tempos}</TempoMap>
      <TimeSignatureMap x:id="timeSignatureMap">{meters}</TimeSignatureMap>
      <List x:id="keySignatureMap"><KeySignatureMapSegment start="0" root="9" scale="minor"/></List>
    </Attributes>
    <List x:id="Tracks">
      <MarkerTrack name="Marker">
        <MarkerEvent markerType="2" timeFormat="2" name="Start"/>
        <MarkerEvent start="16" timeFormat="2" name="Chorus"/>
        <MarkerEvent start="8" timeFormat="1" name="Seconds"/>
      </MarkerTrack>
      <MediaTrack mediaType="Audio" name="Vox" trackID="{{TR-VOX}}" activeLayer="1" timeFormat="2">
        <UID x:id="channelID" uid="{{CH-VOX}}"/>
        <List x:id="Layers">
          <Attributes id="0" layerName="Take 1"><List x:id="Events">
            <AudioEvent clipID="{{CLIP-1}}" start="4" length="8"/><AudioEvent clipID="{{CLIP-1}}" start="20" length="2"/>
          </List></Attributes>
          <Attributes id="1" layerName="Take 2"><List x:id="Events"><AudioEvent clipID="{{CLIP-2}}" start="34" length="3"/></List></Attributes>
        </List>
      </MediaTrack>
      <MediaTrack mediaType="Music" name="Keys"><List x:id="Events"><MusicPart clipID="{{CLIP-M}}" start="4" length="8"/></List></MediaTrack>
      <FolderTrack name="Group"/>
      <ArrangerTrack timeFormat="2"><SectionEvent start="0" length="16" name="Verse"/></ArrangerTrack>
    </List>
  </Attributes>
</Song>"""
ONE_TEMPO = '<TempoMapSegment curveType="0" start="0" end="1e200" tempo="0.5"/>'
TWO_TEMPOS = ('<TempoMapSegment curveType="0" start="0" end="32" tempo="0.5"/>'
              '<TempoMapSegment curveType="1" start="32" end="1e200" tempo="1"/>')
ONE_METER = '<TimeSignatureMapSegment start="0" numerator="3" denominator="4"/>'
TWO_METERS = ONE_METER + '<TimeSignatureMapSegment start="32" numerator="7" denominator="8"/>'
META_XML = """<MetaInformation>
  <Attribute id="Document:Title" value="Fixture Song"/>
  <Attribute id="Document:Generator" value="Studio One/5.5.2.86528"/>
  <Attribute id="Media:SampleRate" value="48000"/>
  <Attribute id="Media:KeySignature" value="A minor"/>
  <Attribute id="Media:Mystery" value="42"/>
</MetaInformation>"""
MEDIA_XML = """<MediaPool><Attributes x:id="rootFolder"><MediaFolder name="Audio">
  <AudioClip mediaID="{CLIP-1}"><Url x:id="path" type="1" url="file:///tmp/secret/Media/Vox%201.wav"/></AudioClip>
  <AudioClip mediaID="{CLIP-2}"><Url x:id="path" type="1" url="file:///C:/Users/me/Media/Vox 2.wav"/></AudioClip>
  <MusicClip mediaID="{CLIP-M}"><Url x:id="dataPath" type="1" url="media:///Performances/Keys/Keys(0).musicx"/></MusicClip>
</MediaFolder></Attributes></MediaPool>"""
MIXER_XML = """<AudioMixer><Attributes x:id="channels">
  <ChannelGroup name="AudioTrack">
    <AudioTrackChannel label="Vox"><UID x:id="uniqueID" uid="{CH-VOX}"/>
      <Attributes x:id="Inserts">
        <Attributes name="FX01"><Attributes x:id="deviceData" name="Pro EQ"/></Attributes>
        <Attributes x:id="Presets" pname="default"/>
      </Attributes>
      <Attributes x:id="PostFaderInserts"><Attributes name="FX01"><Attributes x:id="ghostData"><Attributes x:id="classInfo" name="Limiter"/></Attributes></Attributes></Attributes>
    </AudioTrackChannel>
  </ChannelGroup>
  <ChannelGroup name="FxChannel"><FxChannel label="Verb"><Attributes x:id="Inserts"><Attributes name="FX01"><Attributes x:id="deviceData" name="Room Reverb"/></Attributes></Attributes></FxChannel></ChannelGroup>
</Attributes></AudioMixer>"""


def song_xml(declare=True, tempos=ONE_TEMPO, meters=ONE_METER):
    return SONG_XML.format(ns=' xmlns:x="urn:presonus"' if declare else "", tempos=tempos, meters=meters)


class StudioOneTest(ProjectCliTest):
    def write(self, name="song.song", song=None, **parts):
        files = {"Song/song.xml": song or song_xml(), "metainfo.xml": META_XML, "Song/mediapool.xml": MEDIA_XML,
                 "Devices/audiomixer.xml": MIXER_XML} | parts
        path = self.dir / name
        with zipfile.ZipFile(path, "w") as archive:
            for member, content in files.items():
                if content is not None:
                    archive.writestr(member, content)
        return path

    def test_project_facts_from_a_declared_namespace_song(self):
        data = self.project(self.write())
        self.assertEqual((data["sources"][0]["adapter"], data["sources"][0]["status"]), ("studio-one-song", "ok"))
        self.assertEqual(self.values(data, "project.application"), ["Studio One 5.5.2.86528"])
        tempo = self.items(data, "tempo")[0]
        self.assertEqual((tempo["value"], tempo["unit"], tempo["scope"]), (120.0, "BPM", "project"))
        self.assertEqual(self.values(data, "meter"), ["3/4"])
        self.assertEqual(sorted(item["locator"] for item in self.items(data, "project.sample_rate")),
                         ["Song/song.xml timeContext/@sampleRate", "metainfo.xml Media:SampleRate"])
        self.assertEqual(self.values(data, "project.key"), ["A minor"])
        self.assertEqual(self.values(data, "project.track_count"), [3])
        self.assertEqual({item["scope"]: item["value"] for item in self.items(data, "project.track")}, {
            "project_track:0": {"name": "Vox", "kind": "audio", "index": 0},
            "project_track:1": {"name": "Keys", "kind": "instrument", "index": 1},
            "project_track:2": {"name": "Group", "kind": "folder", "index": 2}})
        self.assertEqual({item["scope"]: item["value"] for item in self.items(data, "project.clip_count")},
                         {"project_track:0": 1, "project_track:1": 1, "project_track:2": 0})
        self.assertEqual([(item["value"]["name"], item["scope"]) for item in self.items(data, "project.plugin")],
                         [("Pro EQ", "project_track:0"), ("Limiter", "project_track:0"), ("Room Reverb", "project")])
        self.assertEqual(sorted(item["value"]["basename"] for item in self.items(data, "project.media_reference")),
                         ["Vox 1.wav", "Vox 2.wav"])
        self.assertEqual([item["value"] for item in self.items(data, "project.marker")],
                         [{"name": "Start", "position": 0.0, "unit": "beats"},
                          {"name": "Chorus", "position": 16.0, "unit": "beats"}])
        self.assertEqual(self.values(data, "studioone.arranger_section"),
                         [{"name": "Verse", "startBeat": 0.0, "lengthBeats": 16.0}])
        self.assertEqual(self.values(data, "studioone.key_signature_map"), [[{"beat": 0.0, "root": "9", "scale": "minor"}]])
        self.assertTrue(any(item["code"] == "studioone_time_format" for item in data["diagnostics"]))
        serialized = json.dumps(data)
        self.assertNotIn("secret", serialized)
        self.assertNotIn("C:/Users", serialized)

    def test_unconfirmed_metadata_is_recorded_literally_never_mapped(self):
        data = self.project(self.write())
        metainfo = {item["value"]["id"]: item["value"]["value"] for item in self.items(data, "studioone.metainfo")}
        self.assertEqual(metainfo["Media:Mystery"], "42")
        self.assertEqual(metainfo["Document:Title"], "Fixture Song")
        self.assertEqual({item["property"] for item in data["observations"]} & {"42"}, set())
        data = self.project(self.write("none.song", **{"metainfo.xml": None}))
        self.assertTrue(any(item["code"] == "studioone_part_missing" and "metainfo.xml" in item["message"]
                            for item in data["diagnostics"]))
        self.assertEqual(self.values(data, "project.application"), [])
        self.assertEqual(len(self.values(data, "tempo")), 1)

    def test_tempo_and_meter_maps_make_no_static_claim(self):
        data = self.project(self.write(song=song_xml(tempos=TWO_TEMPOS, meters=TWO_METERS)))
        self.assertEqual(self.values(data, "tempo"), [])
        self.assertEqual(self.values(data, "meter"), [])
        self.assertEqual(self.values(data, "project.tempo_map"), [[{"beat": 0.0, "bpm": 120.0}, {"beat": 32.0, "bpm": 60.0}]])
        self.assertEqual(self.values(data, "studioone.meter_map"), [[{"beat": 0.0, "numerator": 3, "denominator": 4},
                                                                     {"beat": 32.0, "numerator": 7, "denominator": 8}]])
        self.assertTrue(any(item["code"] == "studioone_tempo_ramps" for item in data["diagnostics"]))

    def test_undeclared_prefix_song_parses_and_conflicts_with_declaration(self):
        data = self.project(self.write(song=song_xml(declare=False)), "--declarations", self.declare_tempo(100))
        self.assertEqual(data["sources"][0]["status"], "ok")
        self.assertEqual([item["property"] for item in data["conflicts"]], ["tempo"])
        self.assertEqual(self.values(data, "project.track_count"), [3])
        self.assertEqual(self.project(self.write(song=song_xml(declare=False)), "--declarations",
                                      self.declare_tempo(120))["conflicts"], [])
        cpp_names = song_xml(declare=False).replace(
            '<FolderTrack name="Group"/>', '<FolderTrack name="Group"><ProcessGraph::SlotNode x:id="n" ns:flag="a b=c"/></FolderTrack>')
        data = self.project(self.write("cpp.song", song=cpp_names))
        self.assertEqual((data["sources"][0]["status"], self.values(data, "project.track_count")), ("ok", [3]))

    def test_rejects_malformed_and_unsafe_archives(self):
        self.assert_rejected(self.write("no-song.song", **{"Song/song.xml": None}), "does not match")
        self.assert_rejected(self.write("bad-root.song", song="<Other/>"), "root must be Song")
        self.assert_rejected(self.write("dtd.song", song='<!DOCTYPE Song [<!ENTITY x "bad">]><Song>&x;</Song>'),
                             "unsafe or malformed XML")
        self.assert_rejected(self.write("broken.song", song="<Song><Attributes></Song>"), "malformed XML")
        (self.dir / "text.song").write_text("not a zip")
        self.assert_rejected(self.dir / "text.song", "does not match")
        many = self.dir / "many.song"
        with zipfile.ZipFile(many, "w") as archive:
            archive.writestr("Song/song.xml", song_xml())
            for index in range(studioone.MAX_ENTRIES):
                archive.writestr(f"Envelopes/{index}.envelopex", b"")
        self.assert_rejected(many, "does not match")
        with zipfile.ZipFile(self.dir / "big.song", "w") as archive:
            archive.writestr("Song/song.xml", "<Song>" + " " * (8 * 1024 * 1024) + "</Song>")
        self.assert_rejected(self.dir / "big.song", "bounded Song/song.xml")

    def test_unreadable_optional_parts_keep_the_song(self):
        data = self.project(self.write(**{"Song/mediapool.xml": "<MediaPool", "Devices/audiomixer.xml": "<a><b></a>"}))
        self.assertEqual(data["sources"][0]["status"], "ok")
        self.assertEqual(self.values(data, "project.media_reference"), [])
        self.assertEqual(len([item for item in data["diagnostics"] if item["code"] == "studioone_part_unreadable"]), 2)
        self.assertEqual(len(self.values(data, "project.track")), 3)


    def invalid(self, data, fragment):
        return [item for item in data["diagnostics"] if item["code"] == "studioone_attribute_invalid"
                and fragment in item["message"]]

    def test_unparsable_attributes_skip_the_item_with_a_warning(self):
        song = song_xml().replace('<MarkerEvent start="16" timeFormat="2" name="Chorus"/>',
                                  '<MarkerEvent start="abc" timeFormat="2" name="Chorus"/>')
        data = self.project(self.write(song=song))
        self.assertEqual([item["value"]["name"] for item in self.items(data, "project.marker")], ["Start"])
        self.assertTrue(self.invalid(data, "start"), data["diagnostics"])
        only = song_xml(tempos='<TempoMapSegment curveType="0" start="junk" end="1e200" tempo="0.5"/>')
        data = self.project(self.write("tempo.song", song=only))
        self.assertEqual(self.values(data, "tempo") + self.values(data, "project.tempo_map"), [])
        self.assertTrue(any(item["code"] == "studioone_segment_dropped" and "start='junk'" in item["message"]
                            for item in data["diagnostics"]), data["diagnostics"])
        rate = song_xml().replace('sampleRate="44100"', 'sampleRate="fast"')
        data = self.project(self.write("rate.song", song=rate))
        self.assertTrue(self.invalid(data, "sampleRate"), data["diagnostics"])

    def test_dropped_tempo_and_meter_segments_are_reported_and_block_a_static_claim(self):
        for bad in ('tempo="0"', 'tempo="junk"'):
            tempos = ONE_TEMPO + f'<TempoMapSegment curveType="0" start="32" end="1e200" {bad}/>'
            data = self.project(self.write("t.song", song=song_xml(tempos=tempos)))
            self.assertEqual(self.values(data, "tempo"), [], bad)
            self.assertEqual(self.values(data, "project.tempo_map"), [[{"beat": 0.0, "bpm": 120.0}]], bad)
            self.assertTrue(any(item["code"] == "studioone_segment_dropped" and "tempo" in item["message"]
                                for item in data["diagnostics"]), bad)
        for numerator, denominator in (("1e300", "4"), ("3", "3"), ("0", "4"), ("3", "0"), ("2.5", "4"), ("3", "256")):
            meters = ONE_METER + (f'<TimeSignatureMapSegment start="32" numerator="{numerator}" denominator="{denominator}"/>')
            data = self.project(self.write("m.song", song=song_xml(meters=meters)))
            self.assertEqual(self.values(data, "meter"), [], (numerator, denominator))
            self.assertTrue(any(item["code"] == "studioone_segment_dropped" and "signature" in item["message"]
                                for item in data["diagnostics"]), (numerator, denominator))

    def test_empty_insert_slots_folders_and_empty_names(self):
        mixer = MIXER_XML.replace('<Attributes x:id="Presets" pname="default"/>',
                                  '<Attributes name="FX02"/><Attributes name="FX03"><Attributes x:id="deviceData"/></Attributes>')
        song = song_xml().replace('<MediaTrack mediaType="Music" name="Keys">', '<MediaTrack mediaType="Music" name="">')
        data = self.project(self.write(song=song, **{"Devices/audiomixer.xml": mixer}))
        self.assertEqual([item["value"]["name"] for item in self.items(data, "project.plugin")],
                         ["Pro EQ", "Limiter", "Room Reverb"])
        tracks = {item["value"]["index"]: item["value"] for item in self.items(data, "project.track")}
        self.assertEqual((tracks[1]["name"], tracks[2]["kind"]), (None, "folder"))


def cm(value, bom=False):
    body = value.encode() + b"\0" + (b"\xef\xbb\xbf" if bom else b"")
    return struct.pack(">I", len(body)) + body


def container(name, version=0):
    return b"\xff\xff\xff\xfe" + cm(name) + struct.pack(">H", version)


def obj(name, payload=b"", version=0):
    return b"\xff\xff\xff\xff" + cm(name) + struct.pack(">HI", version, len(payload)) + payload


def riff(chunks, tag=b"NUND"):
    body = b"".join(chunks)
    return b"RIFF" + struct.pack(">I", len(body) - 4) + tag + body


def pair(name, kind, arch):
    root = cm(name) + cm(kind)
    return [b"ROOT" + struct.pack(">I", len(root)) + root, b"ARCH" + struct.pack(">I", len(arch)) + arch]


def app_version(arch=None, name="Cubase", version="Version 12.0.50", extended=True):
    payload = cm(name) + cm(version) + cm("Mar 30 2022") + struct.pack(">I", 400)
    if extended:
        payload += cm("WIN64") + struct.pack(">I", 0) + cm("UTF-8") + cm("us")
    return container("CmObject") + obj("PAppVersion", payload, 512)


def attribute(attribute_id, kind, value):
    record = b"\x00\x14" + cm("StMedia::PStringIDMediaAttribute") + bytes(8)
    names = [("ID", b"\x00\x08" + cm(attribute_id)), ("Type", b"\x00\x01" + struct.pack(">q", 4)),
             ("Flags", b"\x00\x01" + struct.pack(">q", 2)), (kind, value)]
    return record + struct.pack(">I", len(names)) + b"".join(cm(name) + payload for name, payload in names)


def metadata(rate=44100.0, tempo=120.0, signature=(3, 4)):
    nested = (b"\x00\x14" + cm("CmRational") + bytes(8) + struct.pack(">I", 2) + cm("Numerator")
              + b"\x00\x01" + struct.pack(">q", signature[0]) + cm("Denominator") + b"\x00\x01"
              + struct.pack(">q", signature[1]))
    records = (attribute("AudioSampleRate", "Float", b"\x00\x04" + struct.pack(">d", rate))
               + attribute("MediaProjectName", "String", b"\x00\x08" + cm("Demo", bom=True))
               + attribute("MusicalSignature", "Object", b"\x00\x02" + nested)
               + attribute("MusicalTempo", "Float", b"\x00\x04" + struct.pack(">d", tempo)))
    return container("CmObject") + obj("StMedia::PAttributes", b"\x00\x00\x00\x01" + records)


def track_header(reference_to_base, class_reference, name):
    return class_reference + struct.pack(">I", 99) + bytes(10) + reference_to_base + struct.pack(">I", 77) + cm(name, bom=True)


def arrangement(names=(("Lead", "MAudioTrackEvent"), ("Second", "MAudioTrackEvent"), ("Keys", "MInstrumentTrackEvent"),
                       ("Chords", "MMidiTrackEvent"), ("Bus", "MDeviceTrackEvent"))):
    """Class definitions come first and later tracks refer back to them, as in real projects."""
    prefix = container("MGroupEvent") + container("MListNode") + container("MDataNode")
    base = len(prefix) - len(container("MDataNode")) - len(container("MListNode"))
    stream = bytearray(prefix)
    offsets = {}
    for _, class_name in names:
        if class_name not in offsets:
            offsets[class_name] = len(stream)
            stream += container(class_name)
    base_ref = struct.pack(">I", 0x80000000 | base)
    body = bytearray()
    for name, class_name in names:
        body += track_header(base_ref, struct.pack(">I", 0x80000000 | offsets[class_name]), name)
        body += struct.pack(">I", 12) + b"Plugin Name\0" + b"\x00\x08" + cm("Pro EQ", bom=True)
        body += cm(f"{name} 1.wav", bom=True) + cm("wav", bom=True)
        body += cm("D:\\Secret\\Audio\\" + f"{name} 2.WAV", bom=True)
    inline = obj("MAudioTrackEvent", b"\x00\x01" + track_header(base_ref, b"\x00\x00\x00\x00", "Inline") + bytes(8))
    stream += inline + bytes(body)
    return bytes(stream)


class CubaseTest(ProjectCliTest):
    def write(self, name="song.cpr", version=None, meta=None, arrange=None, extra=()):
        chunks = pair("Version", "PAppVersion", version if version is not None else app_version())
        chunks += pair("Arrangement1", "PArrangement", arrange if arrange is not None else arrangement())
        chunks += pair("Metadata", "StMedia::PAttributes", meta if meta is not None else metadata())
        for item in extra:
            chunks += item
        path = self.dir / name
        path.write_bytes(riff(chunks))
        return path

    def test_application_and_media_attributes_are_structural(self):
        data = self.project(self.write())
        self.assertEqual((data["sources"][0]["adapter"], data["sources"][0]["status"]), ("cubase-project", "ok"))
        application = self.items(data, "project.application")[0]
        self.assertEqual((application["value"], application["confidence"]), ("Cubase 12.0.50", "high"))
        self.assertEqual(self.values(data, "cubase.app_version"), [{
            "name": "Cubase", "version": "Version 12.0.50", "buildDate": "Mar 30 2022", "architecture": "WIN64",
            "encoding": "UTF-8", "locale": "us"}])
        self.assertEqual([item["name"] for item in self.values(data, "cubase.root_chunks")[0]], ["Version", "Arrangement1", "Metadata"])
        attributes = {item["id"]: item["value"] for item in self.values(data, "cubase.media_attribute")}
        self.assertEqual(attributes, {"AudioSampleRate": 44100.0, "MediaProjectName": "Demo",
                                      "MusicalSignature": {"Numerator": 3, "Denominator": 4}, "MusicalTempo": 120.0})
        self.assertEqual([(item["value"], item["confidence"]) for item in self.items(data, "project.sample_rate")],
                         [(44100, "medium")])
        self.assertEqual([(item["value"], item["confidence"]) for item in self.items(data, "tempo")], [(120.0, "low")])
        self.assertEqual([(item["value"], item["confidence"]) for item in self.items(data, "meter")], [("3/4", "low")])
        self.assertIn("tempo and signature tracks", self.items(data, "tempo")[0]["confidenceReason"])
        self.assertTrue(any(item["code"] == "cubase_partial_decoding" for item in data["diagnostics"]))

    def test_pattern_matched_facts_are_low_confidence_and_say_why(self):
        data = self.project(self.write())
        tracks = {item["value"]["index"]: item["value"] for item in self.items(data, "project.track")}
        self.assertEqual([(track["name"], track["kind"]) for track in tracks.values()], [
            ("Inline", "audio"), ("Lead", "audio"), ("Second", "audio"), ("Keys", "instrument"), ("Chords", "midi"),
            ("Bus", "unknown")])
        self.assertEqual(self.values(data, "project.track_count"), [6])
        self.assertEqual(self.values(data, "cubase.track_classes"), [["MAudioTrackEvent", "MDeviceTrackEvent",
                                                                       "MInstrumentTrackEvent", "MMidiTrackEvent"]])
        for property in ("project.track", "project.track_count", "project.plugin", "project.media_reference",
                         "cubase.track_class", "cubase.plugin_mentions"):
            for item in self.items(data, property):
                self.assertEqual(item["confidence"], "low", property)
        self.assertIn("byte pattern", self.items(data, "project.track")[0]["confidenceReason"])
        self.assertEqual(self.values(data, "cubase.plugin_mentions"), [{"name": "Pro EQ", "count": 5}])
        self.assertEqual(self.values(data, "project.plugin"), [{"name": "Pro EQ", "track": None}])
        references = {item["value"]["basename"]: item["value"]["count"] for item in self.items(data, "project.media_reference")}
        self.assertEqual(references, {f"{name} {number}.{ext}": 1 for name in ("Lead", "Second", "Keys", "Chords", "Bus")
                                      for number, ext in ((1, "wav"), (2, "WAV"))})
        self.assertNotIn("Secret", json.dumps(data))

    def test_tempo_conflicts_with_declaration(self):
        data = self.project(self.write(), "--declarations", self.declare_tempo(140))
        self.assertEqual([item["property"] for item in data["conflicts"]], ["tempo"])

    def test_deeper_failures_keep_the_application(self):
        cases = {
            "metadata": dict(meta=metadata()[:-5]),
            "no-metadata-records": dict(meta=container("CmObject")),
            "garbage-arrangement": dict(arrange=bytes(range(256)) * 20),
            "empty-arrangement": dict(arrange=b""),
        }
        for name, kwargs in cases.items():
            with self.subTest(name=name):
                data = self.project(self.write(f"{name}.cpr", **kwargs))
                self.assertEqual(data["sources"][0]["status"], "ok")
                self.assertEqual(self.values(data, "project.application"), ["Cubase 12.0.50"])
                codes = {item["code"] for item in data["diagnostics"]}
                self.assertTrue(codes & {"cubase_deep_parse_failed", "cubase_no_tracks_found"}, codes)
        data = self.project(self.write("old.cpr", version=app_version(extended=False)))
        self.assertEqual(self.values(data, "cubase.app_version")[0], {"name": "Cubase", "version": "Version 12.0.50",
                                                                       "buildDate": "Mar 30 2022"})

    def test_rejects_malformed_oversized_and_mislabeled_files(self):
        complete = self.write().read_bytes()
        cases = {
            "cut.cpr": (complete[:200], "truncated or corrupt"),
            "header-cut.cpr": (complete[:12 + 5], "truncated"),
            "oversized.cpr": (complete[:12] + b"ROOT" + struct.pack(">I", 2**31) + b"x" * 8, "declares"),
            "oversized-string.cpr": (riff(pair("Version", "PAppVersion", container("CmObject") + obj(
                "PAppVersion", struct.pack(">I", 2**30) + b"Cubase\0"))), "declares"),
            "no-version.cpr": (riff(pair("Arrangement1", "PArrangement", arrangement())), "no Version chunk"),
            "not-app-version.cpr": (riff(pair("Version", "PAppVersion", container("CmObject"))), "PAppVersion"),
            "bad-tag.cpr": (riff([b"\x00\x01\x02\x03" + struct.pack(">I", 0)]), "invalid tag"),
        }
        for name, (content, fragment) in cases.items():
            with self.subTest(name=name):
                (self.dir / name).write_bytes(content)
                self.assert_rejected(self.dir / name, fragment)
        for name, content in {"plain.cpr": b"RIFF", "wave.npr": b"RIFF\0\0\0\0WAVEfmt ", "text.cpr": b"hello"}.items():
            (self.dir / name).write_bytes(content)
            data = self.project(self.dir / name)
            self.assertEqual(data["sources"][0]["status"], "error", name)
            self.assertIn("does not match", self.error_message(data), name)

    def test_npr_extension_and_detect(self):
        path = self.write("song.npr", version=app_version(name="Nuendo", version="Version 12.0.70"))
        self.assertTrue(cubase.detect(path))
        data = self.project(path)
        self.assertEqual(self.values(data, "project.application"), ["Nuendo 12.0.70"])
        path.write_bytes(b"RIFF\0\0\0\0NUNX")
        self.assertFalse(cubase.detect(path))

    def test_hostile_object_stream_is_bounded_and_keeps_the_application(self):
        data = self.project(self.write(arrange=b"\xff" * (cubase.MAX_SCAN_HITS + 10)))
        self.assertEqual(self.values(data, "project.application"), ["Cubase 12.0.50"])
        self.assertTrue(any(item["code"] == "cubase_deep_parse_failed" and "too many candidate" in item["message"]
                            for item in data["diagnostics"]))


    def test_non_finite_metadata_floats_are_rejected_and_keep_the_application(self):
        for name, kwargs in {"nan-tempo": dict(tempo=float("nan")), "inf-rate": dict(rate=float("inf")),
                             "neg-inf-tempo": dict(tempo=float("-inf"))}.items():
            with self.subTest(name=name):
                result = self.run_cli("--project", self.write(f"{name}.cpr", meta=metadata(**kwargs)))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("NaN", result.stdout)
                self.assertNotIn("Infinity", result.stdout)
                data = json.loads(result.stdout)
                self.assertEqual(self.values(data, "project.application"), ["Cubase 12.0.50"])
                self.assertEqual(self.values(data, "tempo") + self.values(data, "project.sample_rate"), [])
                self.assertTrue(any(item["code"] == "cubase_deep_parse_failed" and "finite" in item["message"]
                                    for item in data["diagnostics"]))

    def test_implausible_tempo_and_sample_rate_are_reported_not_mapped(self):
        for name, kwargs, fragment in (("tempo", dict(tempo=1e12), "MusicalTempo"), ("fast", dict(tempo=5000.0), "MusicalTempo"),
                                       ("rate", dict(rate=1e12), "AudioSampleRate"), ("slow", dict(rate=3.0), "AudioSampleRate")):
            with self.subTest(name=name):
                data = self.project(self.write(f"{name}.cpr", meta=metadata(**kwargs)))
                self.assertEqual(self.values(data, "tempo") if "tempo" in kwargs else self.values(data, "project.sample_rate"), [])
                self.assertTrue(any(item["code"] == "cubase_value_out_of_range" and fragment in item["message"]
                                    for item in data["diagnostics"]), data["diagnostics"])
                self.assertEqual(len(self.values(data, "cubase.media_attribute")), 4)

    def test_media_name_scan_counts_every_examined_match(self):
        junk = b"x.wav\0\xef\xbb\xbf" * (cubase.MAX_MATCHES + 100)
        valid = cm("Late.wav", bom=True)
        data = self.project(self.write(arrange=arrangement(()) + junk + valid))
        self.assertNotIn("Late.wav", [item["value"]["basename"] for item in self.items(data, "project.media_reference")])
        self.assertTrue(any(item["code"] == "cubase_scan_limit" for item in data["diagnostics"]), data["diagnostics"])

    def test_plugin_facts_warn_that_built_in_modules_are_included(self):
        data = self.project(self.write())
        for item in self.items(data, "project.plugin"):
            self.assertIn("built-in", item["confidenceReason"])


if __name__ == "__main__":
    unittest.main()
