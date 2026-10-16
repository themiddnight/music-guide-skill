"""LMMS projects: .mmp (XML) and .mmpz (a Qt qCompress stream of the same XML).

LMMS itself decides by trial: it parses XML first and falls back to qUncompress, so this adapter does too.
Instrument files, samples and plugins named in the project are recorded as names only and never opened.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import Element

from ..model import Confidence, Detail, Diagnostic, Evidence
from ..notes import Note
from ..project import ProjectFacts, TrackKind
from .common import _finite_number, _parse_xml, _read_bounded

MAX_LMMS_BYTES = 32 * 1024 * 1024
MAX_LMMS_XML_BYTES = 32 * 1024 * 1024
MAX_TRACKS = 10_000
SNIFF_BYTES = 64
BOM = b"\xef\xbb\xbf"
XML_STARTS = (b"<?xml", b"<!DOCTYPE lmms-project", b"<lmms-project")
DOCTYPE = re.compile(rb"<!DOCTYPE\s+lmms-project\s*>")
# DefaultTicksPerTact is 192 ticks per whole note in every meter, so a quarter note is 48 ticks.
TICKS_PER_QUARTER = 48
PROJECT_TYPES = ("song", "projecttemplate")
# LMMS DataFile upgrades: file version 19 and older gain +12 on every key (UpgradeExtendedNoteRange, A4 moved from key 57
# to 69), and file version 26 and older turn negative-length notes into step notes of DefaultTicksPerBar / 16 ticks.
EXTENDED_RANGE_VERSION = 20
STEP_NOTE_VERSION = 27
STEP_NOTE_TICKS = 192 // 16
MIDI_KEYS = range(128)
# LMMS 1.2 tag names first, then the names LMMS 1.3 renamed them to.
SONG_TRACK_TYPES: dict[int, tuple[str, TrackKind]] = {
    0: ("instrument", "instrument"), 1: ("beat_bassline", "group"), 2: ("sample", "audio"), 3: ("event", "unknown"),
    4: ("video", "unknown"), 5: ("automation", "unknown"), 6: ("hidden_automation", "unknown")}
MIDI_CLIPS = ("pattern", "midiclip")
SAMPLE_CLIPS = ("sampletco", "sampleclip")
OTHER_CLIPS = ("bbtco", "patternclip", "automationpattern", "automationclip")
CLIPS = (*MIDI_CLIPS, *SAMPLE_CLIPS, *OTHER_CLIPS)
GROUP_TRACKS = ("bbtrack", "patterntrack")
MIXERS = ("fxmixer", "mixer")
CHANNELS = ("fxchannel", "mixerchannel")


def _looks_like_qcompress(head: bytes) -> bool:
    if len(head) < 6 or int.from_bytes(head[:4], "big") == 0:
        return False
    cmf, flags = head[4], head[5]
    return cmf & 0x0F == 8 and cmf >> 4 <= 7 and (cmf * 256 + flags) % 31 == 0 and not flags & 0x20


def detect(path: Path) -> bool:
    with path.open("rb") as stream:
        head = stream.read(SNIFF_BYTES)
    return head.removeprefix(BOM).lstrip().startswith(XML_STARTS) or _looks_like_qcompress(head)


def _inflate(raw: bytes) -> bytes:
    if not _looks_like_qcompress(raw[:6]):
        raise ValueError("file is neither LMMS XML nor a zlib stream with a 4-byte length prefix")
    declared = int.from_bytes(raw[:4], "big")
    if declared > MAX_LMMS_XML_BYTES:
        raise ValueError(f"compressed project declares {declared} bytes, over the {MAX_LMMS_XML_BYTES} byte limit")
    decompressor = zlib.decompressobj()
    try:
        data = decompressor.decompress(raw[4:], MAX_LMMS_XML_BYTES + 1)
    except zlib.error as error:
        raise ValueError(f"invalid compressed LMMS project: {error}") from error
    if len(data) > MAX_LMMS_XML_BYTES or decompressor.unconsumed_tail:
        raise ValueError(f"compressed project expands beyond {MAX_LMMS_XML_BYTES} byte limit")
    if not decompressor.eof:
        raise ValueError("compressed LMMS project is truncated")
    return data


def _load(path: Path) -> Element:
    raw = _read_bounded(path, MAX_LMMS_BYTES)
    text = raw.removeprefix(BOM).lstrip()
    xml = text if text.startswith(b"<") else _inflate(raw)
    # LMMS writes a bare "<!DOCTYPE lmms-project>"; the XML parser refuses any DTD, so only that exact line is removed.
    xml = DOCTYPE.sub(b"", xml[:1024], count=1) + xml[1024:]
    root = _parse_xml(xml)
    if root.tag != "lmms-project":
        raise ValueError("LMMS project root must be lmms-project")
    return root


def _attribute(element: Element, name: str, label: str) -> float:
    if name not in element.attrib:
        raise ValueError(f"{label} has no {name} attribute")
    try:
        return _finite_number(float(element.attrib[name]), f"{label} {name}")
    except ValueError as error:
        raise ValueError(f"{label} {name} must be a finite number") from error


def _whole(element: Element, name: str, label: str, minimum: int | None = None) -> int:
    value = _attribute(element, name, label)
    if not value.is_integer() or (minimum is not None and value < minimum):
        raise ValueError(f"{label} {name} must be a whole number" + (f" of at least {minimum}" if minimum is not None else ""))
    return int(value)


def _model_value(parent: Element, name: str) -> tuple[float, bool] | None:
    """A model LMMS saves as an attribute, or as a child element when it is automated or log-scaled."""
    if name in parent.attrib:
        return _attribute(parent, name, "head"), False
    child = parent.find(name)
    if child is not None:
        return _attribute(child, "value", f"head {name}"), True
    return None


@dataclass
class _Track:
    element: Element
    locator: str
    nested: bool


def _tracks(song: Element) -> list[_Track]:
    found: list[_Track] = []
    for container in song.findall("trackcontainer"):
        for position, track in enumerate(container.findall("track")):
            locator = f"song/trackcontainer/track[{position}]"
            found.append(_Track(track, locator, False))
            for group_tag in GROUP_TRACKS:
                for store in track.findall(f"{group_tag}/trackcontainer"):
                    for inner, nested in enumerate(store.findall("track")):
                        found.append(_Track(nested, f"{locator}/{group_tag}/trackcontainer/track[{inner}]", True))
        if len(found) > MAX_TRACKS:
            raise ValueError(f"project has more than {MAX_TRACKS} tracks")
    return found


def _specific(track: Element) -> list[Element]:
    return [child for child in track if child.tag not in CLIPS]


@dataclass(frozen=True)
class _Convention:
    """How this file's keys and step-note lengths map to MIDI, from LMMS's own file-version upgrades."""

    key_offset: int
    convert_steps: bool
    confidence: Confidence
    reason: str


def _convention(root: Element) -> _Convention:
    version = root.attrib.get("version", "").strip()
    creator = root.attrib.get("creatorversion", "").strip()
    if version in ("", "1.0"):
        # LMMS derives the file version of these files from creatorversion; every release it can name predates both upgrades.
        if creator:
            return _Convention(12, True, "high", f"The file predates integer file versions (created with LMMS {creator[:40]}), "
                               "so LMMS upgrades it on load: stored key 57 is A4 and the reported pitch is key + 12.")
        return _Convention(12, True, "low", "The file stores neither a file version nor a creator version, so the oldest "
                           "convention was assumed (LMMS does the same): stored key 57 is A4 and the reported pitch is key + 12.")
    if version.isascii() and version.isdigit() and len(version) < 6:
        number = int(version)
        if number >= EXTENDED_RANGE_VERSION:
            return _Convention(0, number < STEP_NOTE_VERSION, "high", f"Stored by LMMS file version {number}, which keys notes "
                               "as MIDI numbers: stored key 69 is A4 and the reported pitch equals the key.")
        return _Convention(12, True, "high", f"Stored by LMMS file version {number}, before the extended note range: "
                           "stored key 57 is A4 and the reported pitch is key + 12.")
    return _Convention(12, True, "low", "The LMMS file version is not a number, so the oldest convention was assumed "
                       "(LMMS does the same): stored key 57 is A4 and the reported pitch is key + 12.")


@dataclass
class _NoteCounts:
    out_of_range: int = 0
    converted_steps: int = 0
    excluded_steps: int = 0


def _notes(track: Element, locator: str, convention: _Convention, counts: _NoteCounts) -> list[Note]:
    notes: list[Note] = []
    for tag in MIDI_CLIPS:
        for number, clip in enumerate(track.findall(tag)):
            label = f"{locator}/{tag}[{number}]"
            start = _whole(clip, "pos", label, 0)
            for note in clip.findall("note"):
                key = _whole(note, "key", f"{label}/note")
                length = _whole(note, "len", f"{label}/note")
                position = start + _whole(note, "pos", f"{label}/note", 0)
                volume = _attribute(note, "vol", f"{label}/note")
                if volume < 0:
                    raise ValueError(f"{label}/note has a negative volume")
                pitch = key + convention.key_offset
                if pitch not in MIDI_KEYS:
                    counts.out_of_range += 1
                    continue
                if length < 0:
                    if not convention.convert_steps:
                        counts.excluded_steps += 1
                        continue
                    length = STEP_NOTE_TICKS
                    counts.converted_steps += 1
                notes.append(Note(pitch, min(127, round(volume * 127 / 200)), position / TICKS_PER_QUARTER,
                                  (position + length) / TICKS_PER_QUARTER))
    return notes


def lmms(path: Path, source_id: str, evidence: Evidence, detail: Detail) -> None:
    root = _load(path)
    if root.attrib.get("type") not in PROJECT_TYPES:
        raise ValueError(f"LMMS file type {root.attrib.get('type')!r} is not a song project")
    method = "lmms-xml"
    facts = ProjectFacts(evidence, source_id, method, detail)

    def extra(property: str, value: Any, unit: str, locator: str, reason: str, scope: str = "project",
              time_domain: str = "static") -> None:
        evidence.add(source_id, property, value, unit, scope, time_domain, locator, method, "extracted", "high", reason)

    application = " ".join(root.attrib.get(name, "").strip() for name in ("creator", "creatorversion")).strip()
    if application:
        facts.application(application, "lmms-project/@creator,@creatorversion")
    for name, label in (("version", "format_version"), ("type", "project_type")):
        if root.attrib.get(name):
            extra(f"lmms.{label}", root.attrib[name], "text", f"lmms-project/@{name}", f"Stored LMMS {label.replace('_', ' ')}.")
    head = root.find("head")
    if head is not None:
        tempo = _model_value(head, "bpm")
        if tempo is not None:
            value, automated = tempo
            reason = ("Tempo stored as an element, which LMMS writes when the tempo is automated or log-scaled; this is the value at save "
                      "time and automation is not read." if automated else
                      "Stored project tempo; tempo automation is not reconstructed.")
            facts.tempo(value, "head/@bpm" if not automated else "head/bpm/@value", reason)
            if automated:
                evidence.diagnostics.append(Diagnostic("tempo_model_is_element", "warning",
                    "The LMMS tempo is stored as an element, so it may be automated; the static tempo is the value at save time.", source_id))
        if "timesig_numerator" in head.attrib and "timesig_denominator" in head.attrib:
            facts.meter(_attribute(head, "timesig_numerator", "head"), _attribute(head, "timesig_denominator", "head"),
                        "head/@timesig_numerator,@timesig_denominator")
    song = root.find("song")
    if song is None:
        return
    convention = _convention(root)
    counts = _NoteCounts()
    tracks = _tracks(song)
    facts.track_count(len(tracks), "song/trackcontainer/track")
    media: list[str] = []
    noted = 0
    for index, item in enumerate(tracks):
        element, locator = item.element, item.locator
        code = _whole(element, "type", locator, 0)
        label, kind = SONG_TRACK_TYPES.get(code, (f"type_{code}", "unknown"))
        facts.track(index, element.attrib.get("name"), kind, locator)
        extra("lmms.track_type", {"code": code, "label": label, "container": "pattern_store" if item.nested else "song"},
              "track_type", f"{locator}/@type", "LMMS track type code.", f"project_track:{index}")
        for specific in _specific(element):
            instrument = specific.find("instrument")
            if instrument is not None and instrument.attrib.get("name"):
                facts.plugin(instrument.attrib["name"], index, f"{locator}/{specific.tag}/instrument/@name")
                media.extend(child.attrib["src"] for child in instrument if child.attrib.get("src"))
            for effect in specific.findall("fxchain/effect"):
                if effect.attrib.get("name"):
                    facts.plugin(effect.attrib["name"], index, f"{locator}/{specific.tag}/fxchain/effect/@name")
        for tag in SAMPLE_CLIPS:
            media.extend(clip.attrib["src"] for clip in element.findall(tag) if clip.attrib.get("src"))
        clips = sum(len(element.findall(tag)) for tag in CLIPS)
        if clips:
            facts.clip_count(clips, f"{locator}/clips", index)
        notes = _notes(element, locator, convention, counts)
        noted += len(notes)
        facts.track_notes(index, notes, f"{locator}/pattern/note",
                          "Aggregated from stored notes; beats are quarter notes, not audible timing. " + convention.reason,
                          convention.confidence)
    for mixer_tag, channel_tag in zip(MIXERS, CHANNELS):
        for channel in song.findall(f"{mixer_tag}/{channel_tag}"):
            for effect in channel.findall("fxchain/effect"):
                if effect.attrib.get("name"):
                    facts.plugin(effect.attrib["name"], None, f"song/{mixer_tag}/{channel_tag}/fxchain/effect/@name")
    facts.media_references(media, "song/(instrument and sample sources)")
    timeline = song.find("timeline")
    if timeline is not None and {"lp0pos", "lp1pos"} <= timeline.attrib.keys():
        extra("lmms.loop_points", {"startBeat": _attribute(timeline, "lp0pos", "timeline") / TICKS_PER_QUARTER,
                                   "endBeat": _attribute(timeline, "lp1pos", "timeline") / TICKS_PER_QUARTER,
                                   "enabled": timeline.attrib.get("lpstate") == "1"},
              "loop", "song/timeline", "Stored loop points; they do not mark arrangement sections.",
              time_domain="project_beats")
    if counts.converted_steps:
        evidence.diagnostics.append(Diagnostic("negative_note_length", "info",
            f"{counts.converted_steps} note(s) store a negative length (beat step notes). LMMS's own upgrade sets their length "
            f"to DefaultTicksPerBar/16 = 192/16 = {STEP_NOTE_TICKS} ticks, and that length is used.", source_id))
    if counts.excluded_steps:
        evidence.diagnostics.append(Diagnostic("negative_note_length", "warning",
            f"{counts.excluded_steps} note(s) store a negative length, which LMMS stopped writing at file version "
            f"{STEP_NOTE_VERSION}; they were excluded because no length can be confirmed.", source_id))
    if counts.out_of_range:
        evidence.diagnostics.append(Diagnostic("lmms_notes_out_of_range", "warning",
            f"{counts.out_of_range} note(s) were skipped because their key falls outside MIDI 0-127 under the convention "
            f"used here (key offset {convention.key_offset:+d}).", source_id))
    if noted:
        evidence.add(source_id, "lmms.pitch_convention", {"keyOffset": convention.key_offset, "a4Key": 69 - convention.key_offset},
                     "convention", "project", "static", "lmms-project/@version,@creatorversion", method, "extracted",
                     convention.confidence, convention.reason)
        evidence.diagnostics.append(Diagnostic("lmms_pitch_convention", "info",
            f"{convention.reason} Base-note and master-pitch transposition are not applied. "
            "Note positions inside a beat/bassline pattern are local to that pattern.", source_id))
