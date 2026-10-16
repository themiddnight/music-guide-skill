"""MusicXML 4.0 (partwise and timewise, plain or compressed .mxl) to the shared score vocabulary."""

from __future__ import annotations

import re
from fractions import Fraction
from pathlib import Path
from xml.etree.ElementTree import Element

from ..model import Detail, Diagnostic, Evidence
from ..notes import Note
from ..project import ProjectFacts
from .common import _child, _local, _parse_xml, _read_bounded, _read_zip_member
from .scores import ScoreData, ScorePart, emit_score

MAX_SCORE_XML_BYTES = 16 * 1024 * 1024
MAX_CONTAINER_BYTES = 256 * 1024
DEFAULT_VELOCITY = 90
ZIP_MAGIC = b"PK\x03\x04"
MEDIA_TYPES = {"application/vnd.recordare.musicxml+xml", "application/vnd.recordare.musicxml"}
STEPS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
# MusicXML beat-unit values in quarter notes.
BEAT_UNITS = {"maxima": 32, "long": 16, "breve": 8, "whole": 4, "half": 2, "quarter": 1, "eighth": 0.5,
              "16th": 0.25, "32nd": 0.125, "64th": 0.0625, "128th": 0.03125, "256th": 0.015625,
              "512th": 0.0078125, "1024th": 0.00390625}
# The public DTD identifiers MusicXML files carry. A DOCTYPE with an internal subset is never accepted.
DOCTYPE = re.compile(rb"<!DOCTYPE\s+score-(?:partwise|timewise)\s+PUBLIC\s+"
                     rb"(?:\"-//Recordare//DTD MusicXML [0-9.]+ (?:Partwise|Timewise)//EN\"|"
                     rb"'-//Recordare//DTD MusicXML [0-9.]+ (?:Partwise|Timewise)//EN')\s+"
                     rb"(?:\"https?://www\.musicxml\.org/dtds/(?:partwise|timewise)\.dtd\"|"
                     rb"'https?://www\.musicxml\.org/dtds/(?:partwise|timewise)\.dtd')\s*>")
PROLOG_BYTES = 4096


def _strip_public_doctype(raw: bytes) -> bytes:
    # Nearly every exported MusicXML file declares its public DTD, which the safe parser would refuse.
    # Removing exactly that declaration keeps entity and external-subset protection intact.
    head = raw[:PROLOG_BYTES]
    match = DOCTYPE.search(head)
    root = re.search(rb"<score-(?:partwise|timewise)", head)
    if match is None or (root is not None and root.start() < match.start()):
        return raw
    return raw[:match.start()] + raw[match.end():]


def _rootfile(container: Element) -> str:
    for item in container.iter():
        if _local(item.tag) == "rootfile":
            media_type = item.attrib.get("media-type")
            if media_type is None or media_type in MEDIA_TYPES:
                path = item.attrib.get("full-path", "").strip()
                if path:
                    return path
    raise ValueError("META-INF/container.xml names no MusicXML rootfile")


def _read_score(path: Path) -> bytes:
    with path.open("rb") as stream:
        compressed = stream.read(len(ZIP_MAGIC)) == ZIP_MAGIC
    if not compressed:
        return _read_bounded(path, MAX_SCORE_XML_BYTES)
    try:
        container = _parse_xml(_read_zip_member(path, "META-INF/container.xml", MAX_CONTAINER_BYTES))
    except ValueError as error:
        raise ValueError(f"compressed MusicXML needs a readable META-INF/container.xml: {error}") from error
    return _read_zip_member(path, _rootfile(container), MAX_SCORE_XML_BYTES)


def _text(parent: Element, name: str) -> str | None:
    item = _child(parent, name)
    return item.text.strip() if item is not None and item.text and item.text.strip() else None


def _fraction(parent: Element, name: str, where: str, *, positive: bool = False,
              signed: bool = False) -> Fraction | None:
    text = _text(parent, name)
    if text is None:
        return None
    try:
        value = Fraction(text)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{where}: <{name}> must be a number, not {text!r}") from error
    if not signed and (value < 0 or (positive and value == 0)):
        raise ValueError(f"{where}: <{name}> must be {'positive' if positive else 'nonnegative'}")
    return value


def _integer(parent: Element, name: str, where: str) -> int | None:
    text = _text(parent, name)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError as error:
        raise ValueError(f"{where}: <{name}> must be an integer, not {text!r}") from error


def _midi_pitch(note: Element, transposition: int, where: str) -> int:
    pitch = _child(note, "pitch")
    step = (_text(pitch, "step") or "").upper()
    octave = _integer(pitch, "octave", where)
    if step not in STEPS or octave is None:
        raise ValueError(f"{where}: <pitch> needs a step A-G and an octave")
    alter = _fraction(pitch, "alter", where, signed=True) or 0
    value = (octave + 1) * 12 + STEPS[step] + round(alter) + transposition
    if not 0 <= value <= 127:
        raise ValueError(f"{where}: pitch {value} is outside the MIDI range 0-127")
    return value


def _velocity(note: Element) -> int:
    dynamics = note.attrib.get("dynamics")
    if dynamics is not None:
        try:
            # 100 percent of the default forte is MIDI velocity 90.
            return max(1, min(127, round(float(dynamics) * 0.9)))
        except ValueError:
            pass
    return DEFAULT_VELOCITY


def _metronome_bpm(metronome: Element) -> float | None:
    unit = _text(metronome, "beat-unit")
    per_minute = _text(metronome, "per-minute")
    if unit not in BEAT_UNITS or per_minute is None or _child(metronome, "beat-unit-tied") is not None:
        return None
    try:
        value = float(per_minute)
    except ValueError:
        return None
    quarters = BEAT_UNITS[unit]
    dots = sum(1 for item in metronome if _local(item.tag) == "beat-unit-dot")
    quarters *= 2 - 0.5 ** dots
    bpm = value * quarters
    return bpm if bpm > 0 else None


def _positive_tempo(text: str, where: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"{where}: sound tempo must be a number, not {text!r}") from error
    if not 0 < value < float("inf"):
        raise ValueError(f"{where}: sound tempo must be positive")
    return value


class _PartReader:
    def __init__(self, part_id: str, data: ScoreData, sounds: list[tuple[float, float]],
                 metronomes: list[tuple[float, float]], flags: set[str]) -> None:
        self.id = part_id
        self.data = data
        self.sounds = sounds
        self.metronomes = metronomes
        self.flags = flags
        self.divisions: Fraction | None = None
        self.transposition = 0
        self.position = Fraction(0)
        self.notes: list[list] = []
        self.open_ties: dict[tuple[str, str, int], int] = {}

    @property
    def transposing(self) -> bool:
        return self.transposition != 0

    def measures(self, measures: list[Element]) -> list[Note]:
        for number, measure in enumerate(measures, 1):
            self._measure(measure, f"part {self.id} measure {measure.attrib.get('number', number)}")
        self.data.measures = max(self.data.measures, len(measures))
        self.data.length_beats = max(self.data.length_beats, float(self.position))
        return [Note(pitch, velocity, float(start), float(end), None) for pitch, velocity, start, end in self.notes]

    def _beats(self, element: Element, where: str) -> Fraction:
        if self.divisions is None:
            raise ValueError(f"{where}: <divisions> must appear before the first duration")
        duration = _fraction(element, "duration", where)
        if duration is None:
            raise ValueError(f"{where}: <{_local(element.tag)}> has no <duration>")
        return duration / self.divisions

    def _measure(self, measure: Element, where: str) -> None:
        start = extent = self.position
        last_start = start
        for item in measure:
            name = _local(item.tag)
            if name == "attributes":
                self._attributes(item, where)
            elif name == "note":
                if _child(item, "grace") is not None:
                    self.flags.add("grace")
                    continue
                chord = _child(item, "chord") is not None
                beats = self._beats(item, where)
                begin = last_start if chord else self.position
                if not chord:
                    last_start = begin
                    self.position += beats
                self._note(item, begin, begin + beats, where)
            elif name == "backup":
                self.position -= self._beats(item, where)
                if self.position < start:
                    raise ValueError(f"{where}: <backup> moves before the start of the measure")
            elif name == "forward":
                self.position += self._beats(item, where)
            elif name == "direction":
                self._direction(item, where)
            elif name == "sound":
                self._sound(item, where)
            elif name == "barline" and _child(item, "repeat") is not None:
                self.flags.add("repeat")
            extent = max(extent, self.position)
        self.position = extent

    def _attributes(self, attributes: Element, where: str) -> None:
        divisions = _fraction(attributes, "divisions", where, positive=True)
        if divisions is not None:
            self.divisions = divisions
        transpose = _child(attributes, "transpose")
        if transpose is not None:
            chromatic = _integer(transpose, "chromatic", where) or 0
            octave = _integer(transpose, "octave-change", where) or 0
            self.transposition = chromatic + 12 * octave
        beat = float(self.position)
        for key in (item for item in attributes if _local(item.tag) == "key"):
            fifths = _integer(key, "fifths", where)
            if fifths is not None and not self.transposing:
                self.data.keys.append((beat, fifths, _text(key, "mode")))
        for time in (item for item in attributes if _local(item.tag) == "time"):
            beats, beat_type = _text(time, "beats"), _text(time, "beat-type")
            if beats and beat_type:
                try:
                    numerator = sum(int(part) for part in beats.split("+"))
                    denominator = int(beat_type)
                except ValueError:
                    self.flags.add("irregular_meter")
                    continue
                if numerator > 0 and denominator > 0:
                    self.data.meters.append((beat, numerator, denominator))

    def _direction(self, direction: Element, where: str) -> None:
        beat = float(self.position)
        for item in direction:
            if _local(item.tag) == "sound":
                self._sound(item, where)
            elif _local(item.tag) == "direction-type":
                for kind in item:
                    if _local(kind.tag) == "metronome":
                        bpm = _metronome_bpm(kind)
                        if bpm is None:
                            self.flags.add("unreadable_metronome")
                        else:
                            self.metronomes.append((beat, bpm))
                    elif _local(kind.tag) == "rehearsal" and kind.text and kind.text.strip():
                        self.data.markers.append((kind.text.strip(), beat))

    def _sound(self, sound: Element, where: str) -> None:
        if "tempo" in sound.attrib:
            self.sounds.append((float(self.position), _positive_tempo(sound.attrib["tempo"], where)))

    def _note(self, note: Element, begin: Fraction, end: Fraction, where: str) -> None:
        if _child(note, "rest") is not None or _child(note, "cue") is not None:
            return
        if _child(note, "pitch") is None:
            self.flags.add("unpitched")
            return
        pitch = _midi_pitch(note, self.transposition, where)
        ties = {item.attrib.get("type") for item in note if _local(item.tag) == "tie"}
        key = (_text(note, "voice") or "1", _text(note, "staff") or "1", pitch)
        if "stop" in ties and key in self.open_ties:
            index = self.open_ties[key]
            self.notes[index][3] = end
            if "start" not in ties:
                del self.open_ties[key]
            return
        self.notes.append([pitch, _velocity(note), begin, end])
        if "start" in ties:
            self.open_ties[key] = len(self.notes) - 1
        else:
            self.open_ties.pop(key, None)


def _score_parts(root: Element) -> tuple[list[tuple[str, list[Element]]], dict[str, Element]]:
    declared: dict[str, Element] = {}
    part_list = _child(root, "part-list")
    if part_list is not None:
        for item in part_list:
            if _local(item.tag) == "score-part" and item.attrib.get("id"):
                declared.setdefault(item.attrib["id"], item)
    bodies: dict[str, list[Element]] = {}
    if _local(root.tag) == "score-partwise":
        for part in root:
            if _local(part.tag) == "part":
                bodies.setdefault(part.attrib.get("id", ""), []).extend(
                    item for item in part if _local(item.tag) == "measure")
    else:
        for measure in root:
            if _local(measure.tag) == "measure":
                for part in measure:
                    if _local(part.tag) == "part":
                        bodies.setdefault(part.attrib.get("id", ""), []).append(part)
    order = [*declared, *(name for name in bodies if name not in declared)]
    return [(name, bodies.get(name, [])) for name in order], declared


def _metadata(evidence: Evidence, source_id: str, root: Element) -> str | None:
    def add(prop: str, value: object, unit: str, locator: str) -> None:
        evidence.add(source_id, prop, value, unit, "project", "static", locator, "musicxml", "extracted", "high",
                     "Stored in the score file.")

    if "version" in root.attrib:
        add("musicxml.version", root.attrib["version"], "version", "score/@version")
    work = _child(root, "work")
    title = (_text(work, "work-title") if work is not None else None) or _text(root, "movement-title")
    if title:
        add("musicxml.title", title[:500], "text", "work-title or movement-title")
    identification = _child(root, "identification")
    if identification is None:
        return None
    for creator in (item for item in identification if _local(item.tag) == "creator"):
        if creator.text and creator.text.strip():
            add("musicxml.creator", {"type": creator.attrib.get("type"), "name": creator.text.strip()[:500]},
                "creator", "identification/creator")
    encoding = _child(identification, "encoding")
    software = _text(encoding, "software") if encoding is not None else None
    return software[:200] if software else None


def musicxml(path: Path, source_id: str, evidence: Evidence, detail: Detail) -> None:
    root = _parse_xml(_strip_public_doctype(_read_score(path)))
    if _local(root.tag) not in {"score-partwise", "score-timewise"}:
        raise ValueError("MusicXML root must be score-partwise or score-timewise")
    software = _metadata(evidence, source_id, root)
    data = ScoreData(default_velocity=DEFAULT_VELOCITY)
    sounds: list[tuple[float, float]] = []
    metronomes: list[tuple[float, float]] = []
    flags: set[str] = set()
    parts, declared = _score_parts(root)
    if not parts:
        raise ValueError("MusicXML score has no parts")
    for part_id, measures in parts:
        score_part = declared.get(part_id)
        name = _text(score_part, "part-name") if score_part is not None else None
        reader = _PartReader(part_id, data, sounds, metronomes, flags)
        notes = reader.measures(measures)
        first = None if score_part is None else next(
            (item for item in score_part if _local(item.tag) == "score-instrument"), None)
        instrument = _text(first, "instrument-name") if first is not None else None
        data.parts.append(ScorePart(name, f"part[@id={part_id!r}]", notes, instrument))
    chosen, data.tempo_locator = ((sounds, "sound/@tempo") if sounds else
                                  (metronomes, "direction-type/metronome"))
    data.tempos = chosen
    if software:
        ProjectFacts(evidence, source_id, "musicxml", detail).application(software, "identification/encoding/software")
    emit_score(evidence, source_id, "musicxml", "musicxml", detail, data)
    _flags(evidence, source_id, flags, bool(sounds), bool(metronomes))


def _flags(evidence: Evidence, source_id: str, flags: set[str], has_sound: bool, has_metronome: bool) -> None:
    notices = {
        "repeat": "Repeats and jumps are not unrolled; each stored measure is read once.",
        "grace": "Grace notes have no duration and are skipped.",
        "unpitched": "Unpitched (percussion) notes are skipped because they carry no pitch.",
        "irregular_meter": "A time signature without integer beats (for example senza misura) was skipped.",
        "unreadable_metronome": "A metronome mark that is not a plain number of one beat unit was skipped.",
    }
    for flag in sorted(flags & notices.keys()):
        evidence.diagnostics.append(Diagnostic(f"musicxml_{flag}", "info", notices[flag], source_id))
    if has_metronome and not has_sound:
        evidence.diagnostics.append(Diagnostic(
            "musicxml_tempo_from_metronome", "info",
            "The score has no playback tempo (sound/@tempo), so tempo comes from displayed metronome marks.",
            source_id))
