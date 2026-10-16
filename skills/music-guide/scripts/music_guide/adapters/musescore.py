"""MuseScore .mscz (a zip holding one .mscx XML score) to the shared score vocabulary."""

from __future__ import annotations

import math
from collections import Counter
from fractions import Fraction
from pathlib import Path
from xml.etree.ElementTree import Element
import zipfile

from ..model import Detail, Diagnostic, Evidence
from ..notes import Note
from ..project import ProjectFacts
from .common import _child, _local, _parse_xml, _read_zip_member
from .scores import ScoreData, ScorePart, emit_score
from .textnum import beat_float, parse_integer, parse_ratio

MAX_SCORE_XML_BYTES = 16 * 1024 * 1024
MAX_CONTAINER_BYTES = 256 * 1024
DEFAULT_VELOCITY = 80
WHOLE_NOTES = {"long": Fraction(4), "breve": Fraction(2), "whole": Fraction(1), "half": Fraction(1, 2),
                "quarter": Fraction(1, 4), "eighth": Fraction(1, 8), "16th": Fraction(1, 16),
                "32nd": Fraction(1, 32), "64th": Fraction(1, 64), "128th": Fraction(1, 128),
                "256th": Fraction(1, 256), "512th": Fraction(1, 512), "1024th": Fraction(1, 1024)}
GRACE_TAGS = {"acciaccatura", "appoggiatura", "grace4", "grace8", "grace8after", "grace16", "grace16after",
              "grace32", "grace32after"}
REPEAT_TAGS = {"startRepeat", "endRepeat", "Jump", "Marker"}
QUARTERS_PER_WHOLE = 4
MAX_DOTS = 16
MAX_TUPLET_DEPTH = 16
MAX_SIGNATURE = 1_000_000


def _score_member(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            names = [item.filename for item in archive.infolist()]
    except zipfile.BadZipFile as error:
        raise ValueError(f"cannot read MuseScore archive: {error}") from error
    if "META-INF/container.xml" in names:
        container = _parse_xml(_read_zip_member(path, "META-INF/container.xml", MAX_CONTAINER_BYTES))
        for item in container.iter():
            if _local(item.tag) == "rootfile" and item.attrib.get("full-path", "").strip():
                return item.attrib["full-path"].strip()
    top_level = [name for name in names if name.lower().endswith(".mscx") and "/" not in name]
    if len(top_level) != 1:
        raise ValueError("MuseScore archive must name one score in META-INF/container.xml or hold exactly one "
                         "top-level .mscx file")
    return top_level[0]


def _text(parent: Element | None, name: str) -> str | None:
    item = _child(parent, name) if parent is not None else None
    return item.text.strip() if item is not None and item.text and item.text.strip() else None


def _fraction(text: str | None, where: str, label: str) -> Fraction | None:
    if text is None:
        return None
    value = parse_ratio(text)
    if value is None:
        raise ValueError(f"{where}: {label} must be a fraction such as 1/4, not {text[:40]!r}")
    return value


def _integer(text: str | None, where: str, label: str) -> int | None:
    if text is None:
        return None
    value = parse_integer(text)
    if value is None:
        raise ValueError(f"{where}: {label} must be an integer, not {text[:40]!r}")
    return value


class _StaffReader:
    def __init__(self, staff_id: str, part: ScorePart, transposing: bool, data: ScoreData, flags: Counter[str]) -> None:
        self.id = staff_id
        self.part = part
        self.transposing = transposing
        self.data = data
        self.flags = flags
        self.signature = Fraction(4, 4)
        self.signature_known = False
        self.start = Fraction(0)
        self.open_ties: dict[tuple[int, int], list] = {}
        self.notes: list[list] = []

    def read(self, staff: Element) -> list[Note]:
        measures = [item for item in staff if _local(item.tag) == "Measure"]
        for number, measure in enumerate(measures, 1):
            self._measure(measure, f"staff {self.id} measure {number}")
        self.data.measures = max(self.data.measures, len(measures))
        self.data.length_beats = max(self.data.length_beats, beat_float(self.start * QUARTERS_PER_WHOLE, "score length"))
        return [Note(pitch, DEFAULT_VELOCITY, beat_float(begin * QUARTERS_PER_WHOLE), beat_float(end * QUARTERS_PER_WHOLE),
                     None) for pitch, begin, end in self.notes]

    def _beat(self, cursor: Fraction) -> float:
        return beat_float((self.start + cursor) * QUARTERS_PER_WHOLE)

    def _measure(self, measure: Element, where: str) -> None:
        declared = _fraction(measure.attrib.get("len"), where, "Measure len")
        extent = Fraction(0)
        voices = 0
        for item in measure:
            name = _local(item.tag)
            if name in REPEAT_TAGS:
                self.flags["repeat"] += 1
            elif name == "multiMeasureRest":
                self.flags["multimeasure"] += 1
            elif name == "voice":
                extent = max(extent, self._voice(item, voices, declared, where))
                voices += 1
        self.start += declared or (self.signature if self.signature_known else extent)

    def _voice(self, voice: Element, number: int, declared: Fraction | None, where: str) -> Fraction:
        cursor = Fraction(0)
        tuplets: list[Fraction] = []
        for item in voice:
            name = _local(item.tag)
            if name == "TimeSig":
                numerator = _integer(_text(item, "sigN"), where, "sigN")
                denominator = _integer(_text(item, "sigD"), where, "sigD")
                if numerator and denominator and 0 < numerator <= MAX_SIGNATURE and 0 < denominator <= MAX_SIGNATURE:
                    self.signature = Fraction(numerator, denominator)
                    self.signature_known = True
                    self.data.meters.append((self._beat(cursor), numerator, denominator))
            elif name == "KeySig":
                fifths = _integer(_text(item, "accidental"), where, "KeySig accidental")
                if fifths is not None and not self.transposing:
                    self.data.keys.append((self._beat(cursor), fifths, _text(item, "mode")))
            elif name == "Tempo":
                tempo = _text(item, "tempo")
                try:
                    bps = float(tempo) if tempo is not None else 0.0
                except ValueError as error:
                    raise ValueError(f"{where}: Tempo tempo must be a number, not {tempo[:40]!r}") from error
                if not math.isfinite(bps * 60):
                    raise ValueError(f"{where}: Tempo tempo must be finite")
                if bps > 0:
                    self.data.tempos.append((self._beat(cursor), bps * 60))
            elif name == "RehearsalMark":
                label = "".join(_child(item, "text").itertext()).strip() if _child(item, "text") is not None else ""
                if label:
                    self.data.markers.append((label, self._beat(cursor)))
            elif name == "location":
                cursor += _fraction(_text(item, "fractions"), where, "location fractions") or 0
            elif name == "Tuplet":
                normal = _integer(_text(item, "normalNotes"), where, "normalNotes")
                actual = _integer(_text(item, "actualNotes"), where, "actualNotes")
                if not normal or not actual or normal < 0 or actual < 0 or len(tuplets) >= MAX_TUPLET_DEPTH:
                    raise ValueError(f"{where}: Tuplet needs positive normalNotes and actualNotes, nested at most "
                                     f"{MAX_TUPLET_DEPTH} deep")
                tuplets.append(Fraction(normal, actual))
            elif name == "endTuplet":
                if tuplets:
                    tuplets.pop()
            elif name in {"Chord", "Rest", "RepeatMeasure"}:
                length = self._length(item, declared, tuplets, where)
                if name == "Chord" and not GRACE_TAGS & {_local(child.tag) for child in item}:
                    self._chord(item, number, cursor, cursor + length, where)
                elif name == "Chord":
                    self.flags["grace"] += 1
                    continue
                elif name == "RepeatMeasure":
                    self.flags["repeat_measure"] += 1
                cursor += length
        return cursor

    def _length(self, item: Element, declared: Fraction | None, tuplets: list[Fraction], where: str) -> Fraction:
        measure_length = declared or self.signature
        if _local(item.tag) == "RepeatMeasure":
            return measure_length
        explicit = _fraction(_text(item, "duration"), where, "duration")
        kind = _text(item, "durationType")
        if explicit is not None:
            if explicit < 0:
                raise ValueError(f"{where}: duration must not be negative")
            length = explicit
        elif kind == "measure":
            length = measure_length
        elif kind in WHOLE_NOTES:
            dots = _integer(_text(item, "dots"), where, "dots") or 0
            if not 0 <= dots <= MAX_DOTS:
                raise ValueError(f"{where}: dots must be between 0 and {MAX_DOTS}")
            length = WHOLE_NOTES[kind] * (2 - Fraction(1, 2 ** dots))
        else:
            raise ValueError(f"{where}: {_local(item.tag)} has unknown durationType {kind!r}")
        for ratio in tuplets:
            length *= ratio
        return length

    def _chord(self, chord: Element, voice: int, begin: Fraction, end: Fraction, where: str) -> None:
        for note in (item for item in chord if _local(item.tag) == "Note"):
            pitch = _integer(_text(note, "pitch"), where, "Note pitch")
            if pitch is None or not 0 <= pitch <= 127:
                raise ValueError(f"{where}: Note pitch must be a MIDI number 0-127")
            if _text(note, "play") == "0":
                self.flags["unsounded"] += 1
                continue
            absolute_begin, absolute_end = self.start + begin, self.start + end
            spanners = {_local(link.tag) for spanner in note if _local(spanner.tag) == "Spanner"
                        and spanner.attrib.get("type") == "Tie" for link in spanner}
            key = (voice, pitch)
            open_note = self.open_ties.get(key)
            if "prev" in spanners and open_note is not None:
                open_note[2] = absolute_end
                if "next" not in spanners:
                    del self.open_ties[key]
                continue
            entry = [pitch, absolute_begin, absolute_end]
            self.notes.append(entry)
            if "next" in spanners:
                self.open_ties[key] = entry
            else:
                self.open_ties.pop(key, None)


def _meta(evidence: Evidence, source_id: str, score: Element) -> None:
    tags = {item.attrib.get("name"): (item.text or "").strip() for item in score
            if _local(item.tag) == "metaTag"}
    if tags.get("workTitle"):
        evidence.add(source_id, "musescore.title", tags["workTitle"][:500], "text", "project", "static",
                     "Score/metaTag[workTitle]", "musescore-mscx", "extracted", "high", "Stored in the score file.")
    if tags.get("composer"):
        evidence.add(source_id, "musescore.creator", {"type": "composer", "name": tags["composer"][:500]}, "creator",
                     "project", "static", "Score/metaTag[composer]", "musescore-mscx", "extracted", "high",
                     "Stored in the score file.")


def _staff_ids(part: Element) -> list[str]:
    return [item.attrib["id"] for item in part if _local(item.tag) == "Staff" and "id" in item.attrib]


def musescore(path: Path, source_id: str, evidence: Evidence, detail: Detail) -> None:
    root = _parse_xml(_read_zip_member(path, _score_member(path), MAX_SCORE_XML_BYTES))
    if _local(root.tag) != "museScore":
        raise ValueError("MuseScore score root must be museScore")
    score = _child(root, "Score")
    if score is None:
        raise ValueError("MuseScore score has no Score element")
    program = _text(root, "programVersion") or _text(score, "programVersion")
    method = "musescore-mscx"
    file_version = root.attrib.get("version")
    if file_version:
        evidence.add(source_id, "musescore.version", file_version, "version", "project", "static", "museScore/@version",
                     method, "extracted", "high", "Stored score file format version.")
    _meta(evidence, source_id, score)
    staves = {item.attrib.get("id"): item for item in score
              if _local(item.tag) == "Staff" and _child(item, "Measure") is not None}
    data = ScoreData(default_velocity=DEFAULT_VELOCITY)
    flags: Counter[str] = Counter()
    parts = [item for item in score if _local(item.tag) == "Part"]
    if not parts or not staves:
        raise ValueError("MuseScore score needs at least one Part and one Staff with measures")
    for index, part in enumerate(parts):
        instrument = _child(part, "Instrument")
        name = _text(part, "trackName") or _text(instrument, "trackName") or _text(instrument, "longName")
        chromatic = _integer(_text(instrument, "transposeChromatic"), f"part {index}", "transposeChromatic") or 0
        transposing = chromatic % 12 != 0
        entry = ScorePart(name, f"Part[{index}]", [], _text(instrument, "trackName"))
        for staff_id in _staff_ids(part):
            if staff_id in staves:
                entry.notes.extend(_StaffReader(staff_id, entry, transposing, data, flags).read(staves[staff_id]))
        entry.placeholder_velocities = len(entry.notes)
        data.parts.append(entry)
    data.tempo_locator = "Tempo/tempo"
    if program:
        ProjectFacts(evidence, source_id, method, detail).application(f"MuseScore {program}", "museScore/programVersion")
    elif file_version:
        ProjectFacts(evidence, source_id, method, detail).application(
            f"MuseScore (file format {file_version})", "museScore/@version")
    emit_score(evidence, source_id, method, "musescore", detail, data)
    _flags(evidence, source_id, flags)


def _flags(evidence: Evidence, source_id: str, flags: Counter[str]) -> None:
    notices = {
        "repeat": "Repeats and jumps are not unrolled; each stored measure is read once.",
        "repeat_measure": "Measure-repeat signs were advanced by one measure but their notes are not copied.",
        "multimeasure": "Multi-measure rests are display-only; the underlying measures were read.",
        "grace": "Grace notes were skipped.",
        "unsounded": "Notes marked as not played were skipped.",
    }
    for flag in sorted(notices):
        if flags[flag]:
            evidence.diagnostics.append(Diagnostic(f"musescore_{flag}", "info", notices[flag], source_id))
