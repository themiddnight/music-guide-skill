from __future__ import annotations

import io
import re
import sys
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, NamedTuple

from ..model import Detail, Diagnostic, Evidence
from ..notes import Note
from ..project import ProjectFacts, TrackKind
from .common import MAX_ZIP_BYTES, _finite_number, _read_bounded
from .graded import GradedFacts
from .listing import capped

METHOD = "reaper-rpp-text"
FX_TAGS = frozenset({"VST", "AU", "JS", "CLAP", "DX", "LV2"})
MAX_LINES = 1_500_000
MAX_DEPTH = 64
MAX_NOTES = 1_000_000
DATA_LINE_LENGTH = 64
INTERN_LENGTH = 12
EPSILON = 1e-9
TOKEN = re.compile(r"\"([^\"]*)\"|'([^']*)'|`([^`]*)`|(\S+)")
KEYWORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
QUOTES = "\"'`"
ONE_DATA_BYTE = frozenset({0xC0, 0xD0})
SQUARE_SHAPE = "1"


class Prop(NamedTuple):
    key: str
    params: list[str]
    line: int


@dataclass
class Block:
    tag: str
    params: list[str]
    line: int
    entries: list[Block | Prop] = field(default_factory=list)

    def props(self, key: str) -> list[Prop]:
        return [entry for entry in self.entries if isinstance(entry, Prop) and entry.key == key]

    def prop(self, key: str) -> Prop | None:
        return next(iter(self.props(key)), None)

    def blocks(self, tag: str) -> list[Block]:
        return [entry for entry in self.entries if isinstance(entry, Block) and entry.tag == tag]

    def walk(self, skip: frozenset[str] = frozenset()) -> Iterator[Block]:
        stack = [self]
        while stack:
            block = stack.pop()
            yield block
            stack.extend(reversed([entry for entry in block.entries
                                   if isinstance(entry, Block) and entry.tag not in skip]))


def _tokens(text: str, number: int) -> list[str]:
    tokens = []
    for match in TOKEN.finditer(text):
        bare = match.group(4)
        if bare is not None and bare[0] in QUOTES:
            raise ValueError(f"unterminated quoted string at line {number}")
        token = next(group for group in match.groups() if group is not None)
        tokens.append(sys.intern(token) if len(token) <= INTERN_LENGTH else token)
    return tokens


def _is_data(line: str) -> bool:
    return line[0] == "|" or (len(line) > DATA_LINE_LENGTH and not any(char.isspace() for char in line))


def _parse_blocks(text: str) -> Block:
    root: Block | None = None
    stack: list[Block] = []
    stored = 0
    for number, raw in enumerate(io.StringIO(text, newline="\n"), 1):
        line = raw.strip()
        if not line or _is_data(line):
            continue
        stored += 1
        if stored > MAX_LINES:
            raise ValueError(f"REAPER project exceeds {MAX_LINES} lines")
        if line == ">":
            if not stack:
                raise ValueError(f"unbalanced '>' at line {number}")
            stack.pop()
        elif line[0] == "<":
            tokens = _tokens(line[1:], number)
            if not tokens:
                raise ValueError(f"block without a tag at line {number}")
            block = Block(tokens[0], tokens[1:], number)
            if stack:
                stack[-1].entries.append(block)
            elif root is None:
                root = block
            else:
                raise ValueError(f"content after the project block at line {number}")
            if len(stack) >= MAX_DEPTH:
                raise ValueError(f"blocks nest deeper than {MAX_DEPTH} at line {number}")
            stack.append(block)
        elif not stack:
            raise ValueError(f"content outside the project block at line {number}")
        else:
            tokens = _tokens(line, number)
            if tokens and KEYWORD.fullmatch(tokens[0]):
                stack[-1].entries.append(Prop(tokens[0], tokens[1:], number))
    if root is None:
        raise ValueError("REAPER project has no block")
    if stack:
        raise ValueError(f"REAPER project is truncated: {len(stack)} blocks are not closed (opened at line {stack[-1].line})")
    return root


def _float(prop: Prop, position: int, label: str) -> float:
    try:
        return _finite_number(float(prop.params[position]), label)
    except (IndexError, ValueError) as error:
        raise ValueError(f"{label} at line {prop.line} is not a finite number") from error


def _int(prop: Prop, position: int, label: str, base: int = 10) -> int:
    try:
        return int(prop.params[position], base)
    except (IndexError, ValueError) as error:
        raise ValueError(f"{label} at line {prop.line} is not an integer") from error


class Timeline:
    """Converts project seconds to quarter-note beats, treating each tempo envelope point as a step."""

    def __init__(self, bpm: float | None, points: list[tuple[float, float]]) -> None:
        self.points = sorted(points, key=lambda point: point[0])
        self.bpm = bpm
        self.starts: list[float] = []
        self.beats_at: list[float] = []
        beat = 0.0
        for index, (second, tempo) in enumerate(self.points):
            if index == 0:
                beat = second * tempo / 60
            else:
                previous_second, previous_tempo = self.points[index - 1]
                beat += (second - previous_second) * previous_tempo / 60
            self.starts.append(second)
            self.beats_at.append(beat)

    def beats(self, seconds: float) -> float:
        if not self.points:
            if self.bpm is None:
                raise ValueError("REAPER project has no TEMPO line or tempo envelope to place items in beats")
            return seconds * self.bpm / 60
        index = max(bisect_right(self.starts, seconds) - 1, 0)
        return self.beats_at[index] + (seconds - self.starts[index]) * self.points[index][1] / 60


@dataclass
class _NoteStats:
    notes: list[Note] = field(default_factory=list)
    muted_items: int = 0
    outside: int = 0
    offsets: int = 0
    tempo_ignoring: int = 0
    unmatched: int = 0
    skipped_sources: int = 0
    invalid_events: int = 0


def _active_source(item: Block) -> Block | None:
    takes: list[list] = [[False, None]]
    for entry in item.entries:
        if isinstance(entry, Prop) and entry.key == "TAKE":
            takes.append(["SEL" in entry.params, None])
        elif isinstance(entry, Block) and entry.tag == "SOURCE" and takes[-1][1] is None:
            takes[-1][1] = entry
    return next((take[1] for take in takes if take[0]), takes[0][1])


def _midi_source(source: Block | None) -> Block | None:
    if source is None:
        return None
    return next((block for block in source.walk() if block.tag == "SOURCE" and block.params[:1] == ["MIDI"]), None)


def _first_param(block: Block, key: str) -> str | None:
    prop = block.prop(key)
    return prop.params[0] if prop is not None and prop.params else None


def _read_item_notes(item: Block, timeline: Timeline, stats: _NoteStats) -> None:
    if _first_param(item, "MUTE") == "1":
        stats.muted_items += 1
        return
    source = _midi_source(_active_source(item))
    hasdata = None if source is None else source.prop("HASDATA")
    if source is None or hasdata is None or hasdata.params[:1] != ["1"]:
        return
    ppq = _int(hasdata, 1, "HASDATA resolution")
    if ppq <= 0 or hasdata.params[2:3] != ["QN"]:
        stats.skipped_sources += 1
        return
    position_prop, length_prop = item.prop("POSITION"), item.prop("LENGTH")
    if position_prop is None or length_prop is None:
        raise ValueError(f"ITEM at line {item.line} needs POSITION and LENGTH to place MIDI notes")
    position, length = _float(position_prop, 0, "POSITION"), _float(length_prop, 0, "LENGTH")
    start_beat, end_beat = timeline.beats(position), timeline.beats(position + length)
    stats.offsets += _first_param(item, "SOFFS") not in (None, "0")
    ignore = source.prop("IGNTEMPO")
    stats.tempo_ignoring += ignore is not None and ignore.params[:1] == ["1"]
    tick = 0
    pending: dict[tuple[int, int], deque[tuple[int, int]]] = {}
    for entry in source.entries:
        if isinstance(entry, Prop) and entry.key in ("E", "e"):
            tick += _int(entry, 0, "MIDI event delta")
            message = _channel_message(entry)
            if message is None:
                stats.invalid_events += 1
                continue
            kind, channel, first, second = message
            if kind == 0x90 and second > 0:
                pending.setdefault((channel, first), deque()).append((tick, second))
            elif kind in (0x80, 0x90) and pending.get((channel, first)):
                begin, velocity = pending[(channel, first)].popleft()
                _add_note(stats, Note(first, velocity, start_beat + begin / ppq, start_beat + tick / ppq, channel),
                          end_beat)
        elif _is_sysex(entry):
            tick += int(entry.params[0]) if entry.params and entry.params[0].lstrip("-").isdigit() else 0
    stats.unmatched += sum(len(starts) for starts in pending.values())


def _channel_message(entry: Prop) -> tuple[int, int, int, int] | None:
    """Status kind, channel and the first two data bytes of a channel message; None when the event is not valid.

    Program change (0xC0) and channel pressure (0xD0) carry one data byte, so their second value is 0."""
    try:
        status, *data = (int(value, 16) for value in entry.params[1:4])
    except ValueError:
        return None
    kind = status & 0xF0
    wanted = 1 if kind in ONE_DATA_BYTE else 2
    if not 0x80 <= status <= 0xEF or len(data) < wanted or any(not 0 <= value <= 127 for value in data[:wanted]):
        return None
    return kind, status & 0x0F, data[0], data[1] if wanted == 2 else 0


def _is_sysex(entry: Block | Prop) -> bool:
    return (entry.key if isinstance(entry, Prop) else entry.tag) in ("X", "x")


def _add_note(stats: _NoteStats, note: Note, item_end: float) -> None:
    if note.start_beat >= item_end - EPSILON:
        stats.outside += 1
        return
    stats.notes.append(Note(note.pitch, note.velocity, note.start_beat, min(note.end_beat, item_end), note.channel))
    if len(stats.notes) > MAX_NOTES:
        raise ValueError(f"REAPER project exceeds {MAX_NOTES} notes")


def _notes_reason(stats: _NoteStats, mapped: bool) -> str:
    tempo = ("by treating each TEMPOENVEX point as a tempo step, so ramps are not interpolated" if mapped
             else "with the project TEMPO")
    return (f"Approximate timing. Item POSITION seconds are converted to beats {tempo}; each note adds its MIDI ticks "
            f"divided by the source resolution to the item start. Source start offsets ({stats.offsets} items), "
            f"item loops and tempo-ignoring sources ({stats.tempo_ignoring}) are not applied, {stats.outside} notes "
            f"past the item end are omitted, {stats.muted_items} muted items are skipped and {stats.skipped_sources} "
            "sources not stored in quarter notes are skipped.")


def _effect_name(block: Block) -> str:
    name = block.params[0].strip() if block.params else ""
    return f"JS: {name}" if name and block.tag == "JS" and not name.startswith("JS") else name


def _effects(parent: Block, skip: frozenset[str] = frozenset()) -> list[Block]:
    return [block for block in parent.walk(skip) if block.tag in FX_TAGS and _effect_name(block)]


def _markers(facts: ProjectFacts, evidence: Evidence, source_id: str, project: Block) -> None:
    regions: dict[int, list[tuple[float, str, int]]] = {}
    markers = []
    for prop in project.props("MARKER"):
        if len(prop.params) < 3:
            raise ValueError(f"MARKER at line {prop.line} is incomplete")
        index, position = _int(prop, 0, "MARKER index"), _float(prop, 1, "MARKER position")
        if prop.params[3:4] == ["1"]:
            regions.setdefault(index, []).append((position, prop.params[2], prop.line))
        else:
            markers.append((prop.params[2], position, prop.line))
    for index, ends in sorted(regions.items()):
        if len(ends) == 2:
            (first, name, line), (second, other, _) = ends
            evidence.add(source_id, "reaper.region", {"index": index, "name": name or other or None,
                                                      "start": min(first, second), "end": max(first, second),
                                                      "unit": "seconds"}, "region", "project", "project_seconds",
                         f"MARKER {index} (line {line})", METHOD, "extracted", "high",
                         "Stored region from a pair of MARKER lines sharing one index.")
        else:
            markers.extend((name or None, position, line) for position, name, line in ends)
            evidence.diagnostics.append(Diagnostic("reaper_incomplete_region", "warning",
                f"Region {index} has {len(ends)} MARKER lines instead of 2; reported as plain markers.", source_id))
    for name, position, line in capped(markers, "markers", source_id, evidence):
        facts.marker(name or None, position, "seconds", f"MARKER (line {line})")


def _tempo(facts: ProjectFacts, evidence: Evidence, source_id: str, project: Block) -> Timeline:
    tempo = project.prop("TEMPO")
    bpm = None if tempo is None else _finite_number(_float(tempo, 0, "TEMPO"), "TEMPO", positive=True)
    if tempo is not None:
        if len(tempo.params) >= 3:
            facts.meter(_int(tempo, 1, "TEMPO numerator"), _int(tempo, 2, "TEMPO denominator"),
                        f"TEMPO (line {tempo.line})",
                        "Stored project meter; meter changes on the tempo envelope are not decoded.")
    points: list[tuple[float, float]] = []
    ramps = 0
    envelope = next(iter(project.blocks("TEMPOENVEX")), None)
    if envelope is not None and _first_param(envelope, "ACT") == "0":
        evidence.diagnostics.append(Diagnostic("reaper_tempo_envelope_inactive", "warning",
            "The tempo envelope is stored as inactive (ACT 0) and was ignored.", source_id))
    elif envelope is not None:
        pt = envelope.props("PT")
        points = [(_float(prop, 0, "tempo point position"), _float(prop, 1, "tempo point bpm")) for prop in pt]
        ramps = sum(prop.params[2:3] != [SQUARE_SHAPE] for prop in pt)
    values = {value for _, value in points}
    locator = f"TEMPOENVEX (line {envelope.line})" if envelope is not None else ""
    timeline = Timeline(bpm, points)
    if len(values) > 1:
        if ramps:
            evidence.diagnostics.append(Diagnostic("reaper_tempo_ramps", "warning",
                f"{ramps} tempo envelope point(s) are not square steps (or have no stored shape); REAPER "
                "interpolates tempo to the next point, which this reader does not model.", source_id))
            map_facts: ProjectFacts = GradedFacts(evidence, source_id, METHOD, facts.detail, grade="low",
                caveat="Some points are ramps or have no stored shape, so tempo between points and the beat "
                       "positions that depend on it are approximated as steps.")
        else:
            map_facts = facts
        map_facts.tempo_map([{"beat": timeline.beats(second), "bpm": value} for second, value in timeline.points], locator)
        if bpm is not None:
            evidence.add(source_id, "reaper.base_tempo", bpm, "BPM", "project", "static", f"TEMPO (line {tempo.line})",
                         METHOD, "extracted", "medium",
                         "TEMPO line stored beside a tempo envelope; the envelope decides the played tempo.")
    elif values:
        facts.tempo(values.pop(), locator, "Tempo envelope holds one value for the whole project.")
    elif tempo is not None:
        facts.tempo(bpm, f"TEMPO (line {tempo.line})", "Stored project tempo; no tempo envelope points were found.")
    return timeline


def _track_facts(facts: ProjectFacts, track: Block, index: int, timeline: Timeline,
                 source_id: str, evidence: Evidence) -> list[str]:
    isbus = track.prop("ISBUS")
    kind: TrackKind = "folder" if isbus is not None and isbus.params[:1] == ["1"] else "unknown"
    facts.track(index, _first_param(track, "NAME") or None, kind, f"TRACK (line {track.line})")
    items = track.blocks("ITEM")
    if items:
        facts.clip_count(len(items), f"TRACK (line {track.line}) ITEM", index)
    stats = _NoteStats()
    for item in items:
        _read_item_notes(item, timeline, stats)
    if stats.unmatched:
        evidence.diagnostics.append(Diagnostic("unmatched_note_on", "warning",
            f"{stats.unmatched} MIDI note(s) in REAPER track {index} have no note-off.", source_id))
    if stats.invalid_events:
        evidence.diagnostics.append(Diagnostic("reaper_invalid_midi_event", "warning",
            f"{stats.invalid_events} MIDI event(s) in REAPER track {index} have an invalid status or data bytes and "
            f"were skipped.", source_id))
    facts.track_notes(index, stats.notes, f"TRACK (line {track.line}) ITEM SOURCE MIDI",
                      _notes_reason(stats, bool(timeline.points)), "medium")
    return [prop.params[0] for item in items for source in item.walk() if source.tag == "SOURCE"
            for prop in source.props("FILE") if prop.params]


def _parse(path: Path, source_id: str, evidence: Evidence, detail: Detail) -> None:
    data = _read_bounded(path, MAX_ZIP_BYTES)
    if b"\x00" in data:
        raise ValueError("REAPER project contains binary data; it is not a text .rpp file")
    project = _parse_blocks(data.decode("utf-8", errors="replace").removeprefix("﻿"))
    if project.tag != "REAPER_PROJECT":
        raise ValueError("REAPER project must start with a REAPER_PROJECT block")
    facts = ProjectFacts(evidence, source_id, METHOD, detail)
    header = {key: value for key, value in zip(("formatVersion", "build", "savedAt"), project.params)}
    facts.application(f"REAPER {header['build']}" if "build" in header else "REAPER", "REAPER_PROJECT header")
    evidence.add(source_id, "reaper.header", header, "header", "project", "static", "REAPER_PROJECT header", METHOD,
                 "extracted", "high", "Stored project header values.")
    rate = project.prop("SAMPLERATE")
    if rate is not None and _float(rate, 0, "SAMPLERATE") > 0:
        facts.sample_rate(int(_float(rate, 0, "SAMPLERATE")), f"SAMPLERATE (line {rate.line})")
    timeline = _tempo(facts, evidence, source_id, project)
    _markers(facts, evidence, source_id, project)
    tracks = project.blocks("TRACK")
    facts.track_count(len(tracks), "TRACK")
    media: list[str] = []
    plugins = [(_effect_name(block), None, f"{block.tag} (line {block.line})")
               for block in _effects(project, frozenset({"TRACK"}))]
    for index, track in enumerate(tracks):
        media += _track_facts(facts, track, index, timeline, source_id, evidence)
        plugins += [(_effect_name(block), index, f"TRACK (line {track.line}) {block.tag} (line {block.line})")
                    for block in _effects(track)]
    for name, index, locator in capped(plugins, "plugins", source_id, evidence):
        facts.plugin(name, index, locator)
    facts.clip_count(sum(len(track.blocks("ITEM")) for track in tracks), "TRACK ITEM")
    facts.media_references(capped(media, "media references", source_id, evidence), "ITEM SOURCE FILE")


def reaper(path: Path, source_id: str, evidence: Evidence, detail: Detail) -> None:
    try:
        _parse(path, source_id, evidence, detail)
    except (AttributeError, IndexError, ArithmeticError, RecursionError) as error:
        raise ValueError(f"unexpected REAPER project structure: {error}") from error
