"""Ardour sessions (.ardour XML), written by Ardour 4 through 8.

Three TempoMap layouts exist. Ardour 4 and older use flat `Tempo` and `Meter` children positioned by a `start` BBT
string. Ardour 5 and 6 (session versions up to 6000) keep the flat children and add whole-note `pulse` positions.
Ardour 7 and later store `Tempos` and `Meters` lists positioned in `quarters` ("beats:ticks"), and write locations
as sample counts (older) or `a<superclock>` and `b<ticks>` (newer). Referenced audio and plugins are recorded as
names only and never opened.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import Element

from ..model import Detail, Diagnostic, Evidence
from ..project import TIME_DOMAINS, PositionUnit, ProjectFacts, TrackKind
from .common import _finite_number, _parse_xml, _read_bounded

MAX_ARDOUR_BYTES = 32 * 1024 * 1024
MAX_ROUTES = 10_000
TICKS_PER_BEAT = 1920
PLUGIN_TYPES = frozenset({"lv2", "ladspa", "luaproc", "vst", "lxvst", "windows-vst", "mac-vst", "vst3", "audiounit"})
SAMPLE_POSITION = re.compile(r"-?\d+")
TIMED_POSITION = re.compile(r"([ab])(-?\d+)")
TRUE_TEXT = frozenset({"1", "yes", "true"})


def _number(node: Element, name: str, label: str) -> float:
    try:
        return _finite_number(float(node.attrib[name]), f"{label} {name}")
    except KeyError as error:
        raise ValueError(f"{label} has no {name} attribute") from error
    except ValueError as error:
        raise ValueError(f"{label} {name} must be a finite number") from error


def _flag(node: Element, name: str) -> bool:
    return node.attrib.get(name, "1").strip().lower() in TRUE_TEXT


def _quarters(text: str, label: str) -> float:
    whole, _, ticks = text.partition(":")
    try:
        return _finite_number(float(whole) + (float(ticks) / TICKS_PER_BEAT if ticks else 0.0), label)
    except ValueError as error:
        raise ValueError(f"{label} must be a beat count such as 4:0") from error


def _whole_number(node: Element, name: str, label: str) -> int:
    value = _number(node, name, label)
    if not value.is_integer() or value < 1:
        raise ValueError(f"{label} {name} must be a positive whole number")
    return int(value)


def _beat(node: Element, label: str) -> float | None:
    if "quarters" in node.attrib:
        return _quarters(node.attrib["quarters"], f"{label} quarters")
    if "pulse" in node.attrib:
        return _number(node, "pulse", label) * 4
    return 0.0 if node.attrib.get("start") == "1|1|0" else None


def _tempo_points(tempo_map: Element) -> list[tuple[float | None, float]]:
    nested = tempo_map.find("Tempos")
    nodes = nested.findall("Tempo") if nested is not None else tempo_map.findall("Tempo")
    points: list[tuple[float | None, float]] = []
    for number, node in enumerate(nodes):
        label = f"TempoMap/Tempo[{number}]"
        if not _flag(node, "active"):
            continue
        # Ardour 7+ stores note types per minute as npm; Ardour 5 and 6 store beats-per-minute. Both count `note-type` notes.
        per_minute = _number(node, "npm" if "npm" in node.attrib else "beats-per-minute", label)
        note_type = _number(node, "note-type", label)
        if per_minute <= 0 or note_type <= 0:
            raise ValueError(f"{label} must have a positive tempo and note type")
        points.append((_beat(node, label), per_minute * 4 / note_type))
    return points


def _meter_points(tempo_map: Element) -> list[tuple[float | None, int, int]]:
    nested = tempo_map.find("Meters")
    nodes = nested.findall("Meter") if nested is not None else tempo_map.findall("Meter")
    points = []
    for number, node in enumerate(nodes):
        label = f"TempoMap/Meter[{number}]"
        denominator = "note-value" if "note-value" in node.attrib else "note-type"
        # Ardour 4 and older name the numerator beats-per-bar.
        numerator = "divisions-per-bar" if "divisions-per-bar" in node.attrib else "beats-per-bar"
        points.append((_beat(node, label), _whole_number(node, numerator, label), _whole_number(node, denominator, label)))
    return points


def _position(text: str, sample_rate: int | None, superclock: float | None) -> tuple[float, PositionUnit] | None:
    if SAMPLE_POSITION.fullmatch(text):
        return (int(text) / sample_rate, "seconds") if sample_rate else None
    match = TIMED_POSITION.fullmatch(text)
    if match is None:
        raise ValueError(f"unreadable location position {text!r}")
    if match.group(1) == "b":
        return int(match.group(2)) / TICKS_PER_BEAT, "beats"
    return (int(match.group(2)) / superclock, "seconds") if superclock else None


def _kind(route: Element, flags: set[str]) -> TrackKind:
    if "MasterOut" in flags:
        return "master"
    if "AudioTrack" in flags:
        return "audio"
    if "MidiTrack" in flags:
        return "midi"
    if flags & {"AudioBus", "MidiBus", "FoldbackBus"}:
        return "group"
    if flags & {"MonitorOut", "Auditioner"}:
        return "unknown"
    # Sessions without PresentationInfo: only a stored playlist proves a track.
    if "audio-playlist" in route.attrib or route.find("Diskstream") is not None:
        return "audio"
    return "midi" if "midi-playlist" in route.attrib else "unknown"


def _locations(session: Element, source_id: str, evidence: Evidence, facts: ProjectFacts, sample_rate: int | None,
               superclock: float | None) -> None:
    skipped = 0
    for number, node in enumerate(session.findall("Locations/Location")):
        flags = set((node.attrib.get("flags") or "").replace(" ", "").split(","))
        if "IsHidden" in flags or not (flags & {"IsMark", "IsRangeMarker", "IsSessionRange", "IsAutoLoop"}):
            continue
        locator = f"Locations/Location[{number}]"
        start = _position(node.attrib.get("start", ""), sample_rate, superclock)
        end = _position(node.attrib.get("end", ""), sample_rate, superclock)
        name = node.attrib.get("name")
        if "IsMark" in flags:
            if start is None:
                skipped += 1
            else:
                facts.marker(name, start[0], start[1], locator)
            continue
        if start is None or end is None or start[1] != end[1]:
            skipped += 1
            continue
        kind = ("session_range" if "IsSessionRange" in flags else "loop_range" if "IsAutoLoop" in flags else "range")
        unit = start[1]
        evidence.add(source_id, f"ardour.{kind}", {"name": name, "start": start[0], "end": end[0], "unit": unit},
                     "range", "project", TIME_DOMAINS[unit], locator, facts.method, "extracted", "high",
                     "Stored location range.")
    if skipped:
        evidence.diagnostics.append(Diagnostic("locations_skipped", "warning",
            f"{skipped} locations were skipped because their time unit could not be converted "
            "(no sample rate or superclock rate in the session, or mixed units).", source_id))


def ardour(path: Path, source_id: str, evidence: Evidence, detail: Detail) -> None:
    session = _parse_xml(_read_bounded(path, MAX_ARDOUR_BYTES))
    if session.tag != "Session":
        raise ValueError("Ardour session root must be Session")
    method = "ardour-xml"
    facts = ProjectFacts(evidence, source_id, method, detail)

    def extra(property: str, value: Any, unit: str, locator: str, reason: str, scope: str = "project",
              time_domain: str = "static") -> None:
        evidence.add(source_id, property, value, unit, scope, time_domain, locator, method, "extracted", "high", reason)

    version = session.attrib.get("version", "")
    if not version.isdigit():
        raise ValueError("Ardour Session has no numeric version attribute")
    extra("ardour.session_version", int(version), "version", "Session/@version", "Stored Ardour session file version.")
    program = session.find("ProgramVersion")
    stamp = (program.attrib.get("modified-with") or program.attrib.get("created-with")) if program is not None else None
    if stamp and stamp.strip():
        facts.application(stamp.strip(), "Session/ProgramVersion/@modified-with")
    sample_rate: int | None = None
    if "sample-rate" in session.attrib:
        sample_rate = _whole_number(session, "sample-rate", "Session")
        facts.sample_rate(sample_rate, "Session/@sample-rate")
    tempo_map = session.find("TempoMap")
    superclock: float | None = None
    if tempo_map is not None:
        rate_name = next((name for name in ("superclocks-per-second", "superclock-ticks-per-second")
                          if name in tempo_map.attrib), None)
        superclock = _number(tempo_map, rate_name, "TempoMap") if rate_name else None
        tempos = _tempo_points(tempo_map)
        if len(tempos) > 1 and any(beat is None for beat, _ in tempos):
            evidence.diagnostics.append(Diagnostic("tempo_positions_unreadable", "warning",
                "TempoMap has several tempos without a readable position (an Ardour 4 or older session), so no tempo is claimed.", source_id))
        elif tempos:
            ordered = sorted(((beat or 0.0), bpm) for beat, bpm in tempos)
            facts.tempo_map([{"beat": beat, "bpm": bpm} for beat, bpm in ordered], "TempoMap/Tempo")
            if len(ordered) == 1:
                facts.tempo(ordered[0][1], "TempoMap/Tempo")
        meters = _meter_points(tempo_map)
        if len(meters) == 1:
            facts.meter(meters[0][1], meters[0][2], "TempoMap/Meter")
        elif meters and not any(beat is None for beat, _, _ in meters):
            extra("ardour.meter_map", [{"beat": beat, "meter": f"{num}/{den}"} for beat, num, den in sorted(meters)],
                  "meter_map", "TempoMap/Meter", "Stored meter changes in beat order; no single static meter is claimed.",
                  time_domain="project_beats")
    _locations(session, source_id, evidence, facts, sample_rate, superclock)
    routes = session.findall("Routes/Route")
    if len(routes) > MAX_ROUTES:
        raise ValueError(f"session has more than {MAX_ROUTES} routes")
    playlists = {node.attrib.get("id"): node for node in session.findall("Playlists/Playlist")}
    facts.track_count(len(routes), "Routes/Route")
    for index, route in enumerate(routes):
        locator = f"Routes/Route[{index}]"
        info = route.find("PresentationInfo")
        flags = set((info.attrib.get("flags", "") if info is not None else "").replace(" ", "").split(",")) - {""}
        facts.track(index, route.attrib.get("name"), _kind(route, flags), locator)
        extra("ardour.route", {"defaultType": route.attrib.get("default-type"), "flags": sorted(flags),
                               "active": _flag(route, "active")}, "route", f"{locator}/@default-type,PresentationInfo/@flags",
              "Stored route type and presentation flags.", f"project_track:{index}")
        for processor in route.findall("Processor"):
            if processor.attrib.get("type") in PLUGIN_TYPES and processor.attrib.get("name"):
                facts.plugin(processor.attrib["name"], index, f"{locator}/Processor/@name")
        playlist = playlists.get(route.attrib.get("audio-playlist") or route.attrib.get("midi-playlist"))
        if playlist is not None:
            facts.clip_count(len(playlist.findall("Region")), "Playlists/Playlist[@id]/Region", index)
    facts.media_references([node.attrib["name"] for node in session.findall("Sources/Source") if node.attrib.get("name")],
                           "Sources/Source/@name")
    evidence.diagnostics.append(Diagnostic("ardour_notes_not_read", "info",
        "Ardour keeps MIDI notes in separate .mid source files, which are not opened; export MIDI regions for note content. "
        "Tempo ramps are reduced to their starting tempo.", source_id))
