from __future__ import annotations

import math
import zipfile
from bisect import bisect_right
from dataclasses import dataclass, replace
from pathlib import Path
from xml.etree.ElementTree import Element

from ..model import Confidence, Detail, Diagnostic, Evidence
from ..notes import Note
from ..project import ProjectFacts, TrackKind
from .common import MAX_XML_BYTES, _child, _finite_number, _local, _parse_xml, _read_zip_member
from .textnum import MAX_BEAT, beat_float

CHANNEL_ROLES: dict[str, TrackKind] = {"master": "master", "effect": "return", "submix": "group"}
CONTENT_TYPES: list[tuple[str, TrackKind]] = [("tracks", "folder"), ("notes", "instrument"), ("audio", "audio")]
METADATA_FIELDS = ("Title", "Artist", "Album", "OriginalArtist", "Composer", "Songwriter", "Producer", "Arranger",
                   "Year", "Genre", "Copyright", "Website", "Comment")
MAX_METADATA_TEXT = 500
MAX_DEPTH = 32
EPSILON = 1e-9
MAX_ATTRIBUTE = 1e15
PLACEHOLDER_VELOCITY = 100
# The DAWproject specification says a timeline without a time unit uses beats.
DEFAULT_UNIT = "beats"
CONTAINERS = {"Lanes", "Clips"}


def _track_kind(track: Element) -> TrackKind:
    channel = _child(track, "Channel")
    role = channel.attrib.get("role", "") if channel is not None else ""
    if role in CHANNEL_ROLES:
        return CHANNEL_ROLES[role]
    content = set(track.attrib.get("contentType", "").split())
    return next((kind for token, kind in CONTENT_TYPES if token in content), "unknown")


def _number(element: Element, name: str, *, required: bool = True) -> float | None:
    text = element.attrib.get(name)
    if text is None:
        if required:
            raise ValueError(f"DAWproject {_local(element.tag)} is missing its {name} attribute")
        return None
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"DAWproject {_local(element.tag)} {name} must be a number, not {text!r}") from error
    if not math.isfinite(value) or abs(value) > MAX_ATTRIBUTE:
        raise ValueError(f"DAWproject {_local(element.tag)} {name} must be finite and below {MAX_ATTRIBUTE:g}")
    return value


def _whole(element: Element, name: str) -> int:
    value = _number(element, name)
    if value is None or value != int(value):
        raise ValueError(f"DAWproject {_local(element.tag)} {name} must be a whole number")
    return int(value)


def _unit(element: Element, attribute: str, inherited: str) -> str:
    unit = element.attrib.get(attribute, inherited)
    if unit not in {"beats", "seconds"}:
        raise ValueError(f"DAWproject {attribute} must be beats or seconds, not {unit!r}")
    return unit


class _Clock:
    """Maps between quarter-note beats and seconds with piecewise-linear knots (beat, seconds)."""

    def __init__(self, knots: list[tuple[float, float]], first_bpm: float, last_bpm: float, ramps: bool = False) -> None:
        self.knots = knots
        self.ramps = ramps
        self.beats = [beat for beat, _ in knots]
        self.seconds = [second for _, second in knots]
        self.first_bpm = first_bpm
        self.last_bpm = last_bpm

    @classmethod
    def constant(cls, bpm: float) -> _Clock:
        return cls([(0.0, 0.0)], bpm, bpm)

    @classmethod
    def automated(cls, points: list[tuple[float, float, bool]], unit: str) -> _Clock:
        """Points are (time, bpm, linear). Ramps use the mean tempo of the segment, holds the earlier tempo."""
        first_time, first_bpm, _ = points[0]
        beat, second = (first_time, first_time * 60 / first_bpm) if unit == "beats" else (
            first_time * first_bpm / 60, first_time)
        knots = [(beat, second)]
        for (time, bpm, linear), (next_time, next_bpm, _) in zip(points, points[1:]):
            tempo = (bpm + next_bpm) / 2 if linear else bpm
            if unit == "beats":
                beat, second = next_time, second + (next_time - time) * 60 / tempo
            else:
                beat, second = beat + (next_time - time) * tempo / 60, next_time
            knots.append((beat, second))
        if not all(math.isfinite(value) and abs(value) <= MAX_BEAT * 1e3 for knot in knots for value in knot):
            raise ValueError("DAWproject tempo automation places points outside the supported range")
        ramps = any(linear for (_, _, linear), _ in zip(points, points[1:]))
        return cls(knots, first_bpm, points[-1][1], ramps)

    def seconds_at(self, beat: float) -> float:
        return self._map(beat, self.beats, self.seconds, 60 / self.first_bpm, 60 / self.last_bpm)

    def beats_at(self, second: float) -> float:
        return self._map(second, self.seconds, self.beats, self.first_bpm / 60, self.last_bpm / 60)

    @staticmethod
    def _map(value: float, source: list[float], target: list[float], head_slope: float, tail_slope: float) -> float:
        if value <= source[0]:
            return target[0] + (value - source[0]) * head_slope
        if value >= source[-1]:
            return target[-1] + (value - source[-1]) * tail_slope
        index = bisect_right(source, value) - 1
        span = source[index + 1] - source[index]
        if span <= 0:
            return target[index]
        return target[index] + (value - source[index]) / span * (target[index + 1] - target[index])


class _NoClock(Exception):
    pass


@dataclass(frozen=True)
class _Scope:
    """Where child times sit: `unit` is their time unit and content time `play` lands on beat `start`."""

    unit: str
    start: float = 0.0
    play: float = 0.0
    track: int | None = None
    window: tuple[float, float | None] | None = None
    depth: int = 0
    in_clip: bool = False


class _Arrangement:
    def __init__(self, facts: ProjectFacts, evidence: Evidence, source_id: str, track_ids: dict[str, int],
                 clock: _Clock | None) -> None:
        self.facts = facts
        self.evidence = evidence
        self.source_id = source_id
        self.track_ids = track_ids
        self.clock = clock
        self.clips: dict[int, int] = {}
        self.ramped: set[int] = set()
        self.placeholders: dict[int, int] = {}
        self.notes: dict[int, list[Note]] = {}
        self.flags: dict[str, int] = {}

    def flag(self, name: str, amount: int = 1) -> None:
        self.flags[name] = self.flags.get(name, 0) + amount

    def place(self, scope: _Scope, value: float, unit: str | None = None) -> float:
        """Beat where `value` (in the scope's time unit) lands."""
        unit = unit or scope.unit
        relative = value - scope.play
        if unit == "beats":
            return beat_float(scope.start + relative)
        if self.clock is None:
            raise _NoClock
        if self.clock.ramps and scope.track is not None:
            self.ramped.add(scope.track)
        return beat_float(self.clock.beats_at(self.clock.seconds_at(scope.start) + relative))

    def walk(self, node: Element, scope: _Scope) -> None:
        if scope.depth > MAX_DEPTH:
            raise ValueError(f"DAWproject arrangement nests deeper than {MAX_DEPTH} levels")
        name = _local(node.tag)
        try:
            if name == "Clip":
                self.clip(node, scope)
            elif name == "Notes":
                self.read_notes(node, replace(scope, unit=_unit(node, "timeUnit", scope.unit)))
            elif name == "Markers":
                self.read_markers(node, scope)
            elif name == "Warps":
                if any(_local(item.tag) == "Notes" for item in node.iter()):
                    self.flag("warped_notes")
            elif name in CONTAINERS:
                track = self.resolve_track(node, scope.track)
                inner = replace(scope, unit=_unit(node, "timeUnit", scope.unit), track=track, depth=scope.depth + 1)
                if track is not None:
                    self.clips.setdefault(track, 0)
                for child in node:
                    self.walk(child, inner)
        except _NoClock:
            self.flag("no_tempo_for_seconds")

    def resolve_track(self, node: Element, inherited: int | None) -> int | None:
        reference = node.attrib.get("track")
        if reference is None:
            return inherited
        if reference not in self.track_ids:
            self.flag("unknown_track")
            return None
        return self.track_ids[reference]

    def clip(self, clip: Element, scope: _Scope) -> None:
        time = _number(clip, "time")
        if not scope.in_clip:
            if scope.track is None:
                self.flag("unassigned_clips")
            else:
                self.clips[scope.track] = self.clips.get(scope.track, 0) + 1
        if clip.attrib.get("reference"):
            self.flag("reference_clips")
        if clip.attrib.get("enable") == "false":
            self.flag("disabled_clips")
        content_unit = _unit(clip, "contentTimeUnit", scope.unit)
        play = _number(clip, "playStart", required=False) or 0.0
        start = self.place(scope, time)
        duration = _number(clip, "duration", required=False)
        end = self.place(scope, time + duration) if duration is not None else None
        inner = _Scope(content_unit, start, play, scope.track, None, scope.depth + 1, True)
        if end is None and (stop := _number(clip, "playStop", required=False)) is not None:
            end = self.place(inner, stop)
        self.check_loop(clip, inner, end)
        windowed = self.windowed(replace(inner, window=scope.window), start, end)
        for child in clip:
            self.walk(child, windowed)

    @staticmethod
    def windowed(inner: _Scope, start: float, end: float | None) -> _Scope:
        outer = inner.window
        low = start if outer is None else max(start, outer[0])
        high = end if outer is None or outer[1] is None else (outer[1] if end is None else min(end, outer[1]))
        return replace(inner, window=(low, high))

    def check_loop(self, clip: Element, inner: _Scope, end: float | None) -> None:
        loop_start, loop_end = _number(clip, "loopStart", required=False), _number(clip, "loopEnd", required=False)
        if loop_start is None or loop_end is None or loop_end <= loop_start or end is None:
            return
        if end > self.place(inner, loop_end) + EPSILON:
            self.flag("looped_clips")

    def read_notes(self, notes: Element, scope: _Scope) -> None:
        for item in notes:
            if _local(item.tag) != "Note":
                continue
            time, duration = _number(item, "time"), _number(item, "duration")
            key, channel = _whole(item, "key"), _whole(item, "channel")
            if not 0 <= key <= 127:
                raise ValueError(f"DAWproject Note key {key} is outside the MIDI range 0-127")
            if duration < 0:
                raise ValueError("DAWproject Note duration must not be negative")
            begin, end = self.place(scope, time), self.place(scope, time + duration)
            if scope.window is not None:
                low, high = scope.window
                if end <= low or (high is not None and begin >= high):
                    self.flag("notes_outside_clip")
                    continue
                begin = max(begin, low)
                end = end if high is None else min(end, high)
            if scope.track is None:
                self.flag("unassigned_notes")
                continue
            velocity = _number(item, "vel", required=False)
            if velocity is None:
                self.flag("placeholder_velocity")
                self.placeholders[scope.track] = self.placeholders.get(scope.track, 0) + 1
                value = PLACEHOLDER_VELOCITY
            else:
                value = max(1, min(127, round(velocity * 127)))
            self.notes.setdefault(scope.track, []).append(Note(key, value, begin, end, channel))

    def read_markers(self, markers: Element, scope: _Scope) -> None:
        if scope.in_clip:
            return
        unit = _unit(markers, "timeUnit", scope.unit)
        for marker in markers:
            if _local(marker.tag) == "Marker":
                name = marker.attrib.get("name")
                self.facts.marker(name.strip()[:MAX_METADATA_TEXT] if name else None, _number(marker, "time"), unit,
                                  "Arrangement//Markers/Marker")


def _has_member(path: Path, member: str) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return any(item.filename == member for item in archive.infolist())
    except zipfile.BadZipFile:
        return False


def _metadata(path: Path, source_id: str, evidence: Evidence) -> None:
    if not _has_member(path, "metadata.xml"):
        return
    try:
        root = _parse_xml(_read_zip_member(path, "metadata.xml", MAX_XML_BYTES))
        if _local(root.tag) != "MetaData":
            raise ValueError("root must be MetaData")
    except ValueError as error:
        evidence.diagnostics.append(Diagnostic("dawproject_metadata_unreadable", "warning",
                                               f"metadata.xml was ignored: {error}", source_id))
        return
    fields = {}
    for name in METADATA_FIELDS:
        item = _child(root, name)
        if item is not None and item.text and item.text.strip():
            fields[name] = item.text.strip()[:MAX_METADATA_TEXT]
    if fields:
        evidence.add(source_id, "dawproject.metadata", fields, "metadata", "project", "static", "metadata.xml",
                     "dawproject-xml", "extracted", "high", "Descriptive metadata stored in metadata.xml.")


def _tempo_automation(arrangement: Element, evidence: Evidence, source_id: str) -> tuple[_Clock | None, list[tuple[float, float]]]:
    automation = _child(arrangement, "TempoAutomation")
    if automation is None:
        return None, []
    points = [item for item in automation if _local(item.tag) == "RealPoint"]
    if not points:
        return None, []
    if automation.attrib.get("unit") != "bpm":
        evidence.diagnostics.append(Diagnostic(
            "dawproject_tempo_automation_unit", "warning",
            "Arrangement/TempoAutomation was ignored because its unit is not an explicit bpm.", source_id))
        return None, []
    unit = _unit(automation, "timeUnit", DEFAULT_UNIT)
    raw = sorted((_number(item, "time"), _finite_number(_number(item, "value"), "tempo automation value", True),
                  item.attrib.get("interpolation") == "linear") for item in points)
    clock = _Clock.automated(raw, unit)
    return clock, [(clock.beats_at(time) if unit == "seconds" else time, bpm) for time, bpm, _ in raw]


def _meter_automation(arrangement: Element, evidence: Evidence, source_id: str, clock: _Clock | None) -> None:
    automation = _child(arrangement, "TimeSignatureAutomation")
    if automation is None:
        return
    unit = _unit(automation, "timeUnit", DEFAULT_UNIT)
    changes = []
    for item in automation:
        if _local(item.tag) != "TimeSignaturePoint":
            continue
        time = _number(item, "time")
        if unit == "seconds":
            if clock is None:
                evidence.diagnostics.append(Diagnostic(
                    "dawproject_meter_automation_unit", "warning",
                    "Time signature automation is in seconds and the project has no tempo to convert it.", source_id))
                return
            time = clock.beats_at(time)
        numerator, denominator = _whole(item, "numerator"), _whole(item, "denominator")
        if numerator <= 0 or denominator <= 0:
            raise ValueError("DAWproject TimeSignaturePoint numerator and denominator must be positive")
        changes.append({"beat": time, "meter": f"{numerator}/{denominator}"})
    if changes:
        evidence.add(source_id, "dawproject.meter_map", sorted(changes, key=lambda change: change["beat"]),
                     "signature", "project", "project_beats", "Arrangement/TimeSignatureAutomation",
                     "dawproject-xml", "extracted", "high", "Stored time signature changes in beat order.")


NOTICES = {
    "looped_clips": ("info", "{n} clip(s) loop beyond one pass; their notes are listed once, not repeated."),
    "notes_outside_clip": ("info", "{n} note(s) outside the played part of their clip were left out."),
    "disabled_clips": ("info", "{n} clip(s) are disabled but still counted and read."),
    "reference_clips": ("warning", "{n} clip(s) reference shared content by id; that content is not followed."),
    "warped_notes": ("warning", "{n} time-warped note timeline(s) were skipped."),
    "no_tempo_for_seconds": ("warning", "{n} clip(s) use seconds but the project has no explicit tempo to convert "
                                        "them; their notes were skipped."),
    "unknown_track": ("warning", "{n} lane(s) name a track that is not in Structure; their content has no track."),
    "unassigned_clips": ("warning", "{n} clip(s) belong to no track and were not counted per track."),
    "unassigned_notes": ("warning", "{n} note(s) belong to no track and were left out."),
    "placeholder_velocity": ("info", "{n} note(s) store no velocity and carry the placeholder "
                                     f"{PLACEHOLDER_VELOCITY}; velocity statistics are unreliable."),
}


def dawproject(path: Path, source_id: str, evidence: Evidence, detail: Detail) -> None:
    root = _parse_xml(_read_zip_member(path, "project.xml", MAX_XML_BYTES))
    if _local(root.tag) != "Project":
        raise ValueError("DAWproject project.xml root must be Project")
    facts = ProjectFacts(evidence, source_id, "dawproject-xml", detail)
    if "version" in root.attrib:
        evidence.add(source_id, "dawproject.version", root.attrib["version"], "version", "project",
                     "static", "Project/@version", "dawproject-xml", "extracted", "high",
                     "Stored DAWproject version.")
    application = _child(root, "Application")
    if application is not None and application.attrib.get("name", "").strip():
        name = f"{application.attrib['name']} {application.attrib.get('version', '')}".strip()
        facts.application(name, "Application/@name")
    _metadata(path, source_id, evidence)
    static_bpm = None
    transport = _child(root, "Transport")
    if transport is not None:
        tempo = _child(transport, "Tempo")
        if tempo is not None and "value" in tempo.attrib:
            if tempo.attrib.get("unit") != "bpm":
                raise ValueError("DAWproject Transport/Tempo unit must be bpm")
            static_bpm = _finite_number(_number(tempo, "value"), "Tempo value", positive=True)
            facts.tempo(static_bpm, "Transport/Tempo/@value", "Stored transport tempo; automation is not reconstructed.")
        meter = _child(transport, "TimeSignature")
        if meter is not None and {"numerator", "denominator"} <= meter.attrib.keys():
            facts.meter(_whole(meter, "numerator"), _whole(meter, "denominator"),
                        "Transport/TimeSignature", "Stored transport meter; automation is not reconstructed.")
    structure = _child(root, "Structure")
    tracks = [] if structure is None else [item for item in structure.iter() if _local(item.tag) == "Track"]
    track_ids = {track.attrib["id"]: index for index, track in enumerate(tracks) if track.attrib.get("id")}
    if structure is not None:
        facts.track_count(len(tracks), "Structure/Track")
        for index, track in enumerate(tracks):
            facts.track(index, track.attrib.get("name"), _track_kind(track), f"Structure/Track[{index}]")
            _plugins(facts, _child(track, "Channel"), index, f"Structure/Track[{index}]/Channel/Devices")
        for channel in (item for item in structure if _local(item.tag) == "Channel"):
            _plugins(facts, channel, None, "Structure/Channel/Devices")
    references = [file.attrib["path"] for media in root.iter() if _local(media.tag) in {"Audio", "Video"}
                  for file in media if _local(file.tag) == "File" and file.attrib.get("path")]
    facts.media_references(references, "Audio/File/@path")
    arrangement = _child(root, "Arrangement")
    if arrangement is None:
        return
    clock = _Clock.constant(static_bpm) if static_bpm else None
    automated, tempo_points = _tempo_automation(arrangement, evidence, source_id)
    if tempo_points:
        facts.tempo_map([{"beat": beat, "bpm": bpm} for beat, bpm in tempo_points], "Arrangement/TempoAutomation")
        clock = automated
        if automated is not None and automated.ramps:
            evidence.diagnostics.append(Diagnostic(
                "dawproject_tempo_ramp_approximation", "info",
                "Tempo automation contains linear ramps. project.tempo_map lists the stored points only, and conversions "
                "between seconds and beats approximate each ramp by its mean tempo.", source_id))
    _meter_automation(arrangement, evidence, source_id, clock)
    reader = _Arrangement(facts, evidence, source_id, track_ids, clock)
    scope = _Scope(DEFAULT_UNIT)
    for child in arrangement:
        if _local(child.tag) in {"Lanes", "Markers"}:
            reader.walk(child, scope)
    for index in sorted(reader.clips):
        facts.clip_count(reader.clips[index], "Arrangement//Lanes//Clip", index)
    for index in sorted(reader.notes):
        _track_notes(facts, evidence, source_id, reader, index)
    for flag, amount in reader.flags.items():
        if flag in NOTICES:
            severity, message = NOTICES[flag]
            evidence.diagnostics.append(Diagnostic(f"dawproject_{flag}", severity, message.format(n=amount), source_id))
    if reader.notes:
        evidence.diagnostics.append(Diagnostic(
            "dawproject_note_timing", "info",
            "Note times are quarter-note beats on the arrangement; seconds-based content is converted with the "
            "stored tempo (a held or mean-ramp approximation when tempo is automated).", source_id))


def _track_notes(facts: ProjectFacts, evidence: Evidence, source_id: str, reader: _Arrangement, index: int) -> None:
    notes = reader.notes[index]
    reason = "Aggregated from stored notes; beats are quarter notes, not audible timing."
    confidence: Confidence = "high"
    placeholders = reader.placeholders.get(index, 0)
    if placeholders:
        reason += (f" {placeholders} of {len(notes)} velocities are placeholders ({PLACEHOLDER_VELOCITY}) because the "
                   "project stores none, so velocity statistics describe notation, not a performance.")
        confidence = "medium"
        evidence.add(source_id, "dawproject.velocity_placeholders",
                     {"placeholder": placeholders, "total": len(notes), "velocity": PLACEHOLDER_VELOCITY}, "count",
                     f"project_track:{index}", "static", "Arrangement//Clip//Notes", "dawproject-xml", "extracted", "high",
                     "Notes whose velocity is the adapter placeholder, not a stored value.")
    if index in reader.ramped:
        reason += " Seconds-based content was placed with a tempo ramp approximated by its mean tempo."
        confidence = "medium"
    facts.track_notes(index, notes, "Arrangement//Clip//Notes", reason, confidence)


def _plugins(facts: ProjectFacts, channel: Element | None, track: int | None, locator: str) -> None:
    devices = _child(channel, "Devices") if channel is not None else None
    for device in devices if devices is not None else ():
        name = device.attrib.get("deviceName", "").strip()
        if name:
            facts.plugin(name[:MAX_METADATA_TEXT], track, f"{locator}/{_local(device.tag)}")
