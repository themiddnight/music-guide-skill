from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from xml.etree.ElementTree import Element

from ..model import Confidence, Detail, Diagnostic, Evidence
from ..notes import PITCH_CLASSES, Note, summarize_notes
from ..project import ProjectFacts, TrackKind, plausible_meter
from .common import MAX_GZIP_BYTES, MAX_ZIP_BYTES, _finite_number, _gunzip_bounded, _local, _parse_xml, _read_bounded
from .listing import capped

METHOD = "ableton-als-xml"
TRACK_TAGS: dict[str, TrackKind] = {"AudioTrack": "audio", "MidiTrack": "midi", "GroupTrack": "group",
                                    "ReturnTrack": "return"}
CLIP_TAGS = frozenset({"MidiClip", "AudioClip"})
PLUGIN_INFO = (("VstPluginInfo", "PlugName"), ("Vst3PluginInfo", "Name"), ("AuPluginInfo", "Name"))
MAX_NOTES = 1_000_000
EPSILON = 1e-9
MAX_DENOMINATOR_EXPONENT = 4


def _number(node: Element | None, label: str) -> float | None:
    """Read the Value attribute of `node`; a missing node or attribute is None, a malformed one is an error."""
    text = None if node is None else node.get("Value")
    if text is None:
        return None
    try:
        return _finite_number(float(text), label)
    except ValueError as error:
        raise ValueError(f"{label} is not a finite number: {text!r}") from error


def _child_number(parent: Element | None, name: str) -> float | None:
    return _number(None if parent is None else parent.find(name), name if parent is None else f"{parent.tag}/{name}")


def _attribute(node: Element, name: str, label: str) -> float:
    text = node.get(name)
    if text is None:
        raise ValueError(f"{label} is missing {name}")
    try:
        return _finite_number(float(text), f"{label} {name}")
    except ValueError as error:
        raise ValueError(f"{label} {name} is not a finite number: {text!r}") from error


def _text(node: Element | None) -> str:
    return "" if node is None else node.get("Value", "").strip()


def _decode_meter(code: float) -> tuple[int, int] | None:
    """Live stores a meter as one integer: (log2(denominator) * 99) + numerator - 1, so 4/4 is 201."""
    if code != int(code) or code < 0:
        return None
    exponent, numerator = divmod(int(code), 99)
    return (numerator + 1, 2 ** exponent) if exponent <= MAX_DENOMINATOR_EXPONENT else None


def _envelope_events(master: Element, device: Element | None) -> list[tuple[float, float]]:
    """Events of the master-track envelope whose target is `device`'s automation target."""
    target = None if device is None else device.find("AutomationTarget")
    target_id = None if target is None else target.get("Id")
    if target_id is None:
        return []
    for envelope in master.findall("AutomationEnvelopes/Envelopes/AutomationEnvelope"):
        pointee = envelope.find("EnvelopeTarget/PointeeId")
        if pointee is not None and pointee.get("Value") == target_id:
            events = envelope.find("Automation/Events")
            return [(_attribute(event, "Time", "automation event"), _attribute(event, "Value", "automation event"))
                    for event in (events if events is not None else []) if "Time" in event.attrib]
    return []


def _breakpoints(events: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sorted (beat, value) pairs; events before beat 0 are the initial value and a later event at the same beat wins."""
    points: dict[float, float] = {}
    for time, value in sorted(events, key=lambda event: event[0]):
        points[max(time, 0.0)] = value
    return sorted(points.items())


@dataclass
class _Transport:
    facts: ProjectFacts
    evidence: Evidence
    source_id: str
    master: Element
    locator: str

    def tempo(self, mixer: Element) -> None:
        tempo = mixer.find("Tempo")
        if tempo is None:
            return
        manual = _child_number(tempo, "Manual")
        points = _breakpoints(_envelope_events(self.master, tempo))
        values = {value for _, value in points}
        base = f"{self.locator}/DeviceChain/Mixer/Tempo"
        if len(values) > 1:
            self.facts.tempo_map([{"beat": beat, "bpm": bpm} for beat, bpm in points], f"{base} automation envelope")
            if manual is not None:
                self.evidence.add(self.source_id, "ableton.manual_tempo", manual, "BPM", "project", "static",
                                  f"{base}/Manual/@Value", METHOD, "extracted", "medium",
                                  "Tempo box value stored beside tempo automation; the automation decides the played tempo.")
        elif values:
            self.facts.tempo(values.pop(), f"{base} automation envelope",
                             "Tempo automation holds one value for the whole set; other settings are not reconstructed.")
        elif manual is not None:
            self.facts.tempo(manual, f"{base}/Manual/@Value",
                             "Stored master tempo; no tempo automation was found in the set.")

    def meter(self, mixer: Element) -> None:
        signature = mixer.find("TimeSignature")
        events = _envelope_events(self.master, signature)
        meters = [(beat, _decode_meter(code)) for beat, code in _breakpoints(events)]
        if any(meter is None for _, meter in meters):
            self._unreadable()
        meters = [(beat, meter) for beat, meter in meters if meter is not None]
        base = f"{self.locator}/DeviceChain/Mixer/TimeSignature"
        encoded = "Decoded from Live's integer meter code (4/4 is 201); the encoding is not publicly documented."
        explicit = self._explicit_meters()
        manual = _child_number(signature, "Manual")
        manual_meter = None if manual is None else _decode_meter(manual)
        if manual is not None and manual_meter is None:
            self._unreadable()
        if len({meter for _, meter in meters}) > 1:
            self._meter_map(meters, f"{base} automation envelope", encoded)
        elif meters:
            self._static(meters[0][1], f"{base} automation envelope", encoded, "low")
        elif manual_meter is not None:
            self._static(manual_meter, f"{base}/Manual/@Value", encoded, "low")
            if explicit and len({meter for _, meter in explicit}) == 1 and explicit[0][1] != manual_meter:
                self.evidence.diagnostics.append(Diagnostic("ableton_meter_disagreement", "warning",
                    f"Ableton meter code reads {manual_meter[0]}/{manual_meter[1]} but TimeSignatures reads "
                    f"{explicit[0][1][0]}/{explicit[0][1][1]}; check the set in Live.", self.source_id))
        elif len({meter for _, meter in explicit}) > 1:
            self._meter_map(explicit, f"{self.locator}/TimeSignatures", "Stored Numerator and Denominator pairs.")
        elif explicit:
            self._static(explicit[0][1], f"{self.locator}/TimeSignatures", "Stored Numerator and Denominator pair.",
                         "medium")

    def _explicit_meters(self) -> list[tuple[float, tuple[int, int]]]:
        found = []
        for entry in self.master.findall("TimeSignatures/RemoteableTimeSignature"):
            numerator, denominator = _child_number(entry, "Numerator"), _child_number(entry, "Denominator")
            if numerator is None or denominator is None:
                continue
            if not plausible_meter(numerator, denominator):
                self.facts.skip_meter(numerator, denominator, f"{self.locator}/TimeSignatures")
                continue
            found.append((max(_child_number(entry, "Time") or 0.0, 0.0), (int(numerator), int(denominator))))
        return sorted(found)

    def _static(self, meter: tuple[int, int], locator: str, reason: str, confidence: Confidence) -> None:
        self.facts.meter(meter[0], meter[1], locator, reason, confidence)

    def _meter_map(self, meters: list[tuple[float, tuple[int, int]]], locator: str, reason: str) -> None:
        self.evidence.add(self.source_id, "ableton.meter_map",
                          [{"beat": beat, "meter": f"{n}/{d}"} for beat, (n, d) in meters], "signature", "project",
                          "project_beats", locator, METHOD, "extracted", "low", reason)

    def _unreadable(self) -> None:
        self.evidence.diagnostics.append(Diagnostic("ableton_unreadable_meter", "warning",
            "An Ableton meter code is outside the known encoding and was ignored.", self.source_id))


@dataclass
class _ClipNotes:
    notes: list[Note] = field(default_factory=list)
    clips: int = 0
    untimed: int = 0
    looped: int = 0
    outside: int = 0
    muted: int = 0
    clamped: int = 0


def _clip_window(clip: Element, arrangement: bool) -> tuple[float, float, float, bool] | None:
    """(start marker, window length, base beat, loops) or None when the clip stores too little to place its notes."""
    loop = clip.find("Loop")
    loop_start, loop_end = _child_number(loop, "LoopStart"), _child_number(loop, "LoopEnd")
    relative = _child_number(loop, "StartRelative")
    if loop_start is None or loop_end is None or relative is None:
        return None
    marker = loop_start + relative
    window = loop_end - marker
    if not arrangement:
        return (marker, window, 0.0, False) if window > 0 else None
    start, end = _child_number(clip, "CurrentStart"), _child_number(clip, "CurrentEnd")
    if start is None or end is None or window <= 0 or end <= start:
        return None
    looping = _text(loop.find("LoopOn")).lower() == "true" and end - start > window + EPSILON
    return marker, min(window, end - start), start, looping


def _read_clip_notes(clip: Element, arrangement: bool, into: _ClipNotes) -> None:
    into.clips += 1
    window = _clip_window(clip, arrangement)
    if window is None:
        into.untimed += 1
        return
    marker, length, base, looping = window
    into.looped += looping
    for key_track in clip.findall("Notes/KeyTracks/KeyTrack"):
        pitch = _child_number(key_track, "MidiKey")
        if pitch is None or pitch != int(pitch) or not 0 <= pitch <= 127:
            raise ValueError("Ableton MidiKey must be an integer from 0 to 127")
        for event in key_track.findall("Notes/MidiNoteEvent"):
            time = _attribute(event, "Time", "MidiNoteEvent")
            duration = _attribute(event, "Duration", "MidiNoteEvent")
            velocity = _attribute(event, "Velocity", "MidiNoteEvent")
            if duration < 0:
                raise ValueError("MidiNoteEvent Duration must not be negative")
            if event.get("IsEnabled", "true").lower() == "false":
                into.muted += 1
            elif not -EPSILON <= time - marker < length - EPSILON:
                into.outside += 1
            else:
                start = base + max(time - marker, 0.0)
                rounded = round(velocity)
                into.clamped += not 0 <= rounded <= 127
                into.notes.append(Note(int(pitch), max(0, min(127, rounded)), start,
                                       min(start + duration, base + length)))
            if len(into.notes) > MAX_NOTES:
                raise ValueError(f"Ableton set exceeds {MAX_NOTES} notes")


def _arrangement_reason(stats: _ClipNotes) -> str:
    return ("Approximate timing. Each position is the clip's CurrentStart plus the note time relative to the clip start "
            "marker (LoopStart + StartRelative), limited to the clip's first pass. Clip loops are not unrolled "
            f"({stats.looped} looped clips run past their first pass; {stats.outside} notes outside the first pass are "
            f"omitted). {stats.muted} muted notes are excluded and {stats.untimed} clips without readable loop and "
            "position fields are skipped. Warping and tempo changes are not applied, and session-view clips have no "
            "arrangement position.")


def _session_reason(stats: _ClipNotes) -> str:
    return ("Notes of session-view clips, which have no arrangement position. Each clip is placed from its own start "
            "marker at beat 0, so clips overlap: pitch statistics are meaningful, but beat span and polyphony are "
            f"not. Loops are not unrolled; {stats.outside} notes outside the first pass and {stats.muted} muted "
            f"notes are excluded, and {stats.untimed} clips without readable loop fields are skipped.")


def _plugin_name(device: Element) -> str | None:
    """The third-party plugin name stored in the device's PluginDesc, or None for any other device element."""
    description = device.find("PluginDesc")
    for info, field_name in PLUGIN_INFO:
        name = _text(None if description is None else description.find(f"{info}/{field_name}"))
        if name:
            return name
    return None


def _sample_names(live_set: Element) -> list[str]:
    names = []
    for reference in live_set.iter("SampleRef"):
        file_ref = reference.find("FileRef")
        if file_ref is None:
            continue
        name = next((text for text in (_text(file_ref.find(tag)) for tag in ("Name", "Path", "RelativePath")) if text), "")
        if name:
            names.append(name)
    return names


@dataclass
class _Track:
    index: int
    element: Element
    kind: TrackKind
    locator: str


def _tracks(live_set: Element, master: Element | None, master_locator: str) -> list[_Track]:
    container = live_set.find("Tracks")
    tracks = [_Track(0, child, TRACK_TAGS[child.tag], f"LiveSet/Tracks/{child.tag}[{position}]")
              for position, child in enumerate(container if container is not None else []) if child.tag in TRACK_TAGS]
    if master is not None:
        tracks.append(_Track(0, master, "master", master_locator))
    for index, track in enumerate(tracks):
        track.index = index
    return tracks


def _track_name(element: Element) -> str | None:
    return _text(element.find("Name/EffectiveName")) or _text(element.find("Name/UserName")) or None


def _scale(facts: ProjectFacts, live_set: Element) -> None:
    scale = live_set.find("ScaleInformation")
    root, name = _child_number(scale, "RootNote"), _text(None if scale is None else scale.find("Name"))
    if root is None or not name or root != int(root) or not 0 <= root <= 11:
        return
    facts.key(f"{PITCH_CLASSES[int(root)]} {name}", "LiveSet/ScaleInformation",
              "Live 12 scale root and name as stored; Live may store a default scale even when Scale Mode is not "
              "used, so this is stored intent, not the audible key.", "low")


def _clips(containers: list[Element]) -> list[Element]:
    return [node for container in containers for node in container.iter() if node.tag in CLIP_TAGS]


def _track_clips(facts: ProjectFacts, track: _Track) -> tuple[int, int]:
    """Emit one track's clip counts and note summaries; returns (arrangement, session) clip counts."""
    sequencer = track.element.find("DeviceChain/MainSequencer")
    if sequencer is None:
        return 0, 0
    base = f"{track.locator}/DeviceChain/MainSequencer"
    arrangement = _clips(sequencer.findall("ClipTimeable"))
    session = _clips(sequencer.findall("ClipSlotList"))
    if arrangement:
        facts.clip_count(len(arrangement), f"{base}/ClipTimeable", track.index)
    if session:
        facts.evidence.add(facts.source_id, "ableton.session_clip_count", len(session), "count",
                           f"project_track:{track.index}", "static", f"{base}/ClipSlotList", METHOD, "extracted",
                           "high", "Count of clips stored in session-view slots.")
    for clips, is_arrangement in ((arrangement, True), (session, False)):
        stats = _ClipNotes()
        for clip in clips:
            if clip.tag == "MidiClip":
                _read_clip_notes(clip, is_arrangement, stats)
        if stats.clamped:
            facts.evidence.diagnostics.append(Diagnostic("ableton_velocity_clamped", "warning",
                f"{stats.clamped} note(s) in Ableton track {track.index} have a velocity outside 0-127 and were "
                "clamped into range.", facts.source_id))
        if not stats.notes:
            continue
        locator = f"{base}/{'ClipTimeable' if is_arrangement else 'ClipSlotList'}//MidiClip"
        if is_arrangement:
            facts.track_notes(track.index, stats.notes, locator, _arrangement_reason(stats), "medium")
        else:
            facts.evidence.add(facts.source_id, "ableton.session_clip_notes", summarize_notes(stats.notes),
                               "note_summary", f"project_track:{track.index}", "static", locator, METHOD,
                               "extracted", "medium", _session_reason(stats))
    return len(arrangement), len(session)


def _parse(path: Path, source_id: str, evidence: Evidence, detail: Detail) -> None:
    root = _parse_xml(_gunzip_bounded(_read_bounded(path, MAX_ZIP_BYTES), MAX_GZIP_BYTES))
    if _local(root.tag) != "Ableton":
        raise ValueError("Ableton set root must be Ableton")
    live_set = root.find("LiveSet")
    if live_set is None:
        raise ValueError("Ableton set has no LiveSet element")
    facts = ProjectFacts(evidence, source_id, METHOD, detail)
    if root.get("Creator", "").strip():
        facts.application(root.attrib["Creator"].strip(), "Ableton/@Creator")
    version = {key: root.attrib[attribute] for key, attribute in
               (("major", "MajorVersion"), ("minor", "MinorVersion"), ("schemaChangeCount", "SchemaChangeCount"))
               if attribute in root.attrib}
    if version:
        evidence.add(source_id, "ableton.version", version, "version", "project", "static", "Ableton/@*", METHOD,
                     "extracted", "high", "Stored Live set schema version attributes.")
    master_tag = "MasterTrack" if live_set.find("MasterTrack") is not None else "MainTrack"
    master = live_set.find(master_tag)
    mixer = None if master is None else master.find("DeviceChain/Mixer")
    if master is not None and mixer is not None:
        transport = _Transport(facts, evidence, source_id, master, f"LiveSet/{master_tag}")
        transport.tempo(mixer)
        transport.meter(mixer)
    _scale(facts, live_set)
    locators = live_set.find("Locators")
    found = [] if locators is None else locators.findall("Locators/Locator") or locators.findall("Locator")
    for number, locator in enumerate(capped(found, "locators", source_id, evidence)):
        position = _child_number(locator, "Time")
        if position is None:
            raise ValueError(f"Locator {number} has no Time")
        facts.marker(_text(locator.find("Name")) or None, position, "beats", f"LiveSet/Locators/Locator[{number}]")
    tracks = _tracks(live_set, master, f"LiveSet/{master_tag}")
    devices = []
    arrangement_total = session_total = 0
    for track in tracks:
        facts.track(track.index, _track_name(track.element), track.kind, track.locator)
        devices += [(_plugin_name(device), device.tag, track.index, f"{track.locator}//Devices/{device.tag}")
                    for chain in track.element.iter("Devices") for device in chain]
        arrangement, session = _track_clips(facts, track)
        arrangement_total += arrangement
        session_total += session
    for plugin, element, index, locator in capped(devices, "devices", source_id, evidence):
        if plugin is not None:
            facts.plugin(plugin, index, locator)
        else:
            evidence.add(source_id, "ableton.device",
                         {"name": element, "track": index, "builtin": not element.startswith("MxDevice")}, "device",
                         f"project_track:{index}", "static", locator, METHOD, "extracted", "medium",
                         "Built-in Ableton device element name from the set XML (a Max for Live device when "
                         "builtin is false); it is an internal tag, not the label shown in Live, and the device "
                         "was not loaded.")
    if live_set.find("Tracks") is not None or master is not None:
        facts.track_count(len(tracks), "LiveSet/Tracks")
        facts.clip_count(arrangement_total, "LiveSet/Tracks//ClipTimeable")
        evidence.add(source_id, "ableton.session_clip_count", session_total, "count", "project", "static",
                     "LiveSet/Tracks//ClipSlotList", METHOD, "extracted", "high",
                     "Count of clips stored in session-view slots.")
    facts.media_references(capped(_sample_names(live_set), "sample references", source_id, evidence),
                           "LiveSet//SampleRef/FileRef")


def ableton(path: Path, source_id: str, evidence: Evidence, detail: Detail) -> None:
    try:
        _parse(path, source_id, evidence, detail)
    except (AttributeError, IndexError, ArithmeticError, RecursionError) as error:
        raise ValueError(f"unexpected Ableton set structure: {error}") from error
