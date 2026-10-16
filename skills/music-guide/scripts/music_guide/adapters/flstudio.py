"""FL Studio (.flp) reader.

The format is undocumented by Image-Line. Event IDs and record layouts below follow the public reverse
engineering in PyFLP (pyflp.readthedocs.io, GPL-3.0) and node-flp (github.com/andrewrk/node-flp). Only
facts about the format were taken from them; this module is an independent implementation.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ..model import Confidence, Detail, Diagnostic, Evidence
from ..notes import Note, summarize_notes
from ..project import ProjectFacts, TrackKind
from .common import MAX_ZIP_BYTES, _read_bounded
from .graded import GradedFacts

METHOD = "flp-events"
MAX_FLP_BYTES = MAX_ZIP_BYTES
MAX_EVENTS = 4_000_000
MAX_NOTES = 1_000_000
MAX_TEXT = 1000
HEADER = struct.Struct("<4sIhHH")
EVENTS_OFFSET = HEADER.size + 8
NOTE = struct.Struct("<IHHIHHBBBBBBBB")
PLAYLIST_ITEM = struct.Struct("<IHHIHH2sH4sff")
PLAYLIST_ITEM_SIZE_BEFORE_21, PLAYLIST_ITEM_SIZE_FROM_21 = 32, 60
WRAPPER = "Fruity Wrapper"
MAX_KEY = 131
MIN_TEMPO, MAX_TEMPO = 10.0, 1000.0

BYTE_END, WORD_END, DWORD_END = 64, 128, 192
ID_TIME_SIG_NUMERATOR, ID_TIME_SIG_BEAT, ID_CHANNEL_TYPE = 17, 18, 21
ID_NEW_CHANNEL, ID_NEW_PATTERN, ID_TEMPO_COARSE, ID_SLOT_INDEX, ID_NEW_ARRANGEMENT = 64, 65, 66, 98, 99
ID_INSERT_OUTPUT, ID_INSERT_INPUT, ID_TEMPO = 147, 154, 156
ID_CHANNEL_NAME, ID_PATTERN_NAME, ID_TITLE, ID_COMMENTS, ID_SAMPLE_PATH = 192, 193, 194, 195, 196
ID_VERSION, ID_PLUGIN_INTERNAL_NAME, ID_PLUGIN_NAME, ID_INSERT_NAME, ID_GENRE, ID_ARTISTS = 199, 201, 203, 204, 206, 207
ID_NOTES, ID_PLAYLIST, ID_INSERT_ROUTING, ID_INSERT_FLAGS, ID_ARRANGEMENT_NAME = 224, 233, 235, 236, 241
MIXER_IDS = frozenset({ID_SLOT_INDEX, ID_INSERT_OUTPUT, ID_INSERT_INPUT, ID_INSERT_NAME, ID_INSERT_ROUTING,
                       ID_INSERT_FLAGS})
PROJECT_STRINGS = {ID_TITLE: "flstudio.title", ID_COMMENTS: "flstudio.comments", ID_GENRE: "flstudio.genre",
                   ID_ARTISTS: "flstudio.artists"}

CHANNEL_SAMPLER, CHANNEL_NATIVE, CHANNEL_LAYER, CHANNEL_INSTRUMENT, CHANNEL_AUTOMATION = 0, 2, 3, 4, 5
CHANNEL_KINDS: dict[int, TrackKind] = {CHANNEL_LAYER: "group", CHANNEL_INSTRUMENT: "instrument"}
METER_DENOMINATORS = frozenset({2, 4, 8, 16})
HEADER_CAVEAT = "Layout taken from public reverse engineering of the undocumented format."
METER_CAVEAT = ("Event IDs 17 and 18 are interpreted as the global time signature after PyFLP; node-flp names "
                "them pattern and block length, and FL's time-division mode is not distinguished.")
LAYOUT_CAVEAT = "Read from the undocumented FLP event layout, which was checked against public reverse engineering only."
KIND_CAVEAT = (LAYOUT_CAVEAT + " The channel kind comes from the stored channel type; the built-in Sampler (type 0) "
               "plays pattern notes and has no audio or instrument flag, and a channel with a sample and no plugin "
               "is either an audio clip or a sampler channel.")
WRAPPER_CAVEAT = ("'Fruity Wrapper' is the internal name FL gives every hosted VST or AU plugin, so the real plugin "
                  "is named only by the plugin display name (event 203), which the user can rename.")
NOTES_CAVEAT = "Note layout follows PyFLP and was not checked against a real project."


def detect(path: Path) -> bool:
    with path.open("rb") as stream:
        head = stream.read(EVENTS_OFFSET)
    return len(head) == EVENTS_OFFSET and head.startswith(b"FLhd") and head[HEADER.size:HEADER.size + 4] == b"FLdt"


@dataclass
class _Channel:
    iid: int
    type: int | None = None
    name: str | None = None
    internal_name: str | None = None
    plugin_name: str | None = None
    sample_path: str | None = None


@dataclass
class _Pattern:
    iid: int
    name: str | None = None
    notes: list[Note] = field(default_factory=list)
    rack_channels: set[int] = field(default_factory=set)
    zero_length_notes: int = 0


@dataclass
class _Slot:
    internal_name: str | None = None
    name: str | None = None


@dataclass
class _Arrangement:
    iid: int
    name: str | None = None
    items: list[tuple[int, int, int, int]] = field(default_factory=list)  # position, length, item index, base


def _u(raw: bytes) -> int:
    return int.from_bytes(raw, "little")


def _varint(data: bytes, pos: int, end: int, offset: int) -> tuple[int, int]:
    value = shift = 0
    for _ in range(5):
        if pos >= end:
            raise ValueError(f"FLP event at byte {offset} ends inside its length field; the file is truncated")
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
    raise ValueError(f"FLP event at byte {offset} has an invalid length field")


def _events(data: bytes, start: int, end: int) -> Iterator[tuple[int, int, bytes]]:
    pos, count = start, 0
    while pos < end:
        count += 1
        if count > MAX_EVENTS:
            raise ValueError(f"FLP file has more than {MAX_EVENTS} events")
        offset, event_id = pos, data[pos]
        pos += 1
        if event_id < BYTE_END:
            size = 1
        elif event_id < WORD_END:
            size = 2
        elif event_id < DWORD_END:
            size = 4
        else:
            size, pos = _varint(data, pos, end, offset)
        if size > end - pos:
            raise ValueError(f"FLP event {event_id} at byte {offset} declares {size} bytes "
                             f"but only {end - pos} remain; the file is truncated or corrupt")
        yield event_id, offset, data[pos:pos + size]
        pos += size


def _version_text(data: bytes, start: int, end: int) -> str:
    for event_id, _, raw in _events(data, start, end):
        if event_id == ID_VERSION:
            return raw.decode("ascii", errors="replace").rstrip("\0")
    raise ValueError("FLP file has no FL Studio version event; it may be truncated or from an unsupported version")


def _uses_utf16(version: str) -> bool:
    try:
        major, minor = (int(part) for part in version.split(".")[:2])
    except ValueError as error:
        raise ValueError(f"FLP version {version[:40]!r} is not in major.minor.patch form") from error
    return (major, minor) >= (11, 5)


def _major(version: str) -> int:
    return int(version.split(".")[0])


def _text(raw: bytes, utf16: bool) -> str:
    text = raw[:len(raw) // 2 * 2].decode("utf-16-le", errors="replace") if utf16 else raw.decode("ascii", errors="replace")
    return text.split("\0")[0][:MAX_TEXT]


def _channel_kind(channel: _Channel) -> TrackKind:
    if channel.type in (CHANNEL_NATIVE, CHANNEL_INSTRUMENT) and channel.sample_path and not channel.internal_name:
        # PyFLP treats a native or instrument channel with a sample and no plugin as a sample channel.
        return "audio"
    if channel.type == CHANNEL_NATIVE:
        return "instrument"
    return CHANNEL_KINDS.get(channel.type, "unknown") if channel.type is not None else "unknown"


def _plugin_label(internal_name: str | None, display_name: str | None) -> tuple[str, bool] | None:
    """The plugin name to report and whether it is the unverified display name of a Fruity Wrapper host."""
    if internal_name != WRAPPER:
        return (internal_name or display_name, False) if internal_name or display_name else None
    return display_name or WRAPPER, True


def flstudio(path: Path, source_id: str, evidence: Evidence, detail: Detail) -> None:
    data = _read_bounded(path, MAX_FLP_BYTES)
    if len(data) < EVENTS_OFFSET or data[:4] != b"FLhd":
        raise ValueError("not an FL Studio project: missing FLhd header")
    _, header_size, file_format, channel_count, ppq = HEADER.unpack_from(data)
    if header_size != 6 or data[HEADER.size:HEADER.size + 4] != b"FLdt":
        raise ValueError("FLP header is malformed: expected a 6 byte FLhd chunk followed by FLdt")
    if ppq <= 0:
        raise ValueError(f"FLP header declares an invalid PPQ of {ppq}")
    events_size = _u(data[HEADER.size + 4:EVENTS_OFFSET])
    available = len(data) - EVENTS_OFFSET
    if events_size > available:
        raise ValueError(f"FLP data chunk declares {events_size} bytes but the file holds only {available}")
    end = EVENTS_OFFSET + events_size
    if events_size < available:
        evidence.diagnostics.append(Diagnostic("flp_trailing_bytes", "info",
            f"{available - events_size} bytes after the FLP data chunk were ignored.", source_id))
    version = _version_text(data, EVENTS_OFFSET, end)
    utf16 = _uses_utf16(version)

    channels: list[_Channel] = []
    patterns: dict[int, _Pattern] = {}
    arrangements: list[_Arrangement] = []
    slots: list[_Slot] = []
    insert_names: list[str] = []
    strings: dict[int, str] = {}
    numbers: dict[int, int] = {}
    context = None
    channel = pattern = arrangement = slot = None
    skipped: list[str] = []
    note_total = 0

    def current_arrangement() -> _Arrangement:
        nonlocal arrangement
        if arrangement is None:
            arrangement = _Arrangement(0)
            arrangements.append(arrangement)
        return arrangement

    for event_id, offset, raw in _events(data, EVENTS_OFFSET, end):
        if event_id == ID_NEW_CHANNEL:
            channel = _Channel(_u(raw))
            channels.append(channel)
            context = "channel"
        elif event_id == ID_NEW_PATTERN:
            pattern = patterns.setdefault(_u(raw), _Pattern(_u(raw)))
            context = "pattern"
        elif event_id == ID_NEW_ARRANGEMENT:
            arrangement = _Arrangement(_u(raw))
            arrangements.append(arrangement)
            context = "arrangement"
        else:
            if event_id in MIXER_IDS:
                context = "mixer"
            if event_id == ID_SLOT_INDEX:
                slot = _Slot()
                slots.append(slot)
            elif event_id in (ID_TIME_SIG_NUMERATOR, ID_TIME_SIG_BEAT, ID_TEMPO, ID_TEMPO_COARSE):
                numbers.setdefault(event_id, _u(raw))
            elif event_id == ID_CHANNEL_TYPE and context == "channel" and channel is not None:
                channel.type = _u(raw)
            elif event_id == ID_INSERT_NAME:
                insert_names.append(_text(raw, utf16))
            elif event_id in PROJECT_STRINGS:
                strings.setdefault(event_id, _text(raw, utf16))
            elif event_id in (ID_CHANNEL_NAME, ID_PLUGIN_NAME, ID_PLUGIN_INTERNAL_NAME, ID_SAMPLE_PATH):
                value = _text(raw, utf16)
                if context == "channel" and channel is not None:
                    if event_id == ID_PLUGIN_INTERNAL_NAME:
                        channel.internal_name = value or None
                    elif event_id == ID_SAMPLE_PATH:
                        channel.sample_path = value or None
                    elif event_id == ID_PLUGIN_NAME:
                        channel.plugin_name = value or channel.plugin_name
                        channel.name = value or channel.name
                    elif channel.name is None:
                        channel.name = value or None
                elif context == "mixer" and event_id in (ID_PLUGIN_NAME, ID_PLUGIN_INTERNAL_NAME):
                    if slot is None:
                        slot = _Slot()
                        slots.append(slot)
                    if event_id == ID_PLUGIN_INTERNAL_NAME:
                        slot.internal_name = value or None
                    else:
                        slot.name = value or None
            elif event_id == ID_PATTERN_NAME and context == "pattern" and pattern is not None:
                pattern.name = _text(raw, utf16) or None
            elif event_id == ID_ARRANGEMENT_NAME and context == "arrangement" and arrangement is not None:
                arrangement.name = _text(raw, utf16) or None
            elif event_id == ID_NOTES and context == "pattern" and pattern is not None:
                note_total += len(raw) // NOTE.size
                if note_total > MAX_NOTES:
                    raise ValueError(f"FLP file has more than {MAX_NOTES} notes")
                _read_notes(pattern, raw, ppq, offset, skipped)
            elif event_id == ID_PLAYLIST and context == "arrangement":
                _read_playlist(current_arrangement(), raw, _major(version), offset, skipped)

    _emit(evidence, source_id, detail, file_format, channel_count, ppq, version, numbers, strings, channels,
          patterns, arrangements, slots, insert_names)
    for message in skipped:
        evidence.diagnostics.append(Diagnostic("flp_event_skipped", "warning", message, source_id))


def _read_notes(pattern: _Pattern, raw: bytes, ppq: int, offset: int, skipped: list[str]) -> None:
    if len(raw) % NOTE.size:
        skipped.append(f"FLP notes event at byte {offset} holds {len(raw)} bytes, not a multiple of "
                       f"{NOTE.size}; pattern {pattern.iid} notes were skipped.")
        return
    for index in range(len(raw) // NOTE.size):
        (position, _flags, rack_channel, length, key, _group, _fine_pitch, _unknown, _release, _midi_channel, _pan,
         velocity, _mod_x, _mod_y) = NOTE.unpack_from(raw, index * NOTE.size)
        if key > MAX_KEY:
            skipped.append(f"FLP pattern {pattern.iid} note {index} has key {key}, above {MAX_KEY}; skipped.")
            continue
        pattern.notes.append(Note(key, velocity, position / ppq, (position + length) / ppq, None))
        pattern.rack_channels.add(rack_channel)
        pattern.zero_length_notes += length == 0


def _read_playlist(arrangement: _Arrangement, raw: bytes, major: int, offset: int, skipped: list[str]) -> None:
    size = PLAYLIST_ITEM_SIZE_FROM_21 if major >= 21 else PLAYLIST_ITEM_SIZE_BEFORE_21
    if len(raw) % size:
        skipped.append(f"FLP playlist event at byte {offset} holds {len(raw)} bytes, which is not a whole number of "
                       f"{size} byte items, the item size for FL Studio {major}; the playlist was skipped.")
        return
    for index in range(len(raw) // size):
        position, base, item_index, length, *_ = PLAYLIST_ITEM.unpack_from(raw, index * size)
        arrangement.items.append((position, length, item_index, base))


def _emit(evidence: Evidence, source_id: str, detail: Detail, file_format: int, channel_count: int,
          ppq: int, version: str, numbers: dict[int, int], strings: dict[int, str], channels: list[_Channel],
          patterns: dict[int, _Pattern], arrangements: list[_Arrangement], slots: list[_Slot],
          insert_names: list[str]) -> None:
    facts = ProjectFacts(evidence, source_id, METHOD, detail)
    low = GradedFacts(evidence, source_id, METHOD, detail, grade="low", caveat=METER_CAVEAT)
    layout = GradedFacts(evidence, source_id, METHOD, detail, grade="medium", caveat=LAYOUT_CAVEAT)
    kinds = GradedFacts(evidence, source_id, METHOD, detail, grade="medium", caveat=KIND_CAVEAT)
    wrapped = GradedFacts(evidence, source_id, METHOD, detail, grade="low", caveat=WRAPPER_CAVEAT)

    def add(property: str, value, unit: str, scope: str, locator: str, reason: str, *,
            time_domain: str = "static", confidence: Confidence = "high") -> None:
        evidence.add(source_id, property, value, unit, scope, time_domain, locator, METHOD, "extracted", confidence,
                     reason)

    def plugin(internal_name: str | None, display_name: str | None, track: int | None, locator: str) -> None:
        label = _plugin_label(internal_name, display_name)
        if label is None:
            return
        name, hosted = label
        (wrapped if hosted else layout).plugin(name, track, locator)

    facts.application(f"FL Studio {version}", "event 199 (FL version text)")
    add("flstudio.header", {"format": file_format, "channelCount": channel_count, "ppq": ppq}, "header",
        "project", "FLhd", f"Stored FLP header fields. {HEADER_CAVEAT}")
    if ID_TEMPO in numbers:
        tempo, tempo_locator = numbers[ID_TEMPO] / 1000, "event 156 (tempo x 1000)"
        tempo_reason = ("Tempo stored in the project events; PyFLP documents it as the tempo at the saved "
                        "playhead, so tempo automation is not reconstructed.")
    elif ID_TEMPO_COARSE in numbers:
        tempo, tempo_locator = float(numbers[ID_TEMPO_COARSE]), "event 66 (integer BPM)"
        tempo_reason = "Legacy whole-BPM tempo event of older FL Studio versions; tempo automation is not reconstructed."
    else:
        tempo = tempo_locator = tempo_reason = None
    if tempo is not None and MIN_TEMPO <= tempo <= MAX_TEMPO:
        facts.tempo(tempo, tempo_locator, tempo_reason)
    elif tempo is not None:
        evidence.diagnostics.append(Diagnostic("flp_tempo_out_of_range", "warning",
            f"The FLP tempo event holds {tempo} BPM, outside {MIN_TEMPO}-{MAX_TEMPO}; it was not reported.", source_id))
    if ID_TIME_SIG_NUMERATOR in numbers and ID_TIME_SIG_BEAT in numbers:
        numerator, beat = numbers[ID_TIME_SIG_NUMERATOR], numbers[ID_TIME_SIG_BEAT]
        add("flstudio.time_signature", {"numerator": numerator, "beat": beat}, "signature", "project",
            "events 17 and 18", METER_CAVEAT, confidence="low")
        if numerator >= 1 and beat in METER_DENOMINATORS:
            low.meter(numerator, beat, "events 17 and 18")
    for event_id, property in PROJECT_STRINGS.items():
        if strings.get(event_id):
            add(property, strings[event_id], "text", "project", f"event {event_id}", "Stored project information text.")
    layout.track_count(len(channels), "channel rack (event 64)")
    media: list[str] = []
    for index, channel in enumerate(channels):
        locator = f"channel iid={channel.iid}"
        kinds.track(index, channel.name, _channel_kind(channel), locator)
        add("flstudio.channel", {"iid": channel.iid, "type": channel.type, "plugin": channel.internal_name},
            "channel", f"project_track:{index}", locator,
            f"Stored channel type ID and plugin internal name. {LAYOUT_CAVEAT}", confidence="medium")
        plugin(channel.internal_name, channel.plugin_name, index, locator)
        if channel.sample_path:
            media.append(channel.sample_path)
    if len(channels) != channel_count:
        evidence.diagnostics.append(Diagnostic("flp_channel_count_mismatch", "info",
            f"The FLP header counts {channel_count} channels but {len(channels)} channel records were read.", source_id))
    for slot in slots:
        plugin(slot.internal_name, slot.name, None, "mixer effect slot")
    for name in insert_names:
        if name:
            add("flstudio.mixer_insert_name", name, "text", "project", "event 204",
                "Name of a renamed mixer insert; insert positions are not decoded.")
    layout.media_references(media, "channel sample paths (event 196)")
    zero_length = 0
    for pattern in sorted(patterns.values(), key=lambda item: item.iid):
        scope = f"flstudio_pattern:{pattern.iid}"
        add("flstudio.pattern", {"id": pattern.iid, "name": pattern.name, "noteCount": len(pattern.notes)},
            "pattern", scope, f"pattern {pattern.iid}", "Stored pattern number, name and note count.")
        if not pattern.notes:
            continue
        summary = summarize_notes(pattern.notes) | {"rackChannelIds": sorted(pattern.rack_channels)}
        evidence.add(source_id, "flstudio.pattern_notes", summary, "note_summary", scope, "static",
                     f"pattern {pattern.iid} notes (event 224)", METHOD, "extracted", "medium",
                     f"Aggregated from stored notes; beats are quarter notes inside the pattern, not arrangement "
                     f"time. {NOTES_CAVEAT}")
        zero_length += pattern.zero_length_notes
        if detail == "full":
            for note in pattern.notes:
                evidence.add(source_id, "flstudio.note", {"pitch": note.pitch, "velocity": note.velocity,
                             "startBeat": note.start_beat, "endBeat": note.end_beat}, "note", scope,
                             "pattern_beats", f"pattern {pattern.iid} notes (event 224)", METHOD, "extracted",
                             "medium", f"Stored note, positioned inside its pattern. {NOTES_CAVEAT}")
    if zero_length:
        evidence.diagnostics.append(Diagnostic("flp_zero_length_notes", "info",
            f"{zero_length} notes have a stored length of 0 (step sequencer notes); their end equals their start "
            f"and maxPolyphony does not count them.", source_id))
    playlist_items = 0
    for index, arrangement in enumerate(arrangements):
        if not arrangement.items:
            continue
        playlist_items += len(arrangement.items)
        uses = sum(item[2] > item[3] for item in arrangement.items)
        span = max((position + length) / ppq for position, length, *_ in arrangement.items)
        add("flstudio.arrangement", {"index": index, "name": arrangement.name, "itemCount": len(arrangement.items),
                                     "patternItemCount": uses,
                                     "channelItemCount": len(arrangement.items) - uses, "spanBeats": span},
            "arrangement", f"flstudio_arrangement:{index}", f"arrangement {index} playlist (event 233)",
            f"Stored playlist item counts and end of the last item in quarter notes. {NOTES_CAVEAT}",
            confidence="medium")
    if sum(bool(item.items) for item in arrangements) == 1:
        layout.clip_count(playlist_items, "playlist (event 233)")
