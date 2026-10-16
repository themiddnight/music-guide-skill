"""PreSonus Studio One (.song) reader.

A .song is a ZIP of XML parts. The element and attribute names below follow the public MIT-licensed
studio-one-mcp reader (github.com/NeanderthalMan/studio-one-mcp), which documents them for songs saved by
Studio One 5.5.2. No real song was available to this project, so tests use synthetic archives in that shape.
Instrument notes (Performances/*.musicx, UBJSON) and automation envelopes are not decoded.
"""

from __future__ import annotations

import re
import struct
import zipfile
from pathlib import Path
from xml.etree.ElementTree import Element

from ..model import Detail, Diagnostic, Evidence
from ..project import ProjectFacts, TrackKind
from .common import MAX_XML_BYTES, MAX_ZIP_BYTES, _finite_number, _local, _parse_xml, _read_zip_member

METHOD = "studioone-xml"
SONG_MEMBER = "Song/song.xml"
MAX_ENTRIES = 8192
MAX_TOTAL_BYTES = 8 * MAX_ZIP_BYTES
MAX_METAINFO = 64
MAX_TEXT = 1000
EOCD = b"PK\x05\x06"
EVENT_TAG = re.compile(r"Event$|^MusicPart$|^AudioPart$|^Part$")
STRUCTURE_TRACKS = frozenset({"MarkerTrack", "ArrangerTrack"})
MEDIA_KINDS: dict[str, TrackKind] = {"Audio": "audio", "Music": "instrument"}
SAMPLE_RATE_LOCATOR = "Song/song.xml timeContext/@sampleRate"
UNDECLARED_PREFIX = re.compile(rb'<(/?)([^\s/>!?][^\s/>]*)((?:\s+[^\s=/>]+\s*=\s*(?:"[^"]*"|\'[^\']*\'))*)\s*(/?)>')
GENERATOR = re.compile(r"^(.+?)/(\d[\w.]*)$")


def _entry_count(path: Path) -> int | None:
    """Entry total from the end-of-central-directory record, read before the archive parser builds its table."""
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - 65557))
        tail = stream.read()
    index = tail.rfind(EOCD)
    if index < 0 or len(tail) < index + 22:
        return None
    return struct.unpack_from("<H", tail, index + 10)[0]


def _check_archive(path: Path) -> None:
    count = _entry_count(path)
    if count is None:
        raise ValueError("source is not a readable ZIP archive")
    if count > MAX_ENTRIES:
        raise ValueError(f"archive holds {count} entries, more than the {MAX_ENTRIES} limit")


def detect(path: Path) -> bool:
    with path.open("rb") as stream:
        if stream.read(4) != b"PK\x03\x04":
            return False
    count = _entry_count(path)
    if count is None or count > MAX_ENTRIES:
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return SONG_MEMBER in archive.namelist()
    except (zipfile.BadZipFile, OSError):
        return False


def _member(path: Path, name: str) -> bytes:
    return _read_zip_member(path, name, MAX_XML_BYTES, max_entries=MAX_ENTRIES, max_total=MAX_TOTAL_BYTES)


def _declare_prefixes(raw: bytes) -> bytes:
    """Studio One XML can use prefixes it never declares (x:id) and C++ style names (A::B)."""
    def rename(match: re.Match[bytes]) -> bytes:
        slash, name, attributes, empty = match.groups()
        name = name.replace(b"::", b"_").replace(b":", b"_")
        attributes = re.sub(rb'(\s)([^\s=/>]+)(\s*=)', lambda item: item.group(1) + item.group(2).replace(b"::", b"_")
                            .replace(b":", b"_") + item.group(3), attributes)
        return b"<" + slash + name + attributes + empty + b">"
    return UNDECLARED_PREFIX.sub(rename, raw.removeprefix(b"\xef\xbb\xbf"))


def _parse(raw: bytes) -> Element:
    try:
        return _parse_xml(raw)
    except ValueError as error:
        try:
            return _parse_xml(_declare_prefixes(raw))
        except ValueError:
            raise error from None


def _xid(element: Element) -> str | None:
    for key, value in element.attrib.items():
        if key.endswith("}id") or key == "x_id":
            return value
    return None


def _by_xid(parent: Element | None, xid: str) -> Element | None:
    return None if parent is None else next((item for item in parent if _xid(item) == xid), None)


def _children(parent: Element | None, tag: str | None = None) -> list[Element]:
    return [] if parent is None else [item for item in parent if tag is None or _local(item.tag) == tag]


def _number(element: Element, name: str, default: float | None = None) -> float | None:
    value = element.attrib.get(name)
    if value is None:
        return default
    try:
        return _finite_number(float(value), name)
    except ValueError:
        return default


def _text(value: str) -> str:
    return value[:MAX_TEXT]


def _song_root(song: Element) -> Element:
    for candidate in (_by_xid(song, "Root"), next(iter(_children(song, "Attributes")), None)):
        if candidate is not None:
            return candidate
    return song


def studioone(path: Path, source_id: str, evidence: Evidence, detail: Detail) -> None:
    _check_archive(path)
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
    if SONG_MEMBER not in names:
        raise ValueError(f"archive has no {SONG_MEMBER}; it is not a Studio One song")
    facts = ProjectFacts(evidence, source_id, METHOD, detail)

    def optional(name: str) -> Element | None:
        if name not in names:
            evidence.diagnostics.append(Diagnostic("studioone_part_missing", "info",
                f"The song has no {name}, so the facts stored there are not reported.", source_id))
            return None
        try:
            return _parse(_member(path, name))
        except ValueError as error:
            evidence.diagnostics.append(Diagnostic("studioone_part_unreadable", "warning",
                f"{name} could not be read and was skipped: {error}", source_id))
            return None

    song = _parse(_member(path, SONG_MEMBER))
    if _local(song.tag) != "Song":
        raise ValueError(f"{SONG_MEMBER} root must be Song")

    meta = {}
    metainfo = optional("metainfo.xml")
    for item in _children(metainfo, "Attribute")[:MAX_METAINFO]:
        if "id" in item.attrib and "value" in item.attrib:
            meta[item.attrib["id"]] = _text(item.attrib["value"])
            evidence.add(source_id, "studioone.metainfo", {"id": item.attrib["id"], "value": meta[item.attrib["id"]]},
                         "attribute", "project", "static", f"metainfo.xml Attribute[@id={item.attrib['id']!r}]",
                         METHOD, "extracted", "high", "Stored song metadata attribute, recorded with its literal id.")
    generator = meta.get("Document:Generator")
    if generator:
        match = GENERATOR.match(generator)
        facts.application(f"{match.group(1)} {match.group(2)}" if match else generator,
                          "metainfo.xml Document:Generator")

    root = _song_root(song)
    context = _by_xid(root, "timeContext")
    _emit_timeline(facts, evidence, source_id, context, meta)

    tracks_element = _by_xid(root, "Tracks")
    media_urls, media_ids = _media_pool(optional("Song/mediapool.xml"))
    mixer = _mixer(optional("Devices/audiomixer.xml"))
    structure = [item for item in _children(tracks_element) if _local(item.tag) in STRUCTURE_TRACKS]
    tracks = [item for item in _children(tracks_element) if _local(item.tag) not in STRUCTURE_TRACKS]
    if tracks_element is not None:
        facts.track_count(len(tracks), "Song/song.xml Tracks")
    channel_to_track: dict[str, int] = {}
    for index, track in enumerate(tracks):
        tag, media_type = _local(track.tag), track.attrib.get("mediaType")
        kind = MEDIA_KINDS.get(media_type or "", "unknown") if tag == "MediaTrack" else "unknown"
        locator = f"Song/song.xml Tracks/{tag}[{index}]"
        facts.track(index, _text(track.attrib["name"]) if "name" in track.attrib else None, kind, locator)
        evidence.add(source_id, "studioone.track_type", {"element": tag, "mediaType": media_type}, "element",
                     f"project_track:{index}", "static", locator, METHOD, "extracted", "high",
                     "Stored track element name and mediaType attribute.")
        channel = _by_xid(track, "channelID")
        if channel is not None and "uid" in channel.attrib:
            channel_to_track[channel.attrib["uid"]] = index
        facts.clip_count(_clip_count(track), locator, index)
    _emit_markers(facts, evidence, source_id, structure)
    _emit_mixer(facts, evidence, source_id, mixer, channel_to_track)
    facts.media_references(media_urls, "Song/mediapool.xml AudioClip Url")
    if media_ids:
        evidence.add(source_id, "studioone.media_clip_count", media_ids, "count", "project", "static",
                     "Song/mediapool.xml", METHOD, "extracted", "high", "Count of media pool clips with a media ID.")
    evidence.diagnostics.append(Diagnostic("studioone_notes_not_decoded", "info",
        "Instrument notes (Performances/*.musicx) and automation envelopes are not decoded; clip counts are reported.",
        source_id))


def _emit_timeline(facts: ProjectFacts, evidence: Evidence, source_id: str, context: Element | None,
                   meta: dict[str, str]) -> None:
    rates = []
    if context is not None and (rate := _number(context, "sampleRate")):
        rates.append((rate, SAMPLE_RATE_LOCATOR))
    if meta.get("Media:SampleRate"):
        try:
            rates.append((_finite_number(float(meta["Media:SampleRate"]), "sample rate", positive=True),
                          "metainfo.xml Media:SampleRate"))
        except ValueError:
            pass
    for rate, locator in rates:
        if rate > 0:
            facts.sample_rate(int(rate) if rate.is_integer() else rate, locator)
    if meta.get("Media:KeySignature") not in (None, "", "-"):
        facts.key(meta["Media:KeySignature"], "metainfo.xml Media:KeySignature")
    if context is None:
        return
    tempos = []
    for segment in _children(_by_xid(context, "tempoMap"), "TempoMapSegment"):
        seconds_per_beat, start = _number(segment, "tempo"), _number(segment, "start", 0.0)
        if seconds_per_beat and seconds_per_beat > 0:
            tempos.append((start, round(60 / seconds_per_beat, 6), _number(segment, "curveType", 0.0) != 0))
    tempos.sort()
    if len(tempos) == 1 and tempos[0][0] == 0:
        facts.tempo(tempos[0][1], "Song/song.xml timeContext/tempoMap/TempoMapSegment",
                    "Stored tempo map with one segment (seconds per quarter note converted to BPM).")
    elif tempos:
        facts.tempo_map([{"beat": start, "bpm": bpm} for start, bpm, _ in tempos],
                        "Song/song.xml timeContext/tempoMap")
        if any(ramp for *_, ramp in tempos):
            evidence.diagnostics.append(Diagnostic("studioone_tempo_ramps", "info",
                "Some tempo segments are ramps; the tempo map lists their start tempo only.", source_id))
    meters = []
    for segment in _children(_by_xid(context, "timeSignatureMap"), "TimeSignatureMapSegment"):
        numerator, denominator = _number(segment, "numerator"), _number(segment, "denominator")
        if numerator and denominator and numerator.is_integer() and denominator.is_integer():
            meters.append((_number(segment, "start", 0.0), int(numerator), int(denominator)))
    meters.sort()
    if len(meters) == 1 and meters[0][0] == 0:
        facts.meter(meters[0][1], meters[0][2], "Song/song.xml timeContext/timeSignatureMap")
    elif meters:
        evidence.add(source_id, "studioone.meter_map", [{"beat": start, "numerator": n, "denominator": d}
                     for start, n, d in meters], "signature", "project", "project_beats",
                     "Song/song.xml timeContext/timeSignatureMap", METHOD, "extracted", "high",
                     "Stored time signature changes in beat order.")
    keys = [{"beat": _number(item, "start", 0.0), "root": item.attrib.get("root"), "scale": item.attrib.get("scale")}
            for item in _children(_by_xid(context, "keySignatureMap"))]
    if keys:
        evidence.add(source_id, "studioone.key_signature_map", keys, "key", "project", "project_beats",
                     "Song/song.xml timeContext/keySignatureMap", METHOD, "extracted", "medium",
                     "Stored key signature entries; the root numbering is not decoded.")


def _clip_count(track: Element) -> int:
    layers = _by_xid(track, "Layers")
    if layers is None:
        boxes = [track, *_children(track, "List")]
        return sum(bool(EVENT_TAG.search(_local(item.tag))) for box in boxes for item in box)
    active = int(_number(track, "activeLayer", 0) or 0)
    chosen = [layer for index, layer in enumerate(_children(layers)) if int(_number(layer, "id", index) or 0) == active]
    return sum(bool(EVENT_TAG.search(_local(item.tag))) for layer in chosen
               for item in _children(_by_xid(layer, "Events")))


def _emit_markers(facts: ProjectFacts, evidence: Evidence, source_id: str, structure: list[Element]) -> None:
    for track in structure:
        kind = _local(track.tag)
        for index, event in enumerate(item for item in track if EVENT_TAG.search(_local(item.tag))):
            start, length = _number(event, "start", 0.0), _number(event, "length")
            timeformat = event.attrib.get("timeFormat", track.attrib.get("timeFormat", "2"))
            if timeformat != "2":
                evidence.diagnostics.append(Diagnostic("studioone_time_format", "warning",
                    f"{kind} event {index} uses timeFormat {timeformat!r}, not quarter-note beats; skipped.", source_id))
                continue
            name = _text(event.attrib["name"]) if "name" in event.attrib else None
            locator = f"Song/song.xml {kind}[{index}]"
            if kind == "MarkerTrack":
                facts.marker(name, start, "beats", locator)
            else:
                evidence.add(source_id, "studioone.arranger_section", {"name": name, "startBeat": start,
                             "lengthBeats": length}, "section", "project", "project_beats", locator, METHOD,
                             "extracted", "high", "Stored arranger section.")


def _media_pool(pool: Element | None) -> tuple[list[str], int]:
    urls, clips = [], 0
    for clip in (pool.iter() if pool is not None else ()):
        if "mediaID" not in clip.attrib:
            continue
        clips += 1
        url = next((item for item in clip if _xid(item) == "path" and item.attrib.get("url")), None)
        if url is not None:
            urls.append(url.attrib["url"])
    return urls, clips


def _mixer(root: Element | None) -> list[tuple[str | None, str, list[str]]]:
    """Per mixer channel: its unique ID, its tag and the insert plugin names."""
    channels = []
    for group in (item for item in (root.iter() if root is not None else ()) if _local(item.tag) == "ChannelGroup"):
        for channel in (item for item in group if _local(item.tag).endswith("Channel")):
            uid = _by_xid(channel, "uniqueID")
            inserts = []
            for rack_id in ("Inserts", "PostFaderInserts"):
                for slot in _children(_by_xid(channel, rack_id), "Attributes"):
                    if _xid(slot) is not None:
                        continue
                    device = _by_xid(slot, "deviceData")
                    ghost = _by_xid(_by_xid(slot, "ghostData"), "classInfo")
                    name = next((item.attrib["name"] for item in (device, ghost, slot)
                                 if item is not None and item.attrib.get("name")), None)
                    if name:
                        inserts.append(_text(name))
            channels.append((uid.attrib.get("uid") if uid is not None else None, _local(channel.tag), inserts))
    return channels


def _emit_mixer(facts: ProjectFacts, evidence: Evidence, source_id: str,
                mixer: list[tuple[str | None, str, list[str]]], channel_to_track: dict[str, int]) -> None:
    for uid, tag, inserts in mixer:
        track = channel_to_track.get(uid) if uid else None
        for name in inserts:
            facts.plugin(name, track, f"Devices/audiomixer.xml {tag} inserts")
