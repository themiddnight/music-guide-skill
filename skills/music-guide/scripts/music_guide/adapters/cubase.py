"""Cubase and Nuendo (.cpr, .npr) reader.

The format is undocumented. Chunk and node layout follow cbconv (code.volse.net/audio/cbconv, GPL-3.0), whose
author reverse engineered the RIFF-like container: a RIFF header with an extra NUND tag, then ROOT/ARCH chunk
pairs holding nodes that start with FF FF FF FE (container) or FF FF FF FF (object). The attribute records in
the Metadata chunk and the track header pattern were found by inspecting one Cubase 11 project and are not
documented anywhere. Only the application version and the Metadata attributes are decoded structurally. Track
names, plugin names and media file names come from byte patterns in the object stream and are reported at
low confidence. This module is an independent implementation.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

from ..model import Detail, Diagnostic, Evidence
from ..project import ProjectFacts, TrackKind, plausible_meter, plausible_tempo
from .common import MAX_ZIP_BYTES, _read_bounded
from .graded import GradedFacts

METHOD = "cubase-chunks"
MAX_CPR_BYTES = MAX_ZIP_BYTES
MAX_CHUNKS = 10_000
MAX_SCAN_HITS = 1_000_000
MAX_DEPTH = 32
MAX_MATCHES = 20_000
MAX_FIELDS = 64
MAX_RECORDS = 512
MAX_NESTING = 4
MAX_CLASS_NAME = 64
MAX_TEXT = 1000
BOM = b"\xef\xbb\xbf"
NUL_BOM = b"\0" + BOM
STRING_TYPE = b"\x00\x08"
CONTAINER, OBJECT = b"\xff\xff\xff\xfe", b"\xff\xff\xff\xff"
IDENTIFIER = re.compile(rb"^[A-Za-z_][A-Za-z0-9_:]*$")
MEDIA_NAME_END = re.compile(rb"\.(?:wav|wave|aif|aiff|flac|mp3|ogg|m4a|w64|caf|bwf)\x00\xef\xbb\xbf", re.IGNORECASE)
MAX_STRING_WALK = 300
RECORD_CLASS = b"StMedia::PStringIDMediaAttribute"
RECORD_MARK = struct.pack(">HI", 0x0014, len(RECORD_CLASS) + 1) + RECORD_CLASS + b"\0"
TRACK_BASE_CLASS = "MListNode"
TRACK_CLASS_KINDS: dict[str, TrackKind] = {"MAudioTrackEvent": "audio", "MMidiTrackEvent": "midi",
                                           "MInstrumentTrackEvent": "instrument"}
VALUE_FIELDS = ("Long", "Float", "String", "Object")
PATTERN_CAVEAT = ("Found by matching a byte pattern in the undocumented object stream, not by decoding it, so "
                  "entries can be missing or extra; checked against one Cubase 11 project only.")
GUIDANCE = "Export a DAWproject (Cubase 14 or later) or a MIDI file, plus a mixdown, for complete data."


def detect(path: Path) -> bool:
    with path.open("rb") as stream:
        head = stream.read(12)
    return len(head) == 12 and head.startswith(b"RIFF") and head[8:12] == b"NUND"


@dataclass(frozen=True)
class Node:
    container: bool
    name: str
    version: int
    length: int
    payload_start: int


@dataclass(frozen=True)
class Chunk:
    tag: bytes
    start: int
    length: int


def _read_chunks(data: bytes) -> list[Chunk]:
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"NUND":
        raise ValueError("not a Cubase or Nuendo project: missing RIFF header with NUND tag")
    chunks, pos = [], 12
    while pos < len(data):
        if len(data) - pos < 8:
            raise ValueError(f"Cubase chunk header at byte {pos} is truncated")
        tag, length = data[pos:pos + 4], struct.unpack_from(">I", data, pos + 4)[0]
        if not tag.isalnum() or not tag.isascii():
            raise ValueError(f"Cubase chunk at byte {pos} has an invalid tag")
        if length > len(data) - pos - 8:
            raise ValueError(f"Cubase chunk {tag.decode()} at byte {pos} declares {length} bytes "
                             f"but only {len(data) - pos - 8} remain; the file is truncated or corrupt")
        chunks.append(Chunk(tag, pos + 8, length))
        if len(chunks) > MAX_CHUNKS:
            raise ValueError(f"Cubase file has more than {MAX_CHUNKS} chunks")
        pos += 8 + length
    return chunks


def _string(buf: bytes, pos: int, end: int) -> tuple[str, int]:
    """Length-prefixed string: big-endian length, then text up to its first NUL (a UTF-8 BOM may follow)."""
    if end - pos < 4:
        raise ValueError("Cubase string header is truncated")
    length = struct.unpack_from(">I", buf, pos)[0]
    if length > end - pos - 4:
        raise ValueError(f"Cubase string declares {length} bytes but only {end - pos - 4} remain")
    return buf[pos + 4:pos + 4 + length].split(b"\0")[0].decode("utf-8", errors="replace")[:MAX_TEXT], pos + 4 + length


def _node(buf: bytes, pos: int, end: int) -> Node | None:
    marker = buf[pos:pos + 4]
    if marker not in (CONTAINER, OBJECT) or end - pos < 12:
        return None
    size = struct.unpack_from(">I", buf, pos + 4)[0]
    name_end = pos + 8 + size
    if not 2 <= size <= MAX_CLASS_NAME + 1 or name_end + 2 > end or buf[name_end - 1] != 0:
        return None
    name = buf[pos + 8:name_end - 1]
    if not IDENTIFIER.match(name):
        return None
    version = struct.unpack_from(">H", buf, name_end)[0]
    if marker == CONTAINER:
        return Node(True, name.decode(), version, 0, name_end + 2)
    if name_end + 6 > end:
        return None
    length = struct.unpack_from(">I", buf, name_end + 2)[0]
    if length > end - name_end - 6:
        return None
    return Node(False, name.decode(), version, length, name_end + 6)


def _app_version(buf: bytes) -> dict[str, str]:
    pos = 0
    while pos < len(buf):
        node = _node(buf, pos, len(buf))
        if node is None or (not node.container and node.name != "PAppVersion"):
            break
        if node.container:
            pos = node.payload_start
            continue
        end = node.payload_start + node.length
        name, at = _string(buf, node.payload_start, end)
        version, at = _string(buf, at, end)
        date, at = _string(buf, at, end)
        if not name or end - at < 4:
            break
        fields = {"name": name, "version": version, "buildDate": date}
        at += 4
        if end - at >= 4:
            try:
                architecture, at = _string(buf, at, end)
                encoding, at = _string(buf, at + 4, end)
                locale, at = _string(buf, at, end)
                fields |= {"architecture": architecture, "encoding": encoding, "locale": locale}
            except ValueError:
                pass
        return fields
    raise ValueError("Cubase version chunk has no readable PAppVersion object")


def _record(buf: bytes, pos: int, end: int, depth: int = 0) -> tuple[dict[str, object], int]:
    """One attribute record: 0x0014, class name, 8 opaque bytes, field count, then name/type/value fields."""
    if depth > MAX_NESTING:
        raise ValueError("Cubase attribute records nest too deeply")
    if end - pos < 2 or struct.unpack_from(">H", buf, pos)[0] != 0x0014:
        raise ValueError("expected an attribute record")
    _, pos = _string(buf, pos + 2, end)
    if end - pos < 12:
        raise ValueError("Cubase attribute record is truncated")
    count = struct.unpack_from(">I", buf, pos + 8)[0]
    if count > MAX_FIELDS:
        raise ValueError(f"Cubase attribute record declares {count} fields")
    pos += 12
    fields: dict[str, object] = {}
    for _ in range(count):
        name, pos = _string(buf, pos, end)
        if end - pos < 2:
            raise ValueError("Cubase attribute field is truncated")
        kind = struct.unpack_from(">H", buf, pos)[0]
        pos += 2
        if kind == 1 and end - pos >= 8:
            value, pos = struct.unpack_from(">q", buf, pos)[0], pos + 8
        elif kind == 4 and end - pos >= 8:
            value, pos = struct.unpack_from(">d", buf, pos)[0], pos + 8
        elif kind == 8:
            value, pos = _string(buf, pos, end)
        elif kind == 2:
            value, pos = _record(buf, pos, end, depth + 1)
        else:
            raise ValueError(f"Cubase attribute field {name!r} has unsupported type {kind}")
        fields[name] = value
    return fields, pos


def _media_attributes(buf: bytes) -> list[dict[str, object]]:
    pos = buf.find(RECORD_MARK)
    if pos < 0:
        raise ValueError("Metadata chunk has no media attribute records")
    attributes = []
    while pos < len(buf) and buf[pos:pos + len(RECORD_MARK)] == RECORD_MARK and len(attributes) < MAX_RECORDS:
        fields, pos = _record(buf, pos, len(buf))
        value = next((fields[name] for name in VALUE_FIELDS if name in fields), None)
        if isinstance(fields.get("ID"), str) and value is not None:
            attributes.append({"id": fields["ID"], "value": value})
    return attributes


def _scan_classes(buf: bytes) -> dict[int, Node]:
    table: dict[int, Node] = {}
    stack, hits = [(0, len(buf), 0)], 0
    while stack:
        pos, end, depth = stack.pop()
        while pos < end:
            index = buf.find(b"\xff\xff\xff", pos, end)
            if index < 0:
                break
            hits += 1
            if hits > MAX_SCAN_HITS:
                raise ValueError("Cubase object stream has too many candidate nodes to scan")
            node = _node(buf, index, end)
            if node is None:
                pos = index + 1
                continue
            table[index] = node
            pos = node.payload_start + node.length
            if not node.container and depth < MAX_DEPTH:
                stack.append((node.payload_start, pos, depth + 1))
    return table


def _printable(text: str) -> bool:
    return bool(text.strip()) and all(char.isprintable() for char in text)


def _bom_string_at(buf: bytes, pos: int) -> str | None:
    """A string whose length counts a trailing NUL and UTF-8 BOM, as Cubase writes user-visible text."""
    if pos + 4 > len(buf):
        return None
    length = struct.unpack_from(">I", buf, pos)[0]
    end = pos + 4 + length
    if not 5 <= length <= MAX_STRING_WALK or end > len(buf) or buf[end - 4:end] != NUL_BOM:
        return None
    try:
        text = buf[pos + 4:end - 4].decode("utf-8")
    except UnicodeDecodeError:
        return None
    return text if _printable(text) else None


def _track_class(buf: bytes, at: int, table: dict[int, Node], inline: dict[int, Node]) -> str | None:
    for pos in range(max(at - 40, 0), at - 3):
        value = struct.unpack_from(">I", buf, pos)[0]
        node = table.get(value & 0x7FFFFFFF) if value & 0x80000000 else None
        if node is not None and node.name.endswith("TrackEvent"):
            return node.name
    for start in range(at - 1, max(at - 40, 0), -1):
        node = inline.get(start)
        if node is not None and node.name.endswith("TrackEvent"):
            return node.name
    return None


def _tracks(buf: bytes, table: dict[int, Node]) -> list[tuple[str, str | None]]:
    base = next((offset for offset, node in table.items() if node.container and node.name == TRACK_BASE_CLASS), None)
    if base is None:
        return []
    reference = struct.pack(">I", 0x80000000 | base)
    inline = {node.payload_start: node for node in table.values() if not node.container}
    found, start = [], 0
    while len(found) < MAX_MATCHES:
        at = buf.find(reference, start)
        if at < 0:
            break
        start = at + 1
        name = _bom_string_at(buf, at + 8)
        if name is not None:
            found.append((name, _track_class(buf, at, table, inline)))
    return found


def _plugin_names(buf: bytes) -> list[str]:
    key = struct.pack(">I", 12) + b"Plugin Name\0"
    names, start = [], 0
    while len(names) < MAX_MATCHES:
        at = buf.find(key, start)
        if at < 0:
            break
        start = at + len(key)
        if buf[start:start + 2] == STRING_TYPE:
            name = _bom_string_at(buf, start + 2)
            if name is not None:
                names.append(name)
    return names


def _media_names(buf: bytes) -> list[str]:
    names = []
    for match in MEDIA_NAME_END.finditer(buf):
        if len(names) >= MAX_MATCHES:
            break
        end = match.end()
        for start in range(match.start(), max(match.start() - MAX_STRING_WALK, 4), -1):
            if struct.unpack_from(">I", buf, start - 4)[0] == end - start:
                text = buf[start:end - 4].decode("utf-8", errors="replace")
                if _printable(text):
                    names.append(text)
                break
    return names


def cubase(path: Path, source_id: str, evidence: Evidence, detail: Detail) -> None:
    data = _read_bounded(path, MAX_CPR_BYTES)
    chunks = _read_chunks(data)
    roots: list[tuple[str, str, Chunk]] = []
    for index, chunk in enumerate(chunks[:-1]):
        if chunk.tag == b"ROOT" and chunks[index + 1].tag == b"ARCH":
            end = chunk.start + chunk.length
            name, at = _string(data, chunk.start, end)
            kind, _ = _string(data, at, end)
            roots.append((name, kind, chunks[index + 1]))
    version_chunk = next((arch for name, kind, arch in roots if name == "Version"), None)
    if version_chunk is None:
        raise ValueError("Cubase project has no Version chunk; it may be truncated or from an unsupported version")
    app = _app_version(data[version_chunk.start:version_chunk.start + version_chunk.length])
    facts = ProjectFacts(evidence, source_id, METHOD, detail)
    version = app["version"].removeprefix("Version ").strip()
    facts.application(f"{app['name']} {version}".strip(), "Version chunk, PAppVersion object")
    evidence.add(source_id, "cubase.app_version", app, "version", "project", "static",
                 "Version chunk, PAppVersion object", METHOD, "extracted", "high",
                 "Stored application name, version, build date, architecture, text encoding and locale.")
    evidence.add(source_id, "cubase.root_chunks", [{"name": name, "type": kind, "bytes": arch.length}
                 for name, kind, arch in roots], "chunk", "project", "static", "ROOT chunks", METHOD, "extracted",
                 "high", "Stored ROOT/ARCH chunk names, declared types and sizes.")
    evidence.diagnostics.append(Diagnostic("cubase_partial_decoding", "info",
        f"Cubase project files are only partly decoded: the application version and media attributes are read "
        f"structurally, while track, plugin and media names come from byte patterns at low confidence. {GUIDANCE}",
        source_id))
    for step in (_emit_metadata, _emit_patterns):
        try:
            step(data, roots, facts, evidence, source_id)
        except (ValueError, struct.error) as error:
            evidence.diagnostics.append(Diagnostic("cubase_deep_parse_failed", "warning",
                f"{step.__name__.removeprefix('_emit_')} was skipped: {error}", source_id))


def _emit_metadata(data: bytes, roots: list[tuple[str, str, Chunk]], facts: ProjectFacts, evidence: Evidence,
                   source_id: str) -> None:
    chunk = next((arch for name, kind, arch in roots if name == "Metadata"), None)
    if chunk is None:
        return
    attributes = _media_attributes(data[chunk.start:chunk.start + chunk.length])
    medium = GradedFacts(evidence, source_id, METHOD, facts.detail, grade="medium",
                         caveat="Mapped by attribute name from the Metadata chunk.")
    low = GradedFacts(evidence, source_id, METHOD, facts.detail, grade="low",
                      caveat="Mapped by attribute name from the Metadata chunk; the project's tempo and signature tracks "
                      "were not decoded, so changes along the timeline are not reflected.")
    by_id = {str(item["id"]): item["value"] for item in attributes}
    for item in attributes:
        evidence.add(source_id, "cubase.media_attribute", item, "attribute", "project", "static",
                     "Metadata chunk, StMedia attribute", METHOD, "extracted", "high",
                     "Stored media attribute record, parsed from its length-prefixed fields.")
    rate = by_id.get("AudioSampleRate")
    if isinstance(rate, (int, float)) and rate > 0:
        medium.sample_rate(int(rate) if float(rate).is_integer() else rate, "Metadata AudioSampleRate")
    tempo = by_id.get("MusicalTempo")
    if plausible_tempo(tempo):
        low.tempo(round(float(tempo), 4), "Metadata MusicalTempo")
    signature = by_id.get("MusicalSignature")
    if isinstance(signature, dict):
        numerator, denominator = signature.get("Numerator"), signature.get("Denominator")
        if isinstance(numerator, int) and isinstance(denominator, int) and plausible_meter(numerator, denominator):
            low.meter(numerator, denominator, "Metadata MusicalSignature")


def _emit_patterns(data: bytes, roots: list[tuple[str, str, Chunk]], facts: ProjectFacts, evidence: Evidence,
                   source_id: str) -> None:
    low = GradedFacts(evidence, source_id, METHOD, facts.detail, grade="low", caveat=PATTERN_CAVEAT)
    plugin_counts: dict[str, int] = {}
    media: list[str] = []
    tracks: list[tuple[str, str | None]] = []
    track_classes: set[str] = set()
    for name, _, arch in roots:
        if name == "Version" or name == "Metadata":
            continue
        buf = data[arch.start:arch.start + arch.length]
        for plugin in _plugin_names(buf):
            plugin_counts[plugin] = plugin_counts.get(plugin, 0) + 1
        if name.startswith("Arrangement"):
            table = _scan_classes(buf)
            track_classes |= {node.name for node in table.values() if node.name.endswith("TrackEvent")}
            tracks += _tracks(buf, table)
            media += _media_names(buf)
    if track_classes:
        evidence.add(source_id, "cubase.track_classes", sorted(track_classes), "class", "project", "static",
                     "Arrangement chunk class headers", METHOD, "extracted", "medium",
                     "Track class names from node headers whose nested lengths were validated; how many tracks "
                     "use each class is unknown because later instances omit the header.")
    if tracks:
        low.track_count(len(tracks), "Arrangement chunk track headers")
    else:
        evidence.diagnostics.append(Diagnostic("cubase_no_tracks_found", "info",
            "No track headers were recognized in the arrangement; this Cubase version may use another layout.",
            source_id))
    for index, (name, track_class) in enumerate(tracks):
        low.track(index, name, TRACK_CLASS_KINDS.get(track_class or "", "unknown"), f"Arrangement chunk track header {index}")
        evidence.add(source_id, "cubase.track_class", track_class, "class", f"project_track:{index}", "static",
                     f"Arrangement chunk track header {index}", METHOD, "inferred", "low",
                     "Class resolved from the reference stored just before the track header. " + PATTERN_CAVEAT)
    for plugin in sorted(plugin_counts):
        low.plugin(plugin, None, "Plugin Name attribute")
        evidence.add(source_id, "cubase.plugin_mentions", {"name": plugin, "count": plugin_counts[plugin]},
                     "mentions", "project", "static", "Plugin Name attribute", METHOD, "inferred", "low",
                     "Number of times the name is stored; one plugin instance can be stored more than once and "
                     "built-in channel modules are included. " + PATTERN_CAVEAT)
    low.media_references(media, "Arrangement chunk media file names")
