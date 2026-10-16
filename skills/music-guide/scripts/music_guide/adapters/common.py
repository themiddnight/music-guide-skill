from __future__ import annotations
import io
import json
import math
import re
import struct
import zipfile
import zlib
from pathlib import Path
from urllib.parse import unquote
from xml.etree.ElementTree import Element

from defusedxml import ElementTree as DefusedET
from defusedxml.common import DefusedXmlException

MAX_JSON_BYTES = 8 * 1024 * 1024
# mido holds every message as an object (about 130 bytes of RAM per file byte), so the bound is tight.
MAX_MIDI_BYTES = 4 * 1024 * 1024
MAX_XML_BYTES = 8 * 1024 * 1024
# Embedded audio is common in project exports, so the entry cap is generous. Only the member that is read is bounded.
MAX_ZIP_ENTRIES = 16384
MAX_CENTRAL_DIRECTORY_BYTES = 8 * 1024 * 1024
MAX_ZIP_BYTES = 64 * 1024 * 1024
MAX_AUDIO_BYTES = 4 * 1024 * 1024 * 1024
# A tree costs memory per element however small the file is, so the shape is bounded as well as the bytes.
MAX_XML_ELEMENTS = 2_000_000
MAX_XML_DEPTH = 256


def _read_bounded(path: Path, limit: int) -> bytes:
    if path.stat().st_size > limit:
        raise ValueError(f"file exceeds {limit} byte limit")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"file exceeds {limit} byte limit")
    return data


def _finite_number(value: object, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} exceeds finite numeric range") from error
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{label} must be a finite positive number" if positive else f"{label} must be finite")
    return number


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"nonfinite JSON value: {value}")


def _load_json(raw: bytes) -> object:
    """Parses JSON that is data: a repeated key or a NaN/Infinity literal is an error, never a silent last-wins."""
    return json.loads(raw, object_pairs_hook=_no_duplicate_keys, parse_constant=_reject_constant)


MAX_GZIP_BYTES = 64 * 1024 * 1024


def _gunzip_bounded(raw: bytes, limit: int) -> bytes:
    decompressor = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
    try:
        data = decompressor.decompress(raw, limit + 1)
    except zlib.error as error:
        raise ValueError(f"invalid gzip data: {error}") from error
    if len(data) > limit or decompressor.unconsumed_tail:
        raise ValueError(f"gzip data expands beyond {limit} byte limit")
    if not decompressor.eof:
        raise ValueError("gzip data is truncated")
    return data


def _central_directory_size(path: Path) -> int | None:
    """Size of the central directory from the end record. zipfile builds one object per 46 bytes of it, so this
    is the number to bound before the archive is opened; the entry count in the same record can be spoofed."""
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - 65557))
        tail = stream.read()
        index = tail.rfind(b"PK\x05\x06")
        if index < 0 or len(tail) < index + 22:
            return None
        directory_size = struct.unpack_from("<I", tail, index + 12)[0]
        if directory_size != 0xFFFFFFFF or index < 20 or tail[index - 20:index - 16] != b"PK\x06\x07":
            return directory_size
        stream.seek(struct.unpack_from("<Q", tail, index - 12)[0])
        record = stream.read(48)
    if len(record) < 48 or record[:4] != b"PK\x06\x06":
        return None
    return struct.unpack_from("<Q", record, 40)[0]


def _read_zip_member(path: Path, member: str, limit: int, *, max_entries: int = MAX_ZIP_ENTRIES,
                     max_total: int | None = None) -> bytes:
    """Reads one bounded member. Only that member's size is limited; `max_total` optionally bounds all members."""
    directory_size = _central_directory_size(path)
    if directory_size is not None and directory_size > MAX_CENTRAL_DIRECTORY_BYTES:
        raise ValueError(f"archive central directory is {directory_size} bytes, more than the {MAX_CENTRAL_DIRECTORY_BYTES} limit")
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > max_entries:
                raise ValueError(f"archive holds {len(entries)} entries, more than the {max_entries} limit")
            if max_total is not None and sum(item.file_size for item in entries) > max_total:
                raise ValueError("archive exceeds metadata safety limits")
            matches = [item for item in entries if item.filename == member]
            if len(matches) != 1 or matches[0].file_size > limit:
                raise ValueError(f"archive must contain one bounded {member}")
            if matches[0].compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                raise ValueError(f"{member} compression must be stored or deflated")
            if matches[0].flag_bits & 0x1:
                raise ValueError(f"{member} is encrypted")
            with archive.open(matches[0]) as stream:
                raw = stream.read(limit + 1)
    except (zipfile.BadZipFile, RuntimeError, zlib.error, EOFError) as error:
        raise ValueError(f"cannot read {member}: {error}") from error
    # The central directory size is untrusted, so the bound applies to the bytes actually read.
    if len(raw) > limit:
        raise ValueError(f"{member} exceeds {limit} byte limit")
    return raw


def _parse_xml(raw: bytes) -> Element:
    try:
        events = DefusedET.iterparse(io.BytesIO(raw), events=("start", "end"), forbid_dtd=True, forbid_entities=True,
                                     forbid_external=True)
        root: Element | None = None
        depth = count = 0
        for event, element in events:
            if event == "end":
                depth -= 1
                continue
            root = element if root is None else root
            depth += 1
            count += 1
            if depth > MAX_XML_DEPTH:
                raise ValueError(f"XML is nested deeper than {MAX_XML_DEPTH} levels")
            if count > MAX_XML_ELEMENTS:
                raise ValueError(f"XML has more than {MAX_XML_ELEMENTS} elements")
    except (DefusedET.ParseError, DefusedXmlException, LookupError) as error:
        raise ValueError(f"unsafe or malformed XML: {error}") from error
    if root is None:
        raise ValueError("unsafe or malformed XML: no root element")
    return root


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(parent: Element, name: str) -> Element | None:
    return next((item for item in parent if _local(item.tag) == name), None)


def _media_basename(raw_path: str) -> str:
    # Embedded media paths are data: keep the final component and never resolve or open it.
    text = raw_path.strip()
    if text.lower().startswith("file:"):
        text = unquote(text[5:])
    return re.split(r"[\\/]", text.rstrip("\\/"))[-1].strip()
