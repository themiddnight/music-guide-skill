from __future__ import annotations
import math
import re
import zipfile
import zlib
from pathlib import Path
from urllib.parse import unquote
from xml.etree.ElementTree import Element

from defusedxml import ElementTree as DefusedET
from defusedxml.common import DefusedXmlException

MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_MIDI_BYTES = 32 * 1024 * 1024
MAX_XML_BYTES = 8 * 1024 * 1024
MAX_ZIP_ENTRIES = 256
MAX_ZIP_BYTES = 64 * 1024 * 1024
MAX_AUDIO_BYTES = 4 * 1024 * 1024 * 1024


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


def _read_zip_member(path: Path, member: str, limit: int, *, max_entries: int = MAX_ZIP_ENTRIES,
                     max_total: int = MAX_ZIP_BYTES) -> bytes:
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > max_entries or sum(item.file_size for item in entries) > max_total:
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
        return DefusedET.fromstring(raw, forbid_dtd=True, forbid_entities=True, forbid_external=True)
    except (DefusedET.ParseError, DefusedXmlException) as error:
        raise ValueError(f"unsafe or malformed XML: {error}") from error


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
