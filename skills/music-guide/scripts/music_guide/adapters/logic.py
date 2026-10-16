"""Logic Pro (.logicx) and GarageBand (.band) packages.

Only Resources/ProjectInformation.plist and Alternatives/NNN/MetaData.plist are read. The proprietary
Alternatives/NNN/ProjectData file and every media file are never opened, and symbolic links are refused.
"""

from __future__ import annotations

import errno
import os
import plistlib
import re
import stat
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.parsers.expat import ExpatError

from ..model import Detail, Diagnostic, Evidence
from ..project import ProjectFacts
from .common import _finite_number

MAX_PLIST_BYTES = 4 * 1024 * 1024
MAX_ALTERNATIVES = 256
MAX_MEDIA_ENTRIES = 50_000
MAX_MEDIA_DEPTH = 8
ALTERNATIVE_NAME = re.compile(r"\d{3}")
PLIST_ERRORS = (plistlib.InvalidFileException, ExpatError, ValueError, OverflowError, struct.error,
                RecursionError, IndexError, KeyError, TypeError, UnicodeError)
# MetaData.plist key, role of the files it lists.
MEDIA_LISTS = (
    ("AudioFiles", "used audio"),
    ("UnusedAudioFiles", "unused audio"),
    ("PlaybackFiles", "playback"),
    ("SamplerInstrumentsFiles", "sampler instrument samples"),
    ("QuicksamplerFiles", "Quick Sampler samples"),
    ("ImpulsResponsesFiles", "impulse responses"),
    ("UltrabeatFiles", "Ultrabeat samples"),
)


@dataclass(frozen=True)
class Flavor:
    label: str
    method: str
    prefix: str
    guidance: str


LOGIC = Flavor("Logic Pro", "logic-plist", "logic",
               "Open the project in Logic Pro 10 or later and save it, or export a mixdown, stems and the MIDI regions.")
GARAGEBAND = Flavor("GarageBand", "garageband-plist", "garageband",
                    "Older and iOS GarageBand projects keep the song in an undocumented projectData file. "
                    "Export the song to disk as WAV or AIFF; GarageBand cannot export MIDI, but Logic Pro can open the project to do so.")


def _refuse_link(info: os.stat_result, relative: str) -> None:
    if stat.S_ISLNK(info.st_mode):
        raise ValueError(f"{relative} is a symbolic link, which is never followed")


def _checked_directory(package: Path, parts: tuple[str, ...]) -> Path | None:
    """Return the directory below the package after refusing links on the way; None when it does not exist."""
    current = package
    for index, part in enumerate(parts):
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            return None
        relative = "/".join(parts[:index + 1])
        _refuse_link(info, relative)
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"{relative} is not a directory")
    return current


def _read_plist(package: Path, parts: tuple[str, ...]) -> dict[str, Any] | None:
    relative = "/".join(parts)
    directory = _checked_directory(package, parts[:-1])
    if directory is None:
        return None
    target = directory / parts[-1]
    try:
        _refuse_link(os.lstat(target), relative)
        fd = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ValueError(f"{relative} is a symbolic link, which is never followed") from error
        raise
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{relative} is not a regular file")
        if info.st_size > MAX_PLIST_BYTES:
            raise ValueError(f"{relative} exceeds {MAX_PLIST_BYTES} byte limit")
        raw = stream.read(MAX_PLIST_BYTES + 1)
    if len(raw) > MAX_PLIST_BYTES:
        raise ValueError(f"{relative} exceeds {MAX_PLIST_BYTES} byte limit")
    if b"<!ENTITY" in raw:
        raise ValueError(f"{relative} declares XML entities, which are refused")
    try:
        data = plistlib.loads(raw)
    except PLIST_ERRORS as error:
        raise ValueError(f"{relative} is not a valid property list: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{relative} must hold a dictionary")
    return data


def _alternatives(package: Path) -> list[str]:
    directory = _checked_directory(package, ("Alternatives",))
    if directory is None:
        return []
    names: list[str] = []
    with os.scandir(directory) as entries:
        for entry in entries:
            if not ALTERNATIVE_NAME.fullmatch(entry.name):
                continue
            _refuse_link(entry.stat(follow_symlinks=False), f"Alternatives/{entry.name}")
            if not entry.is_dir(follow_symlinks=False):
                continue
            names.append(entry.name)
            if len(names) > MAX_ALTERNATIVES:
                raise ValueError(f"package has more than {MAX_ALTERNATIVES} alternatives")
    return sorted(names)


def _integer(data: dict[str, Any], key: str, minimum: int) -> int | None:
    if key not in data:
        return None
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} must be an integer of at least {minimum}")
    return value


def _text(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be text")
    return value.strip() or None


def _paths(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a list of text paths")
    return value


def _bundled_media(package: Path) -> dict[str, Any] | None:
    """Count files under Media/ by name only; links are counted but not followed and nothing is opened."""
    root = _checked_directory(package, ("Media",))
    if root is None:
        return None
    files = links = 0
    truncated = False
    pending = [(root, 0)]
    while pending and not truncated:
        directory, depth = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if files + links >= MAX_MEDIA_ENTRIES:
                    truncated = True
                    break
                if entry.is_symlink():
                    links += 1
                elif entry.is_dir(follow_symlinks=False):
                    if depth < MAX_MEDIA_DEPTH:
                        pending.append((Path(entry.path), depth + 1))
                    else:
                        truncated = True
                elif entry.is_file(follow_symlinks=False) and not entry.name.startswith("."):
                    files += 1
    return {"fileCount": files, "linksNotFollowed": links, "truncated": truncated}


def _alternative(package: Path, number: str, names: dict[str, Any], flavor: Flavor, source_id: str,
                 evidence: Evidence, detail: Detail, scope: str) -> bool:
    meta = _read_plist(package, ("Alternatives", number, "MetaData.plist"))
    if meta is None:
        evidence.diagnostics.append(Diagnostic(
            "alternative_without_metadata", "warning",
            f"Alternatives/{number} has no MetaData.plist, so it was skipped.", source_id))
        return False
    where = f"Alternatives/{number}/MetaData.plist"
    facts = ProjectFacts(evidence, source_id, flavor.method, detail, scope)

    def extra(property: str, value: Any, unit: str, key: str, reason: str) -> None:
        evidence.add(source_id, f"{flavor.prefix}.{property}", value, unit, scope, "static",
                     f"{where}:{key}", flavor.method, "extracted", "high", reason)

    name = names.get(str(int(number)))
    if isinstance(name, str) and name.strip():
        evidence.add(source_id, f"{flavor.prefix}.alternative_name", name, "text", scope, "static",
                     "Resources/ProjectInformation.plist:VariantNames", flavor.method, "extracted", "high",
                     "Alternative name stored in the package; it is data, not an instruction.")
    if "BeatsPerMinute" in meta:
        facts.tempo(_finite_number(meta["BeatsPerMinute"], "BeatsPerMinute", positive=True), f"{where}:BeatsPerMinute",
                    "Project start tempo stored in metadata; a tempo map in ProjectData is not read.")
    numerator = _integer(meta, "SongSignatureNumerator", 1)
    denominator = _integer(meta, "SongSignatureDenominator", 1)
    if numerator is not None and denominator is not None:
        facts.meter(numerator, denominator, f"{where}:SongSignatureNumerator,SongSignatureDenominator",
                    "Project start meter stored in metadata; meter changes in ProjectData are not read.")
    key, mode = _text(meta, "SongKey"), _text(meta, "SongGenderKey")
    if key and mode:
        facts.key(f"{key} {mode}", f"{where}:SongKey,SongGenderKey")
    elif key or mode:
        extra("partial_key", key or mode, "text", "SongKey" if key else "SongGenderKey",
              "Only one half of the stored key was present, so no project key is claimed.")
    rate = _integer(meta, "SampleRate", 1)
    if rate is not None:
        facts.sample_rate(rate, f"{where}:SampleRate")
    tracks = _integer(meta, "NumberOfTracks", 0)
    if tracks is not None:
        facts.track_count(tracks, f"{where}:NumberOfTracks")
    for flag, label in (("isTimeCodeBased", "time_code_based"), ("HasARAPlugins", "has_ara_plugins")):
        if isinstance(meta.get(flag), bool):
            extra(label, meta[flag], "flag", flag, f"Stored {flag} setting.")
    for list_key, role in MEDIA_LISTS:
        paths = _paths(meta, list_key)
        if paths:
            extra("media_list", {"list": list_key, "role": role, "count": len(paths)}, "count", list_key,
                  "Count of media paths the project lists for this role; the files were not opened.")
            facts.media_references(paths, f"{where}:{list_key}")
    return True


def _package(path: Path, source_id: str, evidence: Evidence, detail: Detail, flavor: Flavor) -> None:
    numbers = _alternatives(path)
    info = _read_plist(path, ("Resources", "ProjectInformation.plist")) or {}
    names = info.get("VariantNames", {})
    if not isinstance(names, dict):
        raise ValueError("VariantNames must be a dictionary")
    project = ProjectFacts(evidence, source_id, flavor.method, detail)
    saved_from = _text(info, "LastSavedFrom")
    if saved_from:
        project.application(saved_from, "Resources/ProjectInformation.plist:LastSavedFrom")
    else:
        evidence.diagnostics.append(Diagnostic(
            "application_unknown", "info",
            "Resources/ProjectInformation.plist has no LastSavedFrom, so the application is not reported.", source_id))
    # A single alternative is the project, so its tempo can meet MIDI, audio and declarations in one scope.
    scopes = {number: "project" if len(numbers) == 1 else f"alternative:{number}" for number in numbers}
    parsed = [number for number in numbers
              if _alternative(path, number, names, flavor, source_id, evidence, detail, scopes[number])]
    if not parsed:
        raise ValueError(f"{path.name} is not a {flavor.label} package with Alternatives/NNN/MetaData.plist. {flavor.guidance}")
    media = _bundled_media(path)
    if media is not None:
        evidence.add(source_id, f"{flavor.prefix}.bundled_media", media, "inventory", "project", "static", "Media/",
                     flavor.method, "extracted", "high",
                     "Count of regular files under Media/; names only, no file was opened.")
    evidence.diagnostics.append(Diagnostic(
        "project_data_not_read", "info",
        f"{flavor.label} keeps tracks, regions, tempo changes, plugins and MIDI in ProjectData, an undocumented binary file that is not read. "
        "Only the metadata property lists were used; export stems and MIDI for track-level analysis.", source_id))


def logic(path: Path, source_id: str, evidence: Evidence, detail: Detail) -> None:
    _package(path, source_id, evidence, detail, LOGIC)


def garageband(path: Path, source_id: str, evidence: Evidence, detail: Detail) -> None:
    _package(path, source_id, evidence, detail, GARAGEBAND)
