"""pyaaf2 reading, run only inside the isolated child process that aaf.py starts.

The module imports pyaaf2 itself, so the parent process needs pyaaf2 only when an AAF file is analyzed.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Iterator

try:
    import aaf2
    from aaf2 import components, mobs, mobslots
except ImportError as error:  # reported by main() as a source error with install guidance
    aaf2 = components = mobs = mobslots = None
    IMPORT_ERROR: ImportError | None = error
else:
    IMPORT_ERROR = None

from ..model import Detail, Diagnostic, Evidence
from ..project import ProjectFacts, TrackKind
from .aaf import DEADLINE_EXIT, MEMORY_EXIT

COMPOUND_FILE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
MAX_MOBS = 20_000
MAX_SLOTS = 4_096
MAX_COMPONENTS = 200_000
MAX_DEPTH = 64
MAX_LOG_MESSAGES = 10
MIN_SAMPLE_RATE = 8_000
MAX_TEXT = 200
CONTAINER_PROPERTIES = ("Components", "Slots", "InputSegments", "Alternates", "Choices")
SOUND_DESCRIPTORS = ("WAVEDescriptor", "AIFCDescriptor", "PCMDescriptor", "SoundDescriptor")
METHOD = "pyaaf2"
TRACK_KINDS: dict[str, TrackKind] = {"sound": "audio", "picture": "unknown"}


class _LogCapture(logging.Handler):
    """pyaaf2 reports container inconsistencies through the root logger; keep them as evidence diagnostics."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message not in self.messages and len(self.messages) < MAX_LOG_MESSAGES:
            self.messages.append(message)


def _open_read_only(path: Path) -> Any:
    with warnings.catch_warnings():
        # pyaaf2 leaves its file handle open when the header is invalid. Dropping the exception's
        # traceback frees that handle here, inside this scope, instead of at an arbitrary later collection.
        warnings.simplefilter("ignore", ResourceWarning)
        try:
            return aaf2.open(str(path), "r")
        except Exception as error:
            problem = f"{type(error).__name__}: {error}"
            error.__traceback__ = None
        gc.collect()
    raise ValueError(f"cannot read AAF: {problem}")


def _text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text[:MAX_TEXT] if text else None


def _rate(value: Any) -> float | None:
    try:
        rate = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 and rate != float("inf") else None


def _kind(segment: Any) -> str:
    return str(getattr(segment, "media_kind", None) or "").lower().removeprefix("legacy")


def _segments(segment: Any) -> Iterator[tuple[Any, int]]:
    """Walk a segment tree without recursion, bounded in depth and size."""
    stack = [(segment, 0)]
    seen = 0
    while stack:
        item, depth = stack.pop()
        if item is None:
            continue
        seen += 1
        if seen > MAX_COMPONENTS or depth > MAX_DEPTH:
            raise ValueError("AAF composition is nested or large beyond the supported limits")
        yield item, depth
        for name in CONTAINER_PROPERTIES:
            children = item.getvalue(name) if hasattr(item, "getvalue") else None
            if children:
                stack.extend((child, depth + 1) for child in reversed(list(children)))
        selected = item.getvalue("Selected") if hasattr(item, "getvalue") else None
        if selected is not None:
            stack.append((selected, depth + 1))


def _length(segment: Any) -> int | None:
    if isinstance(segment, components.Sequence):
        total = 0
        for item in segment.components:
            length = item.getvalue("Length") or 0
            total += -length if isinstance(item, components.Transition) else length
        return total
    return segment.getvalue("Length")


def _source_clip_count(segment: Any) -> int:
    count = 0
    for item, _ in _segments(segment):
        if isinstance(item, components.SourceClip):
            source = item.getvalue("SourceID")
            if source is not None and source.int != 0:
                count += 1
    return count


def _locators(descriptor: Any, depth: int = 0) -> Iterator[str]:
    if descriptor is None or depth > 4:
        return
    for locator in descriptor.getvalue("Locator") or ():
        url = locator.getvalue("URLString")
        if url:
            yield str(url)
    for inner in descriptor.getvalue("FileDescriptors") or ():
        yield from _locators(inner, depth + 1)


def _sound_rates(descriptor: Any, depth: int = 0) -> Iterator[float]:
    if descriptor is None or depth > 4:
        return
    if type(descriptor).__name__ in SOUND_DESCRIPTORS:
        rate = _rate(descriptor.getvalue("AudioSamplingRate") or descriptor.getvalue("SampleRate"))
        if rate:
            yield rate
    for inner in descriptor.getvalue("FileDescriptors") or ():
        yield from _sound_rates(inner, depth + 1)


def _number(value: float) -> float | int:
    return int(value) if float(value).is_integer() else round(value, 6)


def _parse(path: Path, source_id: str, evidence: Evidence, detail: Detail) -> None:
    """Read the file in this process. The CLI reaches it through aaf(), which isolates it."""
    with path.open("rb") as stream:
        head = stream.read(len(COMPOUND_FILE))
    if head.lstrip().startswith(b"<"):
        raise ValueError("AAF XML encoding is not supported; export a binary (structured storage) AAF")
    if head != COMPOUND_FILE:
        raise ValueError("file is not a binary AAF (compound file) container")
    capture = _LogCapture()
    root_logger = logging.getLogger()
    root_logger.addHandler(capture)
    try:
        _read(path, source_id, evidence, detail)
    except ValueError:
        raise
    except Exception as error:
        # pyaaf2 signals corruption with AssertionError, struct.error, KeyError, IndexError and more.
        raise ValueError(f"cannot read AAF: {type(error).__name__}: {error}") from error
    finally:
        root_logger.removeHandler(capture)
    for message in capture.messages:
        evidence.diagnostics.append(Diagnostic("aaf_container_warning", "warning",
                                               f"pyaaf2 reported: {message}", source_id))


def _read(path: Path, source_id: str, evidence: Evidence, detail: Detail) -> None:
    facts = ProjectFacts(evidence, source_id, METHOD, detail)
    handle = _open_read_only(path)
    with handle as f:
        identifications = f.header.getvalue("IdentificationList") or []
        if identifications:
            latest = identifications[-1]
            product = _text(latest.getvalue("ProductName"))
            version = _text(latest.getvalue("ProductVersionString"))
            if product:
                facts.application(f"{product} {version}".strip() if version else product,
                                  "Header/IdentificationList[last]/ProductName")
            company = _text(latest.getvalue("CompanyName"))
            if company:
                evidence.add(source_id, "aaf.company", company, "text", "project", "static",
                             "Header/IdentificationList[last]/CompanyName", METHOD, "extracted", "high",
                             "Company named by the application that last wrote the file.")
        all_mobs = list(f.content.mobs)
        if len(all_mobs) > MAX_MOBS:
            raise ValueError(f"AAF holds more than {MAX_MOBS} mobs")
        counts = {"composition": 0, "master": 0, "source": 0}
        sample_rates: set[float] = set()
        references: list[str] = []
        compositions = []
        for mob in all_mobs:
            if isinstance(mob, mobs.CompositionMob):
                counts["composition"] += 1
                compositions.append(mob)
            elif isinstance(mob, mobs.MasterMob):
                counts["master"] += 1
            elif isinstance(mob, mobs.SourceMob):
                counts["source"] += 1
                descriptor = mob.descriptor
                references.extend(_locators(descriptor))
                sample_rates.update(_sound_rates(descriptor))
        evidence.add(source_id, "aaf.mob_counts", counts, "count", "project", "static", "Header/Content/Mobs",
                     METHOD, "extracted", "high", "Count of stored mobs by kind.")
        top_level = [mob for mob in compositions if mob.usage == "Usage_TopLevel"]
        track_index = unrated = 0
        for position, mob in enumerate(top_level):
            slots = list(mob.slots)
            if len(slots) > MAX_SLOTS:
                raise ValueError(f"AAF composition holds more than {MAX_SLOTS} slots")
            evidence.add(source_id, "aaf.composition", {"name": _text(mob.name), "usage": mob.usage,
                                                        "slotCount": len(slots)}, "composition", "project",
                         "static", f"CompositionMob[{position}]", METHOD, "extracted", "high",
                         "Top-level composition mob stored in the file.")
            for slot in slots:
                locator = f"CompositionMob[{position}]/Slots[{slot.slot_id}]"
                segment = slot.segment
                rate = _rate(slot.getvalue("EditRate"))
                if isinstance(slot, mobslots.EventMobSlot):
                    if rate:
                        _markers(facts, segment, rate, locator)
                    else:
                        unrated += 1
                    continue
                kind = _kind(segment)
                if not isinstance(slot, mobslots.TimelineMobSlot) or kind not in TRACK_KINDS:
                    continue
                if not rate:
                    unrated += 1
                    continue
                if kind == "sound" and rate >= MIN_SAMPLE_RATE:
                    sample_rates.add(rate)
                facts.track(track_index, _text(slot.name), TRACK_KINDS[kind], locator)
                length = _length(segment)
                scope = f"project_track:{track_index}"
                evidence.add(source_id, "aaf.slot", {
                    "slotId": slot.slot_id, "composition": _text(mob.name), "mediaKind": kind,
                    "editRate": _number(rate), "lengthEdits": length,
                    "lengthSeconds": None if length is None else _number(length / rate)},
                    "slot", scope, "static", locator, METHOD, "extracted", "high",
                    "Timeline slot as stored; edit units depend on the slot's edit rate.")
                facts.clip_count(_source_clip_count(segment), locator, track_index)
                track_index += 1
        if top_level:
            facts.track_count(track_index, "CompositionMob/Slots")
        if unrated:
            evidence.diagnostics.append(Diagnostic(
                "aaf_slots_without_edit_rate", "warning",
                f"{unrated} slot(s) have a missing or zero edit rate, so their time cannot be converted and they were skipped.",
                source_id))
        facts.media_references(references, "SourceMob/EssenceDescription/Locator/URLString")
        _sample_rate(evidence, source_id, facts, sample_rates)
        if not top_level:
            evidence.diagnostics.append(Diagnostic(
                "aaf_no_top_level_composition", "warning",
                "The file has no top-level composition mob, so no tracks, clips or markers were read.", source_id))


def _markers(facts: ProjectFacts, segment: Any, rate: float, locator: str) -> None:
    for item, _ in _segments(segment):
        if isinstance(item, components.CommentMarker):
            position = item.getvalue("Position")
            if position is None:
                continue
            facts.marker(_text(item.getvalue("Comment")), position / rate, "seconds", locator)


def _sample_rate(evidence: Evidence, source_id: str, facts: ProjectFacts, rates: set[float]) -> None:
    if len(rates) == 1:
        facts.sample_rate(_number(next(iter(rates))), "Slot EditRate (sound) and SoundDescriptor/SampleRate")
    elif rates:
        evidence.add(source_id, "aaf.sample_rates", sorted(_number(rate) for rate in rates), "Hz", "project",
                     "static", "Slot EditRate (sound) and SoundDescriptor/SampleRate", METHOD, "extracted", "medium",
                     "Several distinct audio rates are stored, so no single project sample rate is reported.")


def _apply_limits(config: dict[str, Any]) -> str:
    """Bound this process by itself, so it ends even if the parent is interrupted. Returns the memory guard in force."""
    guard = "none"
    deadline = float(config["deadline"])
    try:
        import signal
        signal.alarm(max(1, int(deadline) + 1))  # the default SIGALRM action ends the process even inside a long C call
    except (ImportError, AttributeError, ValueError):
        import threading
        timer = threading.Timer(deadline, lambda: os._exit(DEADLINE_EXIT))
        timer.daemon = True
        timer.start()
    if sys.platform.startswith("linux"):
        try:
            import resource
            limit = int(config["memory"])
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            guard = "address-space"
        except (ImportError, ValueError, OSError):
            pass
    return guard


def main(config: dict[str, Any]) -> int:
    guard = _apply_limits(config)
    if IMPORT_ERROR is not None:
        reply: dict[str, Any] = {"ok": False, "error": (
            f"pyaaf2 could not be imported by {sys.executable} ({type(IMPORT_ERROR).__name__}: {IMPORT_ERROR}); "
            "install the packages in requirements.txt into the environment that runs analyze.py")}
    else:
        staged = Evidence()
        try:
            _parse(Path(config["path"]), config["sourceId"], staged, config["detail"])
            reply = {"ok": True, **staged.as_json()}
        except ValueError as error:
            reply = {"ok": False, "error": str(error)}
        except MemoryError:
            return MEMORY_EXIT
    reply["memoryGuard"] = guard
    sys.stdout.write(json.dumps(reply, allow_nan=False))
    return 0
