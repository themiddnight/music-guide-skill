"""Binary AAF (Advanced Authoring Format) composition metadata read through pyaaf2, always read-only."""

from __future__ import annotations

import gc
import json
import logging
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterator

import aaf2
from aaf2 import components, mobs

from ..model import Detail, Diagnostic, Evidence
from ..project import ProjectFacts, TrackKind

COMPOUND_FILE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
MAX_MOBS = 20_000
MAX_SLOTS = 4_096
MAX_COMPONENTS = 200_000
MAX_DEPTH = 64
MAX_LOG_MESSAGES = 10
# pyaaf2 sizes buffers from counts inside the file, so a corrupt index can demand gigabytes. It therefore
# runs in a child process that the parent stops when it exceeds a wall-clock or memory limit.
CHILD_TIMEOUT_SECONDS = 60.0
CHILD_MEMORY_BYTES = 1536 * 1024 * 1024
CHILD_OUTPUT_BYTES = 32 * 1024 * 1024
POLL_SECONDS = 0.1
SCRIPTS_DIR = Path(__file__).resolve().parents[2]
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
        track_index = 0
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
                if isinstance(slot, aaf2.mobslots.EventMobSlot):
                    _markers(facts, segment, rate, locator)
                    continue
                kind = _kind(segment)
                if not isinstance(slot, aaf2.mobslots.TimelineMobSlot) or kind not in TRACK_KINDS or not rate:
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
        facts.media_references(references, "SourceMob/EssenceDescription/Locator/URLString")
        _sample_rate(evidence, source_id, facts, sample_rates)
        if not top_level:
            evidence.diagnostics.append(Diagnostic(
                "aaf_no_top_level_composition", "warning",
                "The file has no top-level composition mob, so no tracks, clips or markers were read.", source_id))


def _markers(facts: ProjectFacts, segment: Any, rate: float | None, locator: str) -> None:
    if not rate:
        return
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


CHILD_CODE = (
    "import sys; sys.path.insert(0, sys.argv[1]); "
    "from music_guide.adapters.aaf import child_main; raise SystemExit(child_main(sys.argv[2:]))")


def child_main(argv: list[str]) -> int:
    path, source_id, detail = Path(argv[0]), argv[1], argv[2]
    staged = Evidence()
    try:
        _parse(path, source_id, staged, detail)  # type: ignore[arg-type]
        reply = {"ok": True, **staged.as_json()}
    except ValueError as error:
        reply = {"ok": False, "error": str(error)}
    sys.stdout.write(json.dumps(reply, allow_nan=False))
    return 0


def _resident_bytes(pid: int) -> int:
    try:
        output = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=5)
        return int(output.stdout.strip() or 0) * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def _run_child(path: Path, source_id: str, detail: Detail) -> dict[str, Any]:
    command = [sys.executable, "-I", "-c", CHILD_CODE, str(SCRIPTS_DIR), str(path), source_id, detail]
    started = time.monotonic()
    with subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as child:
        while True:
            try:
                stdout, stderr = child.communicate(timeout=POLL_SECONDS)
                break
            except subprocess.TimeoutExpired:
                reason = None
                if time.monotonic() - started > CHILD_TIMEOUT_SECONDS:
                    reason = f"took longer than {CHILD_TIMEOUT_SECONDS:g} seconds"
                elif _resident_bytes(child.pid) > CHILD_MEMORY_BYTES:
                    reason = f"needed more than {CHILD_MEMORY_BYTES // (1024 * 1024)} MiB of memory"
                if reason:
                    child.kill()
                    child.communicate()
                    raise ValueError(f"AAF parser stopped: reading the file {reason}; the file is corrupt or unsupported")
    if child.returncode != 0 or len(stdout) > CHILD_OUTPUT_BYTES:
        tail = stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["no message"]
        raise ValueError(f"AAF parser exited abnormally (status {child.returncode}): {tail[0][:200]}")
    try:
        reply = json.loads(stdout)
    except ValueError as error:
        raise ValueError("AAF parser returned unreadable output") from error
    if not isinstance(reply, dict):
        raise ValueError("AAF parser returned unreadable output")
    return reply


def aaf(path: Path, source_id: str, evidence: Evidence, detail: Detail) -> None:
    reply = _run_child(path, source_id, detail)
    if not reply.get("ok"):
        raise ValueError(str(reply.get("error", "unknown AAF failure"))[:500])
    for item in reply["observations"]:
        evidence.add(source_id, item["property"], item["value"], item["unit"], item["scope"], item["timeDomain"],
                     item["locator"], item["method"], item["provenance"], item["confidence"],
                     item["confidenceReason"])
    for item in reply["diagnostics"]:
        evidence.diagnostics.append(Diagnostic(item["code"], item["severity"], item["message"], source_id))
