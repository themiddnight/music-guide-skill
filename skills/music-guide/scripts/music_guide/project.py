"""Shared project vocabulary so every DAW adapter emits the same properties."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence, get_args

from .adapters.common import _finite_number, _media_basename
from .adapters.listing import MAX_LISTED, MAX_NOTE_LISTING, limit_reached
from .model import Confidence, Detail, Diagnostic, Evidence
from .notes import Note, summarize_notes

TrackKind = Literal["audio", "midi", "instrument", "group", "return", "master", "folder", "unknown"]
TRACK_KINDS = frozenset(get_args(TrackKind))
PositionUnit = Literal["beats", "seconds"]
TIME_DOMAINS: dict[str, str] = {"beats": "project_beats", "seconds": "project_seconds"}


# Properties that repeat once per item of the file. A hostile file could list them without end, so each gets a cap.
LISTED_PROPERTIES = {"project.track": "tracks", "project.plugin": "plugins", "project.marker": "markers",
                     "project.media_reference": "media references", "project.note": "notes"}


def _list_limit(property: str) -> int:
    return MAX_NOTE_LISTING if property == "project.note" else MAX_LISTED


MAX_TEMPO_BPM = 1000.0
MAX_METER_NUMERATOR = 999
MAX_METER_DENOMINATOR = 128
SHOWN_CHARS = 40


def _whole_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return int(value) if isinstance(value, float) and value.is_integer() else None


def plausible_tempo(bpm: object) -> bool:
    """True for a tempo above 0 and up to 1000 BPM. NaN and infinity fail the comparison."""
    return isinstance(bpm, (int, float)) and not isinstance(bpm, bool) and 0 < bpm <= MAX_TEMPO_BPM


def plausible_meter(numerator: object, denominator: object) -> bool:
    """True for a whole numerator in 1..999 over a whole denominator in 1..128. Odd ones such as 4/3 are real music."""
    top, bottom = _whole_value(numerator), _whole_value(denominator)
    return (top is not None and bottom is not None and 1 <= top <= MAX_METER_NUMERATOR
            and 1 <= bottom <= MAX_METER_DENOMINATOR)


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _shown(value: object) -> str:
    text = repr(value)
    return text if len(text) <= SHOWN_CHARS else f"{text[:SHOWN_CHARS]}..."


def _count(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _track_scope(index: int) -> str:
    return f"project_track:{_count(index, 'track index')}"


@dataclass(frozen=True)
class ProjectFacts:
    """Records project-level evidence for one source. `method` names the parser, e.g. "ableton-als-xml"."""

    evidence: Evidence
    source_id: str
    method: str
    detail: Detail
    # Scope of facts that are not about one track, e.g. "alternative:000" for a package that stores several arrangements.
    scope: str = "project"

    def _add(self, property: str, value: Any, unit: str, locator: str, reason: str, *,
             scope: str | None = None, time_domain: str = "static", confidence: Confidence = "high") -> None:
        if property in LISTED_PROPERTIES:
            listed = self.evidence.listed
            key = (self.source_id, property)
            listed[key] += 1
            limit = _list_limit(property)
            if listed[key] > limit:
                if listed[key] == limit + 1:
                    self.evidence.diagnostics.append(limit_reached(LISTED_PROPERTIES[property], limit, self.source_id))
                return
        self.evidence.add(self.source_id, property, value, unit, self.scope if scope is None else scope, time_domain, locator,
                          self.method, "extracted", confidence, reason)

    def application(self, name: str, locator: str) -> None:
        self._add("project.application", name, "text", locator, "Application name stored in the project file.")

    def _implausible(self, what: str, locator: str, limit: str) -> None:
        self.evidence.diagnostics.append(Diagnostic(
            "implausible_value", "warning",
            f"Skipped {what} at {locator}: it is outside {limit}, so the rest of the project is still reported.",
            self.source_id))

    def skip_meter(self, numerator: object, denominator: object, locator: str) -> None:
        """Warn that a meter was skipped as implausible. Adapters that collect meters themselves call it for each one."""
        self._implausible(f"meter {_shown(numerator)}/{_shown(denominator)}", locator,
                          f"1-{MAX_METER_NUMERATOR} over 1-{MAX_METER_DENOMINATOR}")

    def tempo(self, bpm: float, locator: str,
              reason: str = "Stored project tempo; tempo automation is not reconstructed.") -> None:
        """An implausible tempo is skipped with a warning, never raised, so one odd value keeps the project."""
        if not plausible_tempo(bpm):
            return self._implausible(f"tempo {_shown(bpm)}", locator, f"0 to {MAX_TEMPO_BPM:g} BPM")
        self._add("tempo", float(bpm), "BPM", locator, reason, confidence="medium")

    def meter(self, numerator: float, denominator: float, locator: str,
              reason: str = "Stored project meter; meter changes are not reconstructed.",
              confidence: Confidence = "medium") -> None:
        """An implausible meter is skipped with a warning, never raised. Odd ones such as 7/5 are accepted."""
        if not plausible_meter(numerator, denominator):
            return self.skip_meter(numerator, denominator, locator)
        self._add("meter", f"{_whole_value(numerator)}/{_whole_value(denominator)}", "signature", locator, reason,
                  confidence=confidence)

    def tempo_map(self, points: Sequence[Mapping[str, float]], locator: str) -> None:
        """Each point is {"beat": position in quarter notes, "bpm": tempo from that beat}.

        Points with an implausible tempo or position are dropped and counted in one warning.
        """
        value = [{"beat": beat, "bpm": float(point["bpm"])} for point in points
                 if (beat := _finite(point["beat"])) is not None and plausible_tempo(point["bpm"])]
        if len(value) < len(points):
            self._implausible(f"{len(points) - len(value)} of {len(points)} project.tempo_map points", locator,
                              f"a finite beat and 0 to {MAX_TEMPO_BPM:g} BPM")
        if value:
            self._add("project.tempo_map", value, "BPM", locator,
                      "Stored tempo changes in beat order; interpolation between points is not modeled.",
                      time_domain="project_beats")

    def sample_rate(self, hz: int, locator: str) -> None:
        rate = _finite_number(hz, "sample rate", positive=True)
        self._add("project.sample_rate", int(rate) if rate.is_integer() else rate, "Hz", locator,
                  "Project sample-rate setting; the rendered file may differ.")

    def key(self, text: str, locator: str, reason: str = "Stored key setting; it records intent, not the audible key.",
            confidence: Confidence = "medium") -> None:
        self._add("project.key", text, "text", locator, reason, confidence=confidence)

    def track_count(self, count: int, locator: str) -> None:
        self._add("project.track_count", _count(count, "track count"), "count", locator, "Count of stored tracks.")

    def track(self, index: int, name: str | None, kind: TrackKind, locator: str) -> None:
        if kind not in TRACK_KINDS:
            raise ValueError(f"unknown track kind {kind!r}")
        self._add("project.track", {"name": name, "kind": kind, "index": index}, "track", locator,
                  "Stored track name and type.", scope=_track_scope(index))

    def marker(self, name: str | None, position: float, unit: PositionUnit, locator: str) -> None:
        self._add("project.marker", {"name": name, "position": _finite_number(position, "marker position"),
                                     "unit": unit}, "marker", locator,
                  "Stored arrangement marker.", time_domain=TIME_DOMAINS[unit])

    def plugin(self, name: str, track: int | None, locator: str) -> None:
        self._add("project.plugin", {"name": name, "track": track}, "plugin", locator,
                  "Plugin named in the project; it was not loaded or run.",
                  scope=None if track is None else _track_scope(track))

    def media_references(self, raw_paths: Iterable[str], locator: str) -> None:
        counts = Counter(name for name in map(_media_basename, raw_paths) if name)
        for basename, count in sorted(counts.items()):
            self._add("project.media_reference", {"basename": basename, "count": count}, "reference", locator,
                      "Basename of a referenced media file; the path was not followed or opened.")

    def clip_count(self, count: int, locator: str, track: int | None = None) -> None:
        self._add("project.clip_count", _count(count, "clip count"), "count", locator, "Count of stored clips.",
                  scope=None if track is None else _track_scope(track))

    def track_notes(self, index: int, notes: Sequence[Note], locator: str,
                    reason: str = "Aggregated from stored notes; beats are quarter notes, not audible timing.",
                    confidence: Confidence = "high") -> None:
        """`reason` and `confidence` let an adapter state which timing it approximated."""
        if not notes:
            return
        scope = _track_scope(index)
        self._add("project.track_notes", summarize_notes(notes), "note_summary", locator, reason,
                  scope=scope, confidence=confidence)
        if self.detail != "full":
            return
        for note in notes:
            self._add("project.note", {"pitch": note.pitch, "velocity": note.velocity, "startBeat": note.start_beat,
                                       "endBeat": note.end_beat, "channel": note.channel}, "note", locator,
                      "Stored note; no audible timing is inferred.", scope=scope, time_domain="project_beats")
