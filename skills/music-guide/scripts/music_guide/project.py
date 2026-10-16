"""Shared project vocabulary so every DAW adapter emits the same properties."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence, get_args

from .adapters.common import _finite_number, _media_basename
from .model import Confidence, Detail, Evidence
from .notes import Note, summarize_notes

TrackKind = Literal["audio", "midi", "instrument", "group", "return", "master", "folder", "unknown"]
TRACK_KINDS = frozenset(get_args(TrackKind))
PositionUnit = Literal["beats", "seconds"]
TIME_DOMAINS: dict[str, str] = {"beats": "project_beats", "seconds": "project_seconds"}


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
        self.evidence.add(self.source_id, property, value, unit, self.scope if scope is None else scope, time_domain, locator,
                          self.method, "extracted", confidence, reason)

    def application(self, name: str, locator: str) -> None:
        self._add("project.application", name, "text", locator, "Application name stored in the project file.")

    def tempo(self, bpm: float, locator: str,
              reason: str = "Stored project tempo; tempo automation is not reconstructed.") -> None:
        self._add("tempo", _finite_number(bpm, "tempo", positive=True), "BPM", locator, reason, confidence="medium")

    def meter(self, numerator: int, denominator: int, locator: str,
              reason: str = "Stored project meter; meter changes are not reconstructed.") -> None:
        if _count(numerator, "meter numerator") == 0 or _count(denominator, "meter denominator") == 0:
            raise ValueError("meter numerator and denominator must be positive")
        self._add("meter", f"{numerator}/{denominator}", "signature", locator, reason, confidence="medium")

    def tempo_map(self, points: Sequence[Mapping[str, float]], locator: str) -> None:
        """Each point is {"beat": position in quarter notes, "bpm": tempo from that beat}."""
        value = [{"beat": _finite_number(point["beat"], "tempo map beat"),
                  "bpm": _finite_number(point["bpm"], "tempo map bpm", positive=True)} for point in points]
        self._add("project.tempo_map", value, "BPM", locator,
                  "Stored tempo changes in beat order; interpolation between points is not modeled.",
                  time_domain="project_beats")

    def sample_rate(self, hz: int, locator: str) -> None:
        rate = _finite_number(hz, "sample rate", positive=True)
        self._add("project.sample_rate", int(rate) if rate.is_integer() else rate, "Hz", locator,
                  "Project sample-rate setting; the rendered file may differ.")

    def key(self, text: str, locator: str) -> None:
        self._add("project.key", text, "text", locator,
                  "Stored key setting; it records intent, not the audible key.", confidence="medium")

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

    def track_notes(self, index: int, notes: Sequence[Note], locator: str) -> None:
        if not notes:
            return
        scope = _track_scope(index)
        self._add("project.track_notes", summarize_notes(notes), "note_summary", locator,
                  "Aggregated from stored notes; beats are quarter notes, not audible timing.", scope=scope)
        if self.detail != "full":
            return
        for note in notes:
            self._add("project.note", {"pitch": note.pitch, "velocity": note.velocity, "startBeat": note.start_beat,
                                       "endBeat": note.end_beat, "channel": note.channel}, "note", locator,
                      "Stored note; no audible timing is inferred.", scope=scope, time_domain="project_beats")
