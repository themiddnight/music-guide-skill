"""Shared emission for notated-score adapters (MusicXML, MuseScore): one structure in, one vocabulary out."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..model import Detail, Diagnostic, Evidence
from ..notes import Note
from ..project import ProjectFacts

# Steps on the circle of fifths from the major key with the same signature to the mode's tonic.
MAJOR_MODES = {"major": 0, "ionian": 0, "mixolydian": 1, "dorian": 2, "minor": 3, "aeolian": 3,
               "phrygian": 4, "locrian": 5, "lydian": -1}
LETTERS = "FCGDAEB"


def _tonic(position: int) -> str:
    # Position 0 is C; each step is a fifth up. Sharps and flats accumulate every seven steps.
    letter = LETTERS[(position + 1) % 7]
    accidentals = (position + 1) // 7
    return letter + ("#" * accidentals if accidentals > 0 else "b" * -accidentals)


def key_name(fifths: int, mode: str | None) -> str | None:
    """Name the key for a signature; None when the mode says there is no tonal center."""
    if not -14 <= fifths <= 14:
        raise ValueError(f"key signature of {fifths} fifths is out of range")
    if mode is None:
        return f"{_tonic(fifths)} major or {_tonic(fifths + 3)} minor (mode not stored)"
    mode = mode.strip().lower()
    if mode == "none":
        return None
    if mode not in MAJOR_MODES:
        raise ValueError(f"unknown key mode {mode!r}")
    return f"{_tonic(fifths + MAJOR_MODES[mode])} {mode}"


@dataclass
class ScorePart:
    name: str | None
    locator: str
    notes: list[Note] = field(default_factory=list)
    instrument: str | None = None
    placeholder_velocities: int = 0


@dataclass
class ScoreData:
    """Everything an adapter reads from a score. Beats are quarter notes from the start of the score."""

    parts: list[ScorePart] = field(default_factory=list)
    tempos: list[tuple[float, float]] = field(default_factory=list)
    tempo_locator: str = ""
    meters: list[tuple[float, int, int]] = field(default_factory=list)
    keys: list[tuple[float, int, str | None]] = field(default_factory=list)
    markers: list[tuple[str | None, float]] = field(default_factory=list)
    measures: int = 0
    length_beats: float = 0.0
    default_velocity: int = 90


def _extra(evidence: Evidence, source_id: str, method: str, property: str, value: Any, unit: str,
           scope: str, time_domain: str, locator: str, reason: str) -> None:
    evidence.add(source_id, property, value, unit, scope, time_domain, locator, method, "extracted", "high", reason)


def _distinct(values: list[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def emit_score(evidence: Evidence, source_id: str, method: str, prefix: str, detail: Detail,
               data: ScoreData) -> None:
    facts = ProjectFacts(evidence, source_id, method, detail)
    facts.track_count(len(data.parts), "parts")
    for index, part in enumerate(data.parts):
        facts.track(index, part.name, "instrument", part.locator)
        if part.instrument:
            _extra(evidence, source_id, method, f"{prefix}.instrument", part.instrument, "text",
                   f"project_track:{index}", "static", part.locator, "Instrument name stored for the part.")
        _notes(evidence, source_id, method, prefix, facts, data, index, part)
    placeholders = sum(part.placeholder_velocities for part in data.parts)
    if placeholders:
        total = sum(len(part.notes) for part in data.parts)
        evidence.diagnostics.append(Diagnostic(
            "score_velocity_placeholder", "info",
            f"{placeholders} of {total} notes store no velocity and carry the placeholder {data.default_velocity}. "
            "Velocity statistics describe notation, not a performance.", source_id))
    _tempo(evidence, source_id, method, facts, data)
    _meter(evidence, source_id, method, prefix, facts, data)
    _key(evidence, source_id, method, prefix, facts, data)
    seen_markers: set[tuple[str | None, float]] = set()
    for name, beat in data.markers:
        if (name, beat) not in seen_markers:
            seen_markers.add((name, beat))
            facts.marker(name, beat, "beats", "rehearsal marks")
    _extra(evidence, source_id, method, f"{prefix}.measure_count", data.measures, "count", "project", "static",
           "measures", "Longest part, in stored measures; repeats and jumps are not unrolled.")
    _extra(evidence, source_id, method, f"{prefix}.length", data.length_beats, "quarter_notes", "project", "static",
           "measure contents", "Stored notated length of the longest part; repeats and jumps are not unrolled.")


def _notes(evidence: Evidence, source_id: str, method: str, prefix: str, facts: ProjectFacts, data: ScoreData,
           index: int, part: ScorePart) -> None:
    total = len(part.notes)
    if not part.placeholder_velocities:
        facts.track_notes(index, part.notes, part.locator)
        return
    reason = (f"Aggregated from stored notes; beats are quarter notes, not audible timing. "
              f"{part.placeholder_velocities} of {total} velocities are placeholders ({data.default_velocity}) "
              "because the score stores none, so velocity statistics describe notation, not a performance.")
    facts.track_notes(index, part.notes, part.locator, reason, "medium")
    _extra(evidence, source_id, method, f"{prefix}.velocity_placeholders",
           {"placeholder": part.placeholder_velocities, "total": total, "velocity": data.default_velocity},
           "count", f"project_track:{index}", "static", part.locator,
           "Notes whose velocity is the adapter placeholder, not a stored value.")


def _tempo(evidence: Evidence, source_id: str, method: str, facts: ProjectFacts, data: ScoreData) -> None:
    by_beat: dict[float, float] = {}
    for beat, bpm in data.tempos:
        if by_beat.setdefault(beat, bpm) != bpm:
            evidence.diagnostics.append(Diagnostic(
                "score_tempo_disagreement", "warning",
                f"Parts disagree on the tempo at beat {beat:g} ({by_beat[beat]:g} and {bpm:g} BPM); the first was kept.",
                source_id))
    if not by_beat:
        return
    points = [{"beat": beat, "bpm": by_beat[beat]} for beat in sorted(by_beat)]
    facts.tempo_map(points, data.tempo_locator)
    if len({point["bpm"] for point in points}) == 1 and points[0]["beat"] == 0:
        facts.tempo(points[0]["bpm"], data.tempo_locator,
                    "Every stored tempo mark agrees and the first sits at the start; rubato, ritardando and "
                    "fermatas are not modeled.")


def _meter(evidence: Evidence, source_id: str, method: str, prefix: str, facts: ProjectFacts,
           data: ScoreData) -> None:
    by_beat: dict[float, tuple[int, int]] = {}
    for beat, numerator, denominator in data.meters:
        by_beat.setdefault(beat, (numerator, denominator))
    if not by_beat:
        return
    ordered = sorted(by_beat.items())
    if len(_distinct([meter for _, meter in ordered])) == 1 and ordered[0][0] == 0:
        facts.meter(*ordered[0][1], "time signature")
        return
    _extra(evidence, source_id, method, f"{prefix}.meter_map",
           [{"beat": beat, "meter": f"{n}/{d}"} for beat, (n, d) in ordered], "signature", "project",
           "project_beats", "time signatures", "Stored time signature changes in beat order.")


def _key(evidence: Evidence, source_id: str, method: str, prefix: str, facts: ProjectFacts,
         data: ScoreData) -> None:
    events = sorted(_distinct(data.keys), key=lambda item: item[0])
    if not events:
        return
    names = [(beat, key_name(fifths, mode)) for beat, fifths, mode in events]
    named = [(beat, name) for beat, name in names if name]
    if not named:
        return
    if len({name for _, name in named}) == 1 and len(named) == len(events) and named[0][0] == 0:
        facts.key(named[0][1], "key signature")
        return
    _extra(evidence, source_id, method, f"{prefix}.key_map", [{"beat": beat, "key": name} for beat, name in named],
           "text", "project", "project_beats", "key signatures",
           "Stored key signatures from non-transposing parts, in beat order.")
