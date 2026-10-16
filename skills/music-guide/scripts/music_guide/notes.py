from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


@dataclass(frozen=True)
class Note:
    pitch: int
    velocity: int
    start_beat: float
    end_beat: float
    channel: int | None = None


def pitch_name(pitch: int) -> str:
    # Scientific pitch notation with MIDI 60 as C4.
    return f"{PITCH_CLASSES[pitch % 12]}{pitch // 12 - 1}"


def _number(value: float) -> float | int:
    rounded = round(value, 6)
    return int(rounded) if rounded == int(rounded) else rounded


def _max_polyphony(notes: Sequence[Note]) -> int:
    # An end and a start at the same beat do not overlap, so ends sort first.
    events = sorted([(note.start_beat, 1) for note in notes] + [(note.end_beat, -1) for note in notes])
    current = peak = 0
    for _, change in events:
        current += change
        peak = max(peak, current)
    return peak


def summarize_notes(notes: Sequence[Note]) -> dict[str, Any]:
    if not notes:
        raise ValueError("cannot summarize an empty note list")
    counts = dict.fromkeys(PITCH_CLASSES, 0)
    beats = dict.fromkeys(PITCH_CLASSES, 0.0)
    for note in notes:
        name = PITCH_CLASSES[note.pitch % 12]
        counts[name] += 1
        beats[name] += note.end_beat - note.start_beat
    low = min(note.pitch for note in notes)
    high = max(note.pitch for note in notes)
    velocities = [note.velocity for note in notes]
    return {
        "noteCount": len(notes),
        "pitchMin": {"midi": low, "name": pitch_name(low)},
        "pitchMax": {"midi": high, "name": pitch_name(high)},
        "pitchClassCounts": counts,
        "pitchClassBeats": {name: _number(value) for name, value in beats.items()},
        "velocityMin": min(velocities),
        "velocityMax": max(velocities),
        "velocityMean": _number(sum(velocities) / len(velocities)),
        "startBeat": _number(min(note.start_beat for note in notes)),
        "endBeat": _number(max(note.end_beat for note in notes)),
        "maxPolyphony": _max_polyphony(notes),
        "channels": sorted({note.channel for note in notes if note.channel is not None}),
    }
