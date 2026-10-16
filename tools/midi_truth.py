#!/usr/bin/env python3
"""Derive ground truth for tools/evaluate.py from a constant-tempo MIDI file.

    midi_truth.py SONG.mid --chord-tracks Pad,Chord --bass-track Bass --melody-track Lead [--offset S] [--duration S]
        [--omit-no-chord]

The MIDI file is read and never written. The output is JSON on stdout. Beats are quarter notes. A bar lasts
numerator * 4 / denominator quarter notes. Chords are labelled per beat from the pitch classes of the chord tracks
plus the bass track's note, then repeats are merged. `--omit-no-chord` leaves the N spans out, for songs where the
chord tracks rest while the rest of the band plays on. The melody track is read as monophonic. A note that overlaps
the next one is cut at the next start and counted on stderr.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import mido

TEMPLATES = {"maj": (0, 4, 7), "min": (0, 3, 7), "sus2": (0, 2, 7), "sus4": (0, 5, 7), "dim": (0, 3, 6),
             "maj7": (0, 4, 7, 11), "min7": (0, 3, 7, 10), "7": (0, 4, 7, 10)}
SHARPS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MIN_PRESENCE = 0.25
MIN_MATCH = 0.5
MIN_SHARED = 3
ROOT_BONUS = 0.3
DECIMALS = 6

Note = tuple[float, float, int]


def read_notes(midi: mido.MidiFile) -> tuple[dict[str, list[Note]], float, tuple[int, int]]:
    """Notes per track name as (start_beat, end_beat, pitch), the tempo in BPM and the time signature."""
    tempos, signatures = set(), set()
    notes: dict[str, list[Note]] = {}
    for track in midi.tracks:
        tick, sounding, found = 0, {}, notes.setdefault(track.name.strip(), [])
        for message in track:
            tick += message.time
            if message.type == "set_tempo":
                tempos.add(message.tempo)
            elif message.type == "time_signature":
                signatures.add((message.numerator, message.denominator))
            elif message.type == "note_on" and message.velocity > 0:
                sounding.setdefault((message.channel, message.note), []).append(tick)
            elif message.type in ("note_off", "note_on") and sounding.get((message.channel, message.note)):
                start = sounding[(message.channel, message.note)].pop(0)
                found.append((start / midi.ticks_per_beat, tick / midi.ticks_per_beat, message.note))
    if len(tempos) != 1:
        raise ValueError(f"expected one constant tempo, found {len(tempos)}")
    if len(signatures) != 1:
        raise ValueError(f"expected one time signature, found {len(signatures)}")
    return notes, mido.tempo2bpm(tempos.pop()), signatures.pop()


def tracks_named(notes: dict[str, list[Note]], names: Sequence[str]) -> list[Note]:
    by_lower = {name.lower(): name for name in notes}
    chosen: list[Note] = []
    for name in names:
        if name.lower() not in by_lower:
            raise ValueError(f"no track named {name!r}; tracks are {', '.join(repr(n) for n in notes if n)}")
        chosen += notes[by_lower[name.lower()]]
    return chosen


def pitch_classes_in(notes: Sequence[Note], start: float, end: float) -> set[int]:
    """Pitch classes that sound for a quarter of the window or begin inside it, so staccato arpeggios still count."""
    weights: dict[int, float] = {}
    found: set[int] = set()
    for note_start, note_end, pitch in notes:
        overlap = min(note_end, end) - max(note_start, start)
        if overlap > 0:
            weights[pitch % 12] = weights.get(pitch % 12, 0.0) + overlap / (end - start)
            if start <= note_start < end:
                found.add(pitch % 12)
    return found | {pc for pc, weight in weights.items() if weight >= MIN_PRESENCE}


def bass_pitch_class(notes: Sequence[Note], start: float, end: float) -> int | None:
    best = None
    for note_start, note_end, pitch in notes:
        overlap = min(note_end, end) - max(note_start, start)
        if overlap >= MIN_PRESENCE * (end - start) and (best is None or (overlap, -pitch) > (best[0], -best[1])):
            best = (overlap, pitch)
    return None if best is None else best[1] % 12


def label_chord(pitch_classes: set[int], bass: int | None) -> str:
    """The best template by Jaccard match, with the bass pitch class pulling the root.

    A dyad or a single note names no chord, so it is N.
    """
    if bass is not None:
        pitch_classes = pitch_classes | {bass}
    if len(pitch_classes) < MIN_SHARED:
        return "N"
    best, best_score = "N", MIN_MATCH
    for quality, shape in TEMPLATES.items():
        for root in range(12):
            template = {(root + step) % 12 for step in shape}
            shared = len(pitch_classes & template)
            score = shared / len(pitch_classes | template)
            if shared < MIN_SHARED or score < MIN_MATCH:
                continue
            if bass == root:
                score += ROOT_BONUS
            if score > best_score + 1e-9:
                best, best_score = f"{SHARPS[root]}:{quality}", score
    return best


def chord_segments(chord_notes: Sequence[Note], bass_notes: Sequence[Note], beat_seconds: float, offset: float,
                   end_beat: float) -> list[dict]:
    segments: list[dict] = []
    for beat in range(math.ceil(end_beat - 1e-9)):
        label = label_chord(pitch_classes_in(chord_notes, beat, beat + 1), bass_pitch_class(bass_notes, beat, beat + 1))
        start, end = beat * beat_seconds + offset, (beat + 1) * beat_seconds + offset
        if segments and segments[-1]["label"] == label:
            segments[-1]["endSeconds"] = round(end, DECIMALS)
        else:
            segments.append({"startSeconds": round(start, DECIMALS), "endSeconds": round(end, DECIMALS),
                             "label": label})
    return segments


def melody_notes(notes: Sequence[Note], beat_seconds: float, offset: float) -> tuple[list[dict], int]:
    ordered = sorted(notes, key=lambda n: (n[0], -n[2]))
    kept: list[list] = []
    overlaps = 0
    for start, end, pitch in ordered:
        if kept and start < kept[-1][1]:
            overlaps += 1
            if start == kept[-1][0]:
                continue
            kept[-1][1] = start
        kept.append([start, end, pitch])
    return ([{"startSeconds": round(s * beat_seconds + offset, DECIMALS),
              "endSeconds": round(e * beat_seconds + offset, DECIMALS), "midi": p} for s, e, p in kept if e > s],
            overlaps)


def derive_truth(path: Path, *, chord_tracks: Sequence[str], bass_track: str | None = None,
                 melody_track: str | None = None, offset: float = 0.0, duration: float | None = None,
                 omit_no_chord: bool = False) -> dict:
    notes, bpm, (numerator, denominator) = read_notes(mido.MidiFile(path))
    beat_seconds = 60.0 / bpm
    bar_seconds = numerator * 4 / denominator * beat_seconds
    last_beat = max((end for track in notes.values() for _, end, _ in track), default=0.0)
    length = duration - offset if duration is not None else last_beat * beat_seconds
    chord_notes = tracks_named(notes, chord_tracks)
    bass_notes = tracks_named(notes, [bass_track]) if bass_track else []
    harmony_end = max((end for _, end, _ in chord_notes + bass_notes), default=0.0)
    truth = {
        "tempoBpm": round(bpm, DECIMALS),
        "meter": f"{numerator}/{denominator}",
        "beats": [round(k * beat_seconds + offset, DECIMALS) for k in range(math.ceil(length / beat_seconds - 1e-9))],
        "downbeats": [round(k * bar_seconds + offset, DECIMALS) for k in range(math.ceil(length / bar_seconds - 1e-9))],
        "chords": [c for c in chord_segments(chord_notes, bass_notes, beat_seconds, offset, harmony_end)
                   if not (omit_no_chord and c["label"] == "N")],
    }
    if melody_track:
        truth["melody"], overlaps = melody_notes(tracks_named(notes, [melody_track]), beat_seconds, offset)
        if overlaps:
            print(f"midi_truth: {overlaps} overlapping melody note(s) were cut or dropped; "
                  f"track {melody_track!r} is not monophonic", file=sys.stderr)
    return truth


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Derive evaluation truth from a constant-tempo MIDI file.")
    parser.add_argument("midi", type=Path)
    parser.add_argument("--chord-tracks", required=True, help="comma-separated track names")
    parser.add_argument("--bass-track")
    parser.add_argument("--melody-track")
    parser.add_argument("--offset", type=float, default=0.0, help="seconds between MIDI start and audio start")
    parser.add_argument("--duration", type=float, help="audio length in seconds; the beat grid runs to it")
    parser.add_argument("--omit-no-chord", action="store_true", help="leave N spans out of the chord truth")
    args = parser.parse_args(argv)
    try:
        truth = derive_truth(args.midi, chord_tracks=[n.strip() for n in args.chord_tracks.split(",")],
                             bass_track=args.bass_track, melody_track=args.melody_track, offset=args.offset,
                             duration=args.duration, omit_no_chord=args.omit_no_chord)
    except (ValueError, OSError) as error:
        print(f"midi_truth: {error}", file=sys.stderr)
        return 2
    json.dump(truth, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
