from __future__ import annotations
import io
from collections import deque
from pathlib import Path
import mido
from ..model import Detail, Diagnostic, Evidence, NoteStarts
from ..notes import Note, summarize_notes

GM_PERCUSSION_CHANNEL = 9
from .common import MAX_MIDI_BYTES, _read_bounded
from .listing import MAX_LISTED, MAX_NOTE_LISTING, capped, truncation

def midi(path: Path, source_id: str, evidence: Evidence, detail: Detail) -> None:
    raw = _read_bounded(path, MAX_MIDI_BYTES)
    try:
        song = mido.MidiFile(file=io.BytesIO(raw))
    except Exception as error:  # mido leaks IndexError, KeyError, struct.error and its own KeySignatureError.
        raise ValueError(f"malformed MIDI data ({type(error).__name__}): {error}") from error
    if song.type not in (0, 1):
        raise ValueError("MIDI type 2 is unsupported; export SMF 0 or 1")
    if song.ticks_per_beat <= 0:
        raise ValueError("SMPTE or invalid MIDI time division is unsupported")
    ticks_per_beat = song.ticks_per_beat
    evidence.add(source_id, "midi.ticks_per_beat", ticks_per_beat, "ticks/quarter_note",
                 "project", "static", "header", "mido", "extracted", "high", "Stored MIDI header time division.")
    note_count = listed = names = 0
    tempos: list[tuple[int, str, float]] = []
    meters: list[tuple[int, str, str]] = []
    for track_index, track in enumerate(song.tracks):
        tick = 0
        pending: dict[tuple[int, int], deque[tuple[int, int]]] = {}
        notes: list[Note] = []
        unmatched_offs = 0
        first_unmatched = ""
        for event_index, msg in enumerate(track):
            tick += msg.time
            locator = f"track={track_index};event={event_index};tick={tick}"
            if msg.type == "track_name":
                names += 1
                if names > MAX_LISTED:
                    continue
                evidence.add(source_id, "midi.track_name", msg.name, "text",
                             f"midi_track:{track_index}", "static", locator,
                             "mido", "extracted", "high", "Stored MIDI track name.")
            elif msg.type == "set_tempo":
                if msg.tempo <= 0:
                    raise ValueError(f"invalid MIDI tempo at {locator}")
                tempos.append((tick, locator, round(mido.tempo2bpm(msg.tempo), 8)))
            elif msg.type == "time_signature":
                meters.append((tick, locator, f"{msg.numerator}/{msg.denominator}"))
            elif msg.type == "note_on" and msg.velocity > 0:
                pending.setdefault((msg.channel, msg.note), deque()).append((tick, msg.velocity))
            elif msg.type in ("note_off", "note_on"):
                starts = pending.get((msg.channel, msg.note))
                if starts:
                    start_tick, velocity = starts.popleft()
                    notes.append(Note(msg.note, velocity, start_tick / ticks_per_beat,
                                      tick / ticks_per_beat, msg.channel))
                    if detail == "full" and listed < MAX_NOTE_LISTING:
                        listed += 1
                        evidence.add(source_id, "midi.note", {
                            "pitch": msg.note, "channel": msg.channel, "velocity": velocity,
                            "startTick": start_tick, "endTick": tick,
                            "startBeat": start_tick / ticks_per_beat,
                            "endBeat": tick / ticks_per_beat,
                        }, "MIDI_note", f"midi_track:{track_index}", "midi_ticks",
                            f"track={track_index};ticks={start_tick}-{tick}", "mido",
                            "extracted", "high", "Paired stored note-on and note-off messages; no audible timing is inferred.")
                else:
                    unmatched_offs += 1
                    first_unmatched = first_unmatched or locator
        if unmatched_offs:
            evidence.diagnostics.append(Diagnostic("unmatched_note_off", "warning",
                f"{unmatched_offs} unmatched MIDI note-off message(s) in track {track_index}, the first at {first_unmatched}.",
                source_id))
        unmatched_ons = sum(len(starts) for starts in pending.values())
        if unmatched_ons:
            evidence.diagnostics.append(Diagnostic("unmatched_note_on", "warning",
                f"{unmatched_ons} MIDI note(s) in track {track_index} have no note-off.", source_id))
        if notes:
            note_count += len(notes)
            summary = summarize_notes(notes)
            summary["percussion"] = summary["channels"] == [GM_PERCUSSION_CHANNEL]
            reason = "Aggregated from paired stored notes; beats are quarter notes, not audible timing."
            if summary["percussion"]:
                reason += (" All notes use General MIDI channel 10, so note numbers name drum sounds and the pitch"
                           " fields do not describe pitch.")
            evidence.add(source_id, "midi.track_summary", summary, "note_summary",
                         f"midi_track:{track_index}", "static", f"track={track_index}", "mido+summarize_notes",
                         "extracted", "high", reason)
            source = evidence.observations[-1]
            evidence.note_starts[(source_id, source.scope)] = NoteStarts(
                tuple(note.start_beat for note in notes), source.method, source.locator,
                source.confidence, source.confidenceReason, source.scope)
    if names > MAX_LISTED:
        evidence.diagnostics.append(truncation("track names", MAX_LISTED, names, source_id))
    if note_count > listed and detail == "full":
        evidence.diagnostics.append(truncation("notes", listed, note_count, source_id))
    tempos = list(capped(tempos, "tempo events", source_id, evidence))
    meters = list(capped(meters, "meter events", source_id, evidence))
    _timeline(evidence, source_id, ticks_per_beat, tempos, "tempo", "bpm", "BPM", "set_tempo")
    _timeline(evidence, source_id, ticks_per_beat, meters, "meter", "meter", "signature", "time_signature")
    evidence.add(source_id, "midi.note_count", note_count, "count", "project", "static",
                 "whole file", "mido", "extracted", "high", "Count of paired MIDI note events.")


def _timeline(evidence: Evidence, source_id: str, ticks_per_beat: int, events: list[tuple[int, str, object]],
              prop: str, key: str, unit: str, event_name: str) -> None:
    if not events:
        return
    events = sorted(events, key=lambda event: event[0])
    evidence.add(source_id, f"midi.{prop}_map",
                 [{"tick": tick, "beat": tick / ticks_per_beat, key: value} for tick, _, value in events],
                 unit, "project", "midi_ticks", f"all {event_name} events", "mido", "extracted", "high",
                 f"Stored MIDI {event_name} events in tick order across all tracks.")
    if len(events) == 1 and events[0][0] == 0:
        _, locator, value = events[0]
        evidence.add(source_id, prop, value, unit, "project", "static", locator,
                     f"midi-constant-{prop}", "extracted", "high",
                     f"The file's only {event_name} event, at tick 0, so it holds for the whole file; render not verified.")
