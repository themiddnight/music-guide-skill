from __future__ import annotations
import io
from pathlib import Path
import mido
from ..model import Detail, Diagnostic, Evidence
from ..notes import Note, summarize_notes
from .common import MAX_MIDI_BYTES, _read_bounded

def midi(path: Path, source_id: str, evidence: Evidence, detail: Detail) -> None:
    raw = _read_bounded(path, MAX_MIDI_BYTES)
    song = mido.MidiFile(file=io.BytesIO(raw))
    if song.type not in (0, 1):
        raise ValueError("MIDI type 2 is unsupported; export SMF 0 or 1")
    if song.ticks_per_beat <= 0:
        raise ValueError("SMPTE or invalid MIDI time division is unsupported")
    ticks_per_beat = song.ticks_per_beat
    evidence.add(source_id, "midi.ticks_per_beat", ticks_per_beat, "ticks/quarter_note",
                 "project", "static", "header", "mido", "extracted", "high", "Stored MIDI header time division.")
    note_count = 0
    for track_index, track in enumerate(song.tracks):
        tick = 0
        pending: dict[tuple[int, int], list[tuple[int, int]]] = {}
        notes: list[Note] = []
        for event_index, msg in enumerate(track):
            tick += msg.time
            locator = f"track={track_index};event={event_index};tick={tick}"
            if msg.type == "track_name":
                evidence.add(source_id, "midi.track_name", msg.name, "text",
                             f"midi_track:{track_index}", "static", locator,
                             "mido", "extracted", "high", "Stored MIDI track name.")
            elif msg.type == "set_tempo":
                if msg.tempo <= 0:
                    raise ValueError(f"invalid MIDI tempo at {locator}")
                evidence.add(source_id, "tempo", round(mido.tempo2bpm(msg.tempo), 8), "BPM",
                             "project", "midi_ticks", locator,
                             "mido", "extracted", "high", "Stored MIDI tempo event; may change later.")
            elif msg.type == "time_signature":
                evidence.add(source_id, "meter", f"{msg.numerator}/{msg.denominator}", "signature",
                             "project", "midi_ticks", locator,
                             "mido", "extracted", "high", "Stored MIDI meter event; may change later.")
            elif msg.type == "note_on" and msg.velocity > 0:
                pending.setdefault((msg.channel, msg.note), []).append((tick, msg.velocity))
            elif msg.type in ("note_off", "note_on"):
                starts = pending.get((msg.channel, msg.note), [])
                if starts:
                    start_tick, velocity = starts.pop(0)
                    notes.append(Note(msg.note, velocity, start_tick / ticks_per_beat,
                                      tick / ticks_per_beat, msg.channel))
                    if detail == "full":
                        evidence.add(source_id, "midi.note", {
                            "pitch": msg.note, "channel": msg.channel, "velocity": velocity,
                            "startTick": start_tick, "endTick": tick,
                            "startBeat": start_tick / ticks_per_beat,
                            "endBeat": tick / ticks_per_beat,
                        }, "MIDI_note", f"midi_track:{track_index}", "midi_ticks",
                            f"track={track_index};ticks={start_tick}-{tick}", "mido",
                            "extracted", "high", "Paired stored note-on and note-off messages; no audible timing is inferred.")
                else:
                    evidence.diagnostics.append(Diagnostic("unmatched_note_off", "warning",
                        f"Unmatched MIDI note-off at {locator}.", source_id))
        for starts in pending.values():
            if starts:
                evidence.diagnostics.append(Diagnostic("unmatched_note_on", "warning",
                    f"{len(starts)} MIDI note(s) in track {track_index} have no note-off.", source_id))
        if notes:
            note_count += len(notes)
            evidence.add(source_id, "midi.track_summary", summarize_notes(notes), "note_summary",
                         f"midi_track:{track_index}", "static", f"track={track_index}", "mido+summarize_notes",
                         "extracted", "high", "Aggregated from paired stored notes; beats are quarter notes, not audible timing.")
    evidence.add(source_id, "midi.note_count", note_count, "count", "project", "static",
                 "whole file", "mido", "extracted", "high", "Count of paired MIDI note events.")
