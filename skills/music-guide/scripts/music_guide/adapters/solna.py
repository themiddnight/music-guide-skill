"""Read-only solna adapter."""

from __future__ import annotations
import json
from pathlib import Path
from ..model import Evidence
from .common import MAX_JSON_BYTES, _read_bounded, _finite_number

def solna(path: Path, source_id: str, evidence: Evidence) -> None:
    def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    data = json.loads(_read_bounded(path, MAX_JSON_BYTES), object_pairs_hook=no_duplicates,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON value: {value}")))
    if not isinstance(data, dict):
        raise ValueError("SOLNA root must be an object")
    version = data.get("formatVersion")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("SOLNA formatVersion must be a positive integer")
    if version > 12:
        raise ValueError(f"SOLNA formatVersion {version} is newer than the inspected version 12")
    content = data.get("content")
    if not isinstance(content, dict):
        raise ValueError("SOLNA content must be an object")
    bpm = _finite_number(content.get("bpm"), "content.bpm", positive=True)
    meter = content.get("meterId")
    if isinstance(meter, bool) or not isinstance(meter, (str, int)) or str(meter).strip() == "":
        raise ValueError("content.meterId must be a nonempty string or integer")
    loops = content.get("loops")
    if not isinstance(loops, list) or len(loops) > 10000:
        raise ValueError("content.loops must be a bounded array")
    evidence.add(source_id, "solna.format_version", version, "version", "project", "static",
                 "formatVersion", "solna-json", "extracted", "high", "Stored SOLNA format version.")
    evidence.add(source_id, "tempo", bpm, "BPM", "project", "static", "content.bpm",
                 "solna-json", "extracted", "high", "Stored SOLNA project tempo; render not verified.")
    evidence.add(source_id, "solna.meter_id", meter, "identifier", "project", "static",
                 "content.meterId", "solna-json", "extracted", "high", "Stored meter ID; its meaning is not decoded.")
    for index, loop in enumerate(loops):
        if not isinstance(loop, dict):
            raise ValueError(f"content.loops[{index}] must be an object")
        for key in ("scaleRoot", "scaleType"):
            value = loop.get(key)
            if isinstance(value, bool) or not isinstance(value, (str, int)) or str(value).strip() == "":
                raise ValueError(f"content.loops[{index}].{key} must be a nonempty string or integer")
            evidence.add(source_id, f"solna.{key}", value, "identifier", f"loop:{index}",
                         "static", f"content.loops[{index}].{key}", "solna-json", "extracted", "high",
                         "Stored SOLNA field; no audio inference.")
        chords = loop.get("chords")
        if not isinstance(chords, list) or len(chords) > 10000:
            raise ValueError(f"content.loops[{index}].chords must be a bounded array")
        for chord_index, chord in enumerate(chords):
            if not isinstance(chord, dict):
                raise ValueError(f"chord {chord_index} in loop {index} must be an object")
            for key in ("root", "quality"):
                value = chord.get(key)
                if isinstance(value, bool) or not isinstance(value, (str, int)) or str(value).strip() == "":
                    raise ValueError(f"chord {chord_index} in loop {index} has invalid {key}")
            bars = _finite_number(chord.get("bars"), "chord bars", positive=True)
            value = {"root": chord["root"], "quality": chord["quality"], "bars": bars}
            if "bassNote" in chord:
                bass = chord["bassNote"]
                if bass is not None and (isinstance(bass, bool) or not isinstance(bass, (str, int)) or str(bass).strip() == ""):
                    raise ValueError(f"chord {chord_index} in loop {index} has invalid bassNote")
                value["bassNote"] = bass
            evidence.add(source_id, "solna.chord", value, "stored_chord", f"loop:{index}",
                         "loop_sequence", f"content.loops[{index}].chords[{chord_index}]",
                         "solna-json", "extracted", "high", "Stored chord data; pattern and render timing are not decoded.")


