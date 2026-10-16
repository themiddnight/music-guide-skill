"""Command-line orchestration and strict JSON evidence output."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import soundfile as sf
from defusedxml.common import DefusedXmlException

from .adapters import audio, dawproject, midi, solna
from .adapters.common import MAX_JSON_BYTES, _read_bounded
from .model import Diagnostic, Evidence, Source, detect_conflicts

ADAPTERS = {
    "mixdown": audio.audio,
    "stem": audio.audio,
    "midi": midi.midi,
    "solna": solna.solna,
    "dawproject": dawproject.dawproject,
}
LIMITS = {"mixdown": 4 * 1024**3, "stem": 4 * 1024**3,
          "midi": 32 * 1024**2, "solna": 8 * 1024**2,
          "dawproject": 64 * 1024**2}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_declarations(path: Path, evidence: Evidence) -> None:
    data = json.loads(_read_bounded(path, MAX_JSON_BYTES),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite declaration: {value}")))
    if not isinstance(data, list) or len(data) > 1000:
        raise ValueError("declarations must be a bounded JSON array")
    source_id = f"s{len(evidence.sources) + 1}"
    source = Source(source_id, path.name, "declarations", "json-declarations", "ok", _hash(path))
    staged = Evidence()
    for index, item in enumerate(data):
        if not isinstance(item, dict) or set(item) != {"property", "value", "unit", "scope"}:
            raise ValueError(f"declarations[{index}] must have property, value, unit, scope only")
        if any(not isinstance(item[key], str) or not item[key].strip() for key in ("property", "unit", "scope")):
            raise ValueError(f"declarations[{index}] has an empty property, unit or scope")
        value = item["value"]
        if value is None or isinstance(value, (dict, list)) or (isinstance(value, float) and not math.isfinite(value)):
            raise ValueError(f"declarations[{index}] value must be a finite scalar")
        staged.add(source_id, item["property"], value, item["unit"], item["scope"], "static",
                   f"declarations[{index}]", "user-manifest", "user_declared", "low",
                   "Supplied by user; not independently verified.")
    evidence.sources.append(source)
    _merge(evidence, staged)


def _merge(target: Evidence, staged: Evidence) -> None:
    for item in staged.observations:
        target.add(item.sourceIds[0], item.property, item.value, item.unit, item.scope,
                   item.timeDomain, item.locator, item.method, item.provenance,
                   item.confidence, item.confidenceReason)
    target.diagnostics.extend(staged.diagnostics)


def build(mixdown: Path, optionals: list[tuple[str, Path]], declarations: Path | None) -> Evidence:
    evidence = Evidence(limitations=[
        "WAV sample peak is not true peak; RMS is not LUFS.",
        "MIDI, SOLNA and DAWproject describe stored intent, not rendered sound.",
        "No semantic listening, key estimation, genre classification or artwork generation is performed.",
    ])
    for role, path in [("mixdown", mixdown), *optionals]:
        source_id = f"s{len(evidence.sources) + 1}"
        adapter_name = role if role in ADAPTERS else "unsupported"
        source = Source(source_id, path.name, role, adapter_name, "ok", None)
        evidence.sources.append(source)
        try:
            if role == "project":
                if path.is_file() and path.stat().st_size <= 64 * 1024**2:
                    source.sha256 = _hash(path)
                source.status = "unsupported"
                evidence.diagnostics.append(Diagnostic("unsupported_project", "warning",
                    f"{path.name}: unsupported native project. Export a WAV mixdown and, if available, MIDI or DAWproject metadata.", source_id))
                continue
            if not path.is_file():
                raise ValueError("source is not a readable regular file")
            if path.stat().st_size > LIMITS[role]:
                raise ValueError(f"source exceeds {LIMITS[role]} byte limit")
            source.sha256 = _hash(path)
            staged = Evidence()
            if role == "stem":
                audio.audio(path, source_id, staged, scope=f"stem:{source_id}")
            else:
                ADAPTERS[role](path, source_id, staged)
            _merge(evidence, staged)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, EOFError,
                sf.LibsndfileError, zipfile.BadZipFile, ET.ParseError, DefusedXmlException) as error:
            source.status = "error"
            if role == "mixdown":
                raise ValueError(f"required mixdown {path.name}: {error}") from error
            evidence.diagnostics.append(Diagnostic("optional_source_error", "warning",
                f"Could not parse optional {role} {path.name}: {error}", source_id))
    if declarations:
        try:
            _load_declarations(declarations, evidence)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            source_id = f"s{len(evidence.sources) + 1}"
            evidence.sources.append(Source(source_id, declarations.name, "declarations", "json-declarations", "error", None))
            evidence.diagnostics.append(Diagnostic("optional_source_error", "warning",
                f"Could not parse declarations {declarations.name}: {error}", source_id))
    detect_conflicts(evidence)
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract local music evidence as JSON")
    parser.add_argument("--mixdown", required=True, type=Path, help="Readable WAV PCM or IEEE float mixdown")
    parser.add_argument("--midi", action="append", type=Path, default=[])
    parser.add_argument("--stem", action="append", type=Path, default=[], help="Optional WAV stem; measured independently")
    parser.add_argument("--solna", action="append", type=Path, default=[])
    parser.add_argument("--dawproject", action="append", type=Path, default=[])
    parser.add_argument("--project", action="append", type=Path, default=[], help="Unsupported native project to inventory")
    parser.add_argument("--declarations", type=Path, help="Optional JSON array of user-declared facts")
    args = parser.parse_args(argv)
    optionals = [(kind, path) for kind in ("stem", "midi", "solna", "dawproject", "project")
                 for path in getattr(args, kind)]
    try:
        result = build(args.mixdown, optionals, args.declarations)
        print(json.dumps(result.as_json(), ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    except ValueError as error:
        print(f"music-guide: {error}", file=sys.stderr)
        return 2
