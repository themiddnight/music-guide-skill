from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from defusedxml.common import DefusedXmlException

from . import formats
from .adapters.common import MAX_AUDIO_BYTES, MAX_JSON_BYTES, MAX_ZIP_BYTES, _load_json, _read_bounded
from .formats import ProjectFormat
from .model import Detail, Diagnostic, Evidence, Source, detect_conflicts

AUDIO_ROLES = ("mixdown", "stem")
# Explicit flags name their format, so their files skip extension and signature detection.
FLAG_FORMATS = ("midi", "solna", "dawproject")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_declarations(path: Path, evidence: Evidence) -> None:
    data = _load_json(_read_bounded(path, MAX_JSON_BYTES))
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


# Failures a parser reports on purpose. Anything else is a defect worth naming by type.
EXPECTED_ERRORS = (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile, ET.ParseError,
                   DefusedXmlException, RecursionError)


def _describe(error: Exception) -> str:
    return str(error) if isinstance(error, EXPECTED_ERRORS) else f"unexpected {type(error).__name__}: {error}"


def _require_file(path: Path, limit: int) -> None:
    if not path.is_file():
        raise ValueError("source is not a readable regular file")
    if path.stat().st_size > limit:
        raise ValueError(f"source exceeds {limit} byte limit")


def _inventory(path: Path, fmt: ProjectFormat | None, source: Source, evidence: Evidence) -> None:
    if not (path.is_file() or path.is_dir()):
        raise ValueError("source is not a regular file or package directory")
    limit = fmt.byte_limit if fmt else MAX_ZIP_BYTES
    if path.is_file() and path.stat().st_size <= limit:
        source.sha256 = _hash(path)
    source.status = "unsupported"
    if fmt is None:
        message = f"{path.name}: unrecognized project format, inventoried only. {formats.GENERIC_GUIDANCE}"
    else:
        message = f"{path.name}: {fmt.label} files are inventoried, not parsed. {fmt.guidance}"
    evidence.diagnostics.append(Diagnostic("unsupported_project", "warning", message, source.id))


def build(mixdown: Path | None, optionals: list[tuple[str, Path]], declarations: Path | None,
          detail: Detail = "summary", analysis_backend: str = "builtin") -> Evidence:
    if analysis_backend != "builtin":
        raise ValueError(f"unknown analysis backend: {analysis_backend}")
    evidence = Evidence(limitations=[
        "Loudness and true peak follow ITU-R BS.1770-4 with every channel weighted 1.0; no surround channel map is read.",
        "Tempo and key estimates from audio are inferred, low-confidence candidates, not facts.",
        "MIDI and project files describe stored intent, not rendered sound.",
        "No semantic listening, instrument recognition, genre classification or artwork generation is performed.",
    ])
    if mixdown is None:
        evidence.diagnostics.append(Diagnostic("no_mixdown", "info",
            "No mixdown was supplied, so the rendered sound is unknown. Only stored intent and any stem measurements are available."))
    for role, path in [*([("mixdown", mixdown)] if mixdown else []), *optionals]:
        source_id = f"s{len(evidence.sources) + 1}"
        source = Source(source_id, path.name, role, role if role in AUDIO_ROLES else "unsupported", "ok", None)
        evidence.sources.append(source)
        try:
            if not path.exists():
                raise ValueError("source does not exist")
            if role in AUDIO_ROLES:
                from .adapters import audio  # Loaded on demand: numpy, scipy and libsndfile cost about 0.7 s.
                _require_file(path, MAX_AUDIO_BYTES)
                source.sha256 = _hash(path)
                staged = Evidence()
                audio.audio(path, source_id, staged, "mixdown" if role == "mixdown" else f"stem:{source_id}", detail, analysis_backend)
                _merge(evidence, staged)
                continue
            fmt = formats.BY_NAME[role] if role in FLAG_FORMATS else formats.resolve(path)
            if fmt is not None:
                source.adapter = fmt.name
            if fmt is None or fmt.adapter is None:
                _inventory(path, fmt, source, evidence)
                continue
            if fmt.kind == "file":
                _require_file(path, fmt.byte_limit)
                source.sha256 = _hash(path)
            staged = Evidence()
            fmt.adapter(path, source_id, staged, detail)
            from .project import attach_track_rhythm
            attach_track_rhythm(staged, source_id)
            _merge(evidence, staged)
        except Exception as error:  # Last-resort boundary: one hostile optional source must not lose the others.
            source.status = "error"
            if role == "mixdown":
                raise ValueError(f"required mixdown {path.name}: {_describe(error)}") from error
            evidence.diagnostics.append(Diagnostic("optional_source_error", "warning",
                f"Could not parse optional {role} {path.name}: {_describe(error)}", source_id))
    if declarations:
        try:
            _load_declarations(declarations, evidence)
        except Exception as error:
            source_id = f"s{len(evidence.sources) + 1}"
            evidence.sources.append(Source(source_id, declarations.name, "declarations", "json-declarations", "error", None))
            evidence.diagnostics.append(Diagnostic("optional_source_error", "warning",
                f"Could not parse declarations {declarations.name}: {_describe(error)}", source_id))
    detect_conflicts(evidence)
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="analyze.py", description="Extract local music evidence as JSON")
    parser.add_argument("--mixdown", type=Path, help="Rendered mixdown audio; without it only stored intent is analyzed")
    parser.add_argument("--midi", action="append", type=Path, default=[])
    parser.add_argument("--stem", action="append", type=Path, default=[], help="Stem audio; measured independently")
    parser.add_argument("--solna", action="append", type=Path, default=[])
    parser.add_argument("--dawproject", action="append", type=Path, default=[], help="DAWproject archive")
    parser.add_argument("--project", action="append", type=Path, default=[],
                        help="Project file or package; the format is detected from its extension and content")
    parser.add_argument("--declarations", type=Path, help="Optional JSON array of user-declared facts")
    parser.add_argument("--detail", choices=("summary", "full"), default="summary",
                        help="summary (default) reports one summary per note track; full also lists every note")
    parser.add_argument("--analysis-backend", choices=("builtin",), default="builtin",
                        help="Audio estimator backend (builtin only)")
    args = parser.parse_args(argv)
    optionals = [(kind, path) for kind in ("stem", "midi", "solna", "dawproject", "project")
                 for path in getattr(args, kind)]
    if args.mixdown is None and not optionals:
        parser.error("supply at least one input: --mixdown, --stem, --midi, --solna, --project or --dawproject")
    try:
        result = build(args.mixdown, optionals, args.declarations, args.detail, args.analysis_backend)
        print(json.dumps(result.as_json(), ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    except ValueError as error:
        print(f"music-guide: {error}", file=sys.stderr)
        return 2
