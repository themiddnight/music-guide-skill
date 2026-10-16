from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Provenance = Literal["measured", "extracted", "inferred", "user_declared"]
Confidence = Literal["high", "medium", "low"]
Detail = Literal["summary", "full"]


@dataclass
class Source:
    id: str
    basename: str
    role: str
    adapter: str
    status: Literal["ok", "error", "unsupported"]
    sha256: str | None


@dataclass
class Observation:
    id: str
    sourceIds: list[str]
    property: str
    value: Any
    unit: str
    scope: str
    timeDomain: str
    locator: str
    method: str
    methodVersion: str
    provenance: Provenance
    confidence: Confidence
    confidenceReason: str


@dataclass
class Conflict:
    observationIds: list[str]
    property: str
    scope: str
    unit: str
    timeDomain: str
    status: Literal["unresolved"] = "unresolved"
    reason: str = "Compatible observations disagree; inspect sources and timing."


@dataclass
class Diagnostic:
    code: str
    severity: Literal["info", "warning", "error"]
    message: str
    sourceId: str | None = None


MAX_INTEGER_BITS = 1024


def _check_finite(property: str, value: Any) -> None:
    """Reject values the JSON report cannot carry, in nested containers too."""
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{property} holds a nonfinite number")
        if isinstance(item, int) and item.bit_length() > MAX_INTEGER_BITS:
            raise ValueError(f"{property} holds an integer beyond the reportable range")
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)


@dataclass
class Evidence:
    sources: list[Source] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def add(self, source_id: str, property: str, value: Any, unit: str,
            scope: str, time_domain: str, locator: str, method: str,
            provenance: Provenance, confidence: Confidence,
            reason: str) -> None:
        _check_finite(property, value)
        self.observations.append(Observation(
            id=f"o{len(self.observations) + 1}", sourceIds=[source_id],
            property=property, value=value, unit=unit, scope=scope,
            timeDomain=time_domain, locator=locator, method=method,
            methodVersion="1", provenance=provenance, confidence=confidence,
            confidenceReason=reason))

    def as_json(self) -> dict[str, Any]:
        return {
            "schemaVersion": "1.1.0",
            "sources": [asdict(item) for item in self.sources],
            "observations": [asdict(item) for item in self.observations],
            "conflicts": [asdict(item) for item in self.conflicts],
            "diagnostics": [asdict(item) for item in self.diagnostics],
            "limitations": self.limitations,
        }


TEMPO_TOLERANCE_BPM = 0.01
# Inferred estimates are interpretations, so they never raise automatic conflicts.
COMPARABLE_PROVENANCE = {"measured", "extracted", "user_declared"}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _disagree(property: str, values: list[Any]) -> bool:
    # MIDI stores tempo in whole microseconds per beat, so 92 BPM reads back as 91.99999.
    if property == "tempo" and all(_is_number(value) for value in values):
        return max(values) - min(values) > TEMPO_TOLERANCE_BPM
    first = values[0]
    return any(isinstance(value, bool) != isinstance(first, bool) or value != first for value in values[1:])


def detect_conflicts(evidence: Evidence) -> None:
    shared_properties = {"tempo", "meter"}
    groups: dict[tuple[str, str, str, str], list[Observation]] = {}
    for item in evidence.observations:
        if (item.timeDomain != "static" or item.property not in shared_properties
                or item.provenance not in COMPARABLE_PROVENANCE):
            continue
        key = (item.property, item.scope, item.unit, item.timeDomain)
        groups.setdefault(key, []).append(item)
    for (property, scope, unit, domain), group in groups.items():
        if len(group) >= 2 and _disagree(property, [item.value for item in group]):
            evidence.conflicts.append(Conflict(
                observationIds=[item.id for item in group], property=property,
                scope=scope, unit=unit, timeDomain=domain))
