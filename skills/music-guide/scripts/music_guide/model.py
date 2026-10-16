"""Small, explicit evidence records shared by the local adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Provenance = Literal["measured", "extracted", "inferred", "user_declared"]
Confidence = Literal["high", "medium", "low"]


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
        self.observations.append(Observation(
            id=f"o{len(self.observations) + 1}", sourceIds=[source_id],
            property=property, value=value, unit=unit, scope=scope,
            timeDomain=time_domain, locator=locator, method=method,
            methodVersion="1", provenance=provenance, confidence=confidence,
            confidenceReason=reason))

    def as_json(self) -> dict[str, Any]:
        return {
            "schemaVersion": "1.0.0",
            "sources": [asdict(item) for item in self.sources],
            "observations": [asdict(item) for item in self.observations],
            "conflicts": [asdict(item) for item in self.conflicts],
            "diagnostics": [asdict(item) for item in self.diagnostics],
            "limitations": self.limitations,
        }


def detect_conflicts(evidence: Evidence) -> None:
    """Compare static claims in the same property, scope, unit and time domain."""
    shared_properties = {"tempo", "meter"}
    groups: dict[tuple[str, str, str, str], list[Observation]] = {}
    for item in evidence.observations:
        # File inventories and event streams describe distinct sources, not shared claims.
        if item.timeDomain != "static" or item.property not in shared_properties:
            continue
        key = (item.property, item.scope, item.unit, item.timeDomain)
        groups.setdefault(key, []).append(item)
    for (property, scope, unit, domain), group in groups.items():
        if len(group) < 2:
            continue
        first = group[0].value
        if any((isinstance(item.value, bool) != isinstance(first, bool)) or item.value != first
               for item in group[1:]):
            evidence.conflicts.append(Conflict(
                observationIds=[item.id for item in group], property=property,
                scope=scope, unit=unit, timeDomain=domain))
