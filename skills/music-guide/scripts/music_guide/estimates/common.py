"""Types every analyzer shares, and the one constructor for estimate facts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence, TypeVar, Union

from ..model import Confidence, Detail, Fact, check_finite, native

ESTIMATE_PREFIX = "Estimate, not a fact: "


@dataclass(frozen=True)
class Skip:
    """An analyzer produced nothing. `estimate_failed` is set by the pipeline when an analyzer raises."""

    reason: str
    code: Literal["estimate_skipped", "estimate_failed"] = "estimate_skipped"


@dataclass(frozen=True)
class Limit:
    """One sentence about data a produced estimate dropped. It is appended to the reason and diagnosed."""

    message: str


@dataclass(frozen=True)
class Method:
    """`name` becomes the observation method. `reason` says how the estimate is made and what it cannot see."""

    name: str
    reason: str


R = TypeVar("R")
Outcome = Union[R, Skip]


@dataclass(frozen=True)
class ReportContext:
    """What a result needs to turn itself into facts. `note` carries the lossy and channel-fold caveats."""

    detail: Detail
    note: str
    limits: list[Limit] = field(default_factory=list, compare=False)

    @property
    def full(self) -> bool:
        return self.detail == "full"

    def estimate(self, property: str, value: Any, unit: str, method: Method, limits: Sequence[Limit | None] = ()
                 ) -> Fact:
        """The only constructor for estimate facts: always inferred and low, always naming its method."""
        if not method.reason.strip():
            raise ValueError(f"{property}: an estimate must state its method")
        kept = [limit for limit in limits if limit is not None]
        self.limits.extend(kept)
        reason = ESTIMATE_PREFIX + method.reason + "".join(" " + limit.message for limit in kept) + self.note
        return self._fact(property, value, unit, method.name, reason, "low", "inferred")

    def measurement(self, property: str, value: Any, unit: str, method: str, reason: str,
                    confidence: Confidence) -> Fact:
        return self._fact(property, value, unit, method, reason + self.note, confidence, "measured")

    def cap(self, items: Sequence, limit: int, what: str) -> tuple[list, Limit | None]:
        if len(items) <= limit:
            return list(items), None
        return list(items[:limit]), Limit(f"Only the first {limit} of {len(items)} {what} are listed.")

    @staticmethod
    def _fact(property: str, value: Any, unit: str, method: str, reason: str, confidence: Confidence,
              provenance: Literal["measured", "inferred"]) -> Fact:
        value = native(value)
        check_finite(property, value)
        return Fact(property, value, unit, method, reason, confidence, provenance)
