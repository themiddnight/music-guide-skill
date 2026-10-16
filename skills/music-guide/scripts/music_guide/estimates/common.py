"""Types every analyzer shares, and the one constructor for estimate facts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, Sequence, Union

from ..model import Confidence, Detail, Fact, check_finite, native

ESTIMATE_PREFIX = "Estimate, not a fact: "
FULL_LIMIT = 50_000


@dataclass(frozen=True)
class Skip:
    """An analyzer produced nothing. `estimate_failed` is set by the pipeline when an analyzer raises."""

    reason: str
    code: Literal["estimate_skipped", "estimate_failed"] = "estimate_skipped"


@dataclass(frozen=True)
class Limit:
    """One sentence about data an estimate dropped, kept with the property it was dropped from."""

    property: str
    message: str


@dataclass(frozen=True)
class Method:
    """`name` becomes the observation method. `reason` says how the estimate is made and what it cannot see."""

    name: str
    reason: str




@dataclass(frozen=True)
class Report:
    """What a result reports: its facts, and the cuts each fact made, which become truncation diagnostics."""

    facts: tuple[Fact, ...] = ()
    limits: tuple[Limit, ...] = ()

    def __add__(self, other: Report) -> Report:
        return Report(self.facts + other.facts, self.limits + other.limits)


@dataclass(frozen=True)
class ReportContext:
    """What a result needs to turn itself into facts. `note` carries the lossy and channel-fold caveats."""

    detail: Detail
    note: str

    @property
    def full(self) -> bool:
        return self.detail == "full"

    def estimate(self, property: str, value: Any, unit: str, method: Method, cuts: Sequence[str | None] = ()
                 ) -> Report:
        """The only constructor for estimate facts: always inferred and low, always naming its method.

        `cuts` are the sentences `cap` returned; each is appended to the reason and becomes a Limit on `property`.
        """
        if not method.reason.strip():
            raise ValueError(f"{property}: an estimate must state its method")
        kept = tuple(cut for cut in cuts if cut is not None)
        reason = ESTIMATE_PREFIX + method.reason + "".join(" " + cut for cut in kept) + self.note
        fact = self._fact(property, value, unit, method.name, reason, "low", "inferred")
        return Report((fact,), tuple(Limit(property, cut) for cut in kept))

    def measurement(self, property: str, value: Any, unit: str, method: str, reason: str,
                    confidence: Confidence) -> Report:
        return Report((self._fact(property, value, unit, method, reason + self.note, confidence, "measured"),))

    @staticmethod
    def cap(items: Sequence, limit: int, what: str) -> tuple[list, str | None]:
        """At most `limit` items, and the sentence saying how many were dropped when any were."""
        if len(items) <= limit:
            return list(items), None
        return list(items[:limit]), f"Only the first {limit} of {len(items)} {what} are listed."

    @staticmethod
    def _fact(property: str, value: Any, unit: str, method: str, reason: str, confidence: Confidence,
              provenance: Literal["measured", "inferred"]) -> Fact:
        value = native(value)
        check_finite(property, value)
        return Fact(property, value, unit, method, reason, confidence, provenance)


class Result(Protocol):
    """A produced estimate: anything that can report itself."""

    def report(self, ctx: ReportContext) -> Report: ...


Outcome = Union[Result, Skip]

