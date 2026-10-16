"""ProjectFacts variant for evidence whose meaning is only partly confirmed."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..model import Confidence
from ..project import ProjectFacts


@dataclass(frozen=True)
class GradedFacts(ProjectFacts):
    """Emits the shared project vocabulary at a fixed confidence, with `caveat` appended to every reason."""

    grade: Confidence = field(default="low", kw_only=True)
    caveat: str = field(default="", kw_only=True)

    def _add(self, property: str, value: Any, unit: str, locator: str, reason: str, *,
             scope: str | None = None, time_domain: str = "static", confidence: Confidence = "high") -> None:
        super()._add(property, value, unit, locator, f"{reason} {self.caveat}".strip(), scope=scope,
                     time_domain=time_domain, confidence=self.grade)
