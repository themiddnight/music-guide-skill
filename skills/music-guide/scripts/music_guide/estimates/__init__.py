"""The analyzer pipeline: a straight-line `analyze` whose call arguments are the dependency graph.

A positional argument is a required input: when it is a Skip, the analyzer is skipped and says which input was
missing. A keyword argument is optional and reaches the analyzer as `result | Skip`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from ..model import Diagnostic, Fact
from . import beats, harmony
from .common import Outcome, ReportContext, Skip
from .store import FeatureStore

Role = Literal["mixdown", "stem"]


@dataclass(frozen=True)
class Backend:
    name: str
    beat_grid: Callable[..., Any] = beats.beat_grid
    meter: Callable[..., Any] = beats.meter


BACKENDS = {"builtin": Backend("builtin")}


def analyze(store: FeatureStore, backend: Backend, role: Role) -> list[tuple[str, Outcome]]:
    """Every analyzer's outcome in dependency order, each with the label its diagnostics use."""
    outcomes: list[tuple[str, Outcome]] = []

    def run(label: str, analyzer: Callable[..., Any], *required: Any, **optional: Any) -> Outcome:
        missing = next((item for item in required if isinstance(item, Skip)), None)
        if missing is not None:
            upstream = next(name for name, outcome in outcomes if outcome is missing)
            outcome: Outcome = Skip(f"needs the {upstream}, which was not produced: {missing.reason}")
        else:
            try:
                outcome = analyzer(*required, **optional)
            # Isolation boundary: one analyzer defect, MemoryError included, must not lose the other estimates.
            except Exception as error:
                outcome = Skip(f"the analyzer failed with {type(error).__name__}: {error}", "estimate_failed")
        outcomes.append((label, outcome))
        return outcome

    run("pitch class profile", harmony.pitch_profile, store)
    tempo = run("tempo estimate", beats.tempo, store)
    run("key estimate", harmony.key, store)
    grid = run("beat grid", backend.beat_grid, store, tempo)
    if role == "mixdown":
        run("meter estimate", backend.meter, store, grid)
        run("swing estimate", beats.swing, store, grid)
    return outcomes


def render(outcomes: list[tuple[str, Outcome]], ctx: ReportContext) -> tuple[list[Fact], list[Diagnostic]]:
    facts: list[Fact] = []
    diagnostics: list[Diagnostic] = []
    for label, outcome in outcomes:
        if isinstance(outcome, Skip):
            severity: Literal["info", "warning"] = "warning" if outcome.code == "estimate_failed" else "info"
            diagnostics.append(Diagnostic(outcome.code, severity, f"No {label}: {outcome.reason}"))
            continue
        try:
            report = outcome.report(ctx)
        # Isolation boundary: a reporting defect, a nonfinite value included, loses only this estimate.
        except Exception as error:
            diagnostics.append(Diagnostic("estimate_failed", "warning",
                                          f"No {label}: reporting failed with {type(error).__name__}: {error}"))
            continue
        facts.extend(report.facts)
        diagnostics.extend(Diagnostic("estimate_truncated", "warning",
                                      f"The {label} ({limit.property}) was cut. {limit.message}")
                           for limit in report.limits)
    return facts, diagnostics
