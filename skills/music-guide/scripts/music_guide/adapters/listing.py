from __future__ import annotations

from typing import Sequence, TypeVar

from ..model import Diagnostic, Evidence

T = TypeVar("T")
MAX_LISTED = 5000
# Full detail lists notes one by one, so a dense track needs a larger but still finite bound.
MAX_NOTE_LISTING = 50_000


def truncation(what: str, listed: int, total: int, source_id: str) -> Diagnostic:
    return Diagnostic("project_list_truncated", "warning",
                      f"Reported the first {listed} of {total} {what}; the rest were not listed.", source_id)


def capped(items: Sequence[T], what: str, source_id: str, evidence: Evidence, limit: int = MAX_LISTED) -> Sequence[T]:
    """Keep the first `limit` items so a hostile file cannot inflate the report, and say what was dropped."""
    if len(items) <= limit:
        return items
    evidence.diagnostics.append(truncation(what, limit, len(items), source_id))
    return items[:limit]
