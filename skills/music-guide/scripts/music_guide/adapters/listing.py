from __future__ import annotations

from typing import Sequence, TypeVar

from ..model import Diagnostic, Evidence

T = TypeVar("T")
MAX_LISTED = 5000


def capped(items: Sequence[T], what: str, source_id: str, evidence: Evidence, limit: int = MAX_LISTED) -> Sequence[T]:
    """Keep the first `limit` items so a hostile file cannot inflate the report, and say what was dropped."""
    if len(items) <= limit:
        return items
    evidence.diagnostics.append(Diagnostic("project_list_truncated", "warning",
        f"Reported the first {limit} of {len(items)} {what}; the rest were not listed.", source_id))
    return items[:limit]
