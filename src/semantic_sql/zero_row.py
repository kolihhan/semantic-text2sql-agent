from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
import re

from .catalog import DatabaseCatalog
from .grounding import build_value_index

_SQL_STRING_RE = re.compile(r"'((?:''|[^'])*)'")
_QUESTION_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_:./-]{2,}")
_STOPWORDS = {
    "the", "and", "for", "from", "what", "which", "whose", "with", "that",
    "this", "name", "list", "please", "provide", "indicate", "return", "where",
}


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _targets(question: str, sql: str) -> tuple[tuple[str, float], ...]:
    targets: list[tuple[str, float]] = []
    seen: set[str] = set()
    for raw in _SQL_STRING_RE.findall(sql):
        value = raw.replace("''", "'").strip()
        key = _normalize(value)
        if len(key) >= 3 and key not in seen:
            seen.add(key)
            targets.append((value, 20.0))
    for value in _QUESTION_TOKEN_RE.findall(question):
        key = _normalize(value)
        if key in _STOPWORDS or len(key) < 3 or key in seen:
            continue
        seen.add(key)
        targets.append((value, 0.0))
    return tuple(targets)


def _match_score(value: str, target: str, bonus: float) -> float:
    left, right = _normalize(value), _normalize(target)
    if not left or not right:
        return 0.0
    if left == right:
        return 100.0 + bonus
    if min(len(left), len(right)) >= 3 and (left in right or right in left):
        return 80.0 + 20.0 * min(len(left), len(right)) / max(len(left), len(right)) + bonus
    ratio = SequenceMatcher(None, left, right).ratio()
    if ratio >= 0.72:
        return 60.0 * ratio + bonus
    return 0.0


def build_zero_row_evidence(
    database: str | Path,
    *,
    question: str,
    sql: str,
    max_hits: int = 16,
    max_values_per_column: int = 256,
) -> str:
    """Return a small set of DB values relevant to an empty-result diagnosis.

    This intentionally runs only after a query has executed successfully with
    zero rows. It reuses the bounded value index and does not alter generation
    for normal non-empty queries.
    """
    if max_hits <= 0:
        return ""
    targets = _targets(question, sql)
    if not targets:
        return ""

    catalog = DatabaseCatalog.from_sqlite(database)
    index = build_value_index(catalog, max_values_per_column=max_values_per_column)
    scored: list[tuple[float, str, str, str, str]] = []
    for item in index:
        best_score = 0.0
        best_target = ""
        for target, bonus in targets:
            score = _match_score(item.value, target, bonus)
            if score > best_score:
                best_score, best_target = score, target
        if best_score:
            scored.append((best_score, item.table, item.column, item.value, best_target))

    scored.sort(key=lambda row: (-row[0], row[1], row[2], row[3].casefold()))
    lines = []
    seen: set[tuple[str, str, str]] = set()
    for score, table, column, value, target in scored:
        key = (table, column, _normalize(value))
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {table}.{column} = {value!r} (near {target!r}, score={score:.1f})")
        if len(lines) >= max_hits:
            break
    return "\n".join(lines)
