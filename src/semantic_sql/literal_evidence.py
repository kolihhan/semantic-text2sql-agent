"""Read-only, exact evidence for *quoted* entities in a NL-to-SQL question.

This supplements the existing schema; it never changes a query or chooses
a column. It rejects guessed values and returns no negative evidence when a
probe is inconclusive (unindexed table, time budget or absent quoted span).
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import quote

from .catalog import DatabaseCatalog


def quoted_literals(question: str, *, limit: int = 4) -> tuple[str, ...]:
    """Only question-authored quoted strings; never infer aliases/synonyms."""
    if limit < 0:
        raise ValueError("negative literal budget")
    values: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"""(?<!\w)(?:'([^'\n]{2,64})'|"([^"\n]{2,64})")""", question):
        value = (match.group(1) or match.group(2)).strip()
        if value and value.casefold() not in seen:
            values.append(value)
            seen.add(value.casefold())
        if len(values) >= limit:
            break
    return tuple(values)


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def exact_literal_evidence(
    catalog: DatabaseCatalog,
    question: str,
    *,
    max_probes: int = 128,
    max_hits: int = 6,
    max_vm_steps_per_probe: int = 50_000,
) -> tuple[dict[str, str], ...]:
    """Read-only bounded SQLite equality probes, never fuzzy string matching.

    A found value is a fact about data, not proof of the user's intended join
    or filter. All input data goes through SQL parameters, not interpolation.
    """
    if max_probes < 0 or max_hits < 0 or max_vm_steps_per_probe <= 0:
        raise ValueError("invalid probe budget")
    literals = quoted_literals(question)
    if not literals or not max_probes or not max_hits:
        return ()
    uri = "file:" + quote(str(Path(catalog.database).resolve()), safe="/") + "?mode=ro"
    hits: list[dict[str, str]] = []
    used = 0
    with sqlite3.connect(uri, uri=True, timeout=5) as db:
        db.execute("PRAGMA query_only=ON")
        # A single unindexed large BIRD table should not stall the request.
        for table in sorted(catalog.tables):
            if used >= max_probes or len(hits) >= max_hits:
                break
            for row in db.execute(f"PRAGMA table_info({_q(table)})").fetchall():
                name, sql_type = str(row[1]), str(row[2] or "").upper()
                if (not any(t in sql_type for t in ("CHAR", "TEXT", "CLOB"))
                        and sql_type != ""):
                    continue
                if name not in catalog.tables[table].columns:
                    raise ValueError("catalog/schema drift during literal probe")
                for literal in literals:
                    if used >= max_probes or len(hits) >= max_hits:
                        break
                    used += 1
                    steps = 0

                    def budget() -> int:
                        nonlocal steps
                        steps += 1000
                        return int(steps > max_vm_steps_per_probe)

                    db.set_progress_handler(budget, 1000)
                    try:
                        exists = db.execute(
                            f"SELECT 1 FROM {_q(table)} WHERE {_q(name)} = ? COLLATE NOCASE LIMIT 1",
                            (literal,),
                        ).fetchone()
                    except sqlite3.OperationalError:
                        exists = None  # interrupted/inconclusive; never claim absent
                    finally:
                        db.set_progress_handler(None, 0)
                    if exists:
                        hits.append({"literal": literal, "table": table, "column": name})
    return tuple(hits)


def append_literal_evidence(
    catalog: DatabaseCatalog,
    question: str,
    *,
    schema_context: str,
    max_probes: int = 128,
) -> str:
    """Append verified physical value hits without removing any baseline schema."""
    hits = exact_literal_evidence(catalog, question, max_probes=max_probes)
    if not hits:
        return schema_context
    lines = [
        "Verified exact quoted-value hits (data existence ONLY, not intended predicate):"
    ]
    for hit in hits:
        lines.append(
            f"- {_q(hit['table'])}.{_q(hit['column'])} contains "
            f"{json.dumps(hit['literal'], ensure_ascii=False)} (case-insensitive equality); "
            "only use if it matches the user's requested relationship"
        )
    return schema_context + "\n" + "\n".join(lines)
