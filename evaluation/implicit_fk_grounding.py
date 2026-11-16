"""Reify SQLite implicit FK target columns using actual target PRIMARY KEY metadata.

SQLite PRAGMA foreign_key_list() reports "to" as NULL when REFERENCES parent
omits an explicit target-column list. This does NOT mean the FK is unknown:
its target is the referenced table's ordered primary-key columns.

Conservative experiment-only transform. Original catalog/runtime unchanged.
Ambiguous/invalid/unresolved keys are NOT fabricated. No access to task labels.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sqlite3

from semantic_sql.catalog import DatabaseCatalog


@dataclass(frozen=True)
class FKResolution:
    original_missing: int
    inferred: int
    unresolved: int
    original_explicit: int


def _identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def resolve_implicit_fk_targets(catalog: DatabaseCatalog) -> tuple[DatabaseCatalog, FKResolution]:
    """Return a fresh catalog with implicit SQLite FK "to" columns reconstructed.

    Groups PRAGMA rows by FK id, orders by seq, matches SQLite's implicit
    referenced primary key. Exact existing named targets stay untouched.
    If no exact PK arity agreement, omit only unresolved FK edge, never invent.
    """
    tables = catalog.tables
    con = sqlite3.connect(f"file:{catalog.database.as_posix()}?mode=ro", uri=True)
    try:
        pk: dict[str, tuple[str, ...]] = {}
        for name in tables:
            info = list(con.execute(f"PRAGMA table_info({_identifier(name)})"))
            numbered = sorted(((int(r[5]), str(r[1])) for r in info if int(r[5]) > 0))
            pk[name] = tuple(col for _pos, col in numbered)
        final: list[tuple[str, str, str, str]] = []
        counts: dict[str, int] = defaultdict(int)
        for source in sorted(tables):
            groups: dict[int, list[tuple]] = defaultdict(list)
            for record in con.execute(f"PRAGMA foreign_key_list({_identifier(source)})"):
                groups[int(record[0])].append(record)
            for _, records in sorted(groups.items()):
                records.sort(key=lambda row: int(row[1]))
                target = str(records[0][2])
                implicit = [row for row in records if row[4] is None]
                arity_matches = target in pk and len(pk[target]) == len(records)
                for row in records:
                    from_col = str(row[3])
                    if row[4] is not None:
                        final.append((source, from_col, target, str(row[4])))
                        counts["original_explicit"] += 1
                        continue
                    counts["original_missing"] += 1
                    if arity_matches and 0 <= int(row[1]) < len(pk[target]):
                        to_col = pk[target][int(row[1])]
                        final.append((source, from_col, target, to_col))
                        counts["inferred"] += 1
                    else:
                        counts["unresolved"] += 1
        return (
            DatabaseCatalog(database=Path(catalog.database),tables=dict(tables),foreign_keys=tuple(final)),
            FKResolution(
                original_missing=counts["original_missing"],
                inferred=counts["inferred"],
                unresolved=counts["unresolved"],
                original_explicit=counts["original_explicit"],
            ),
        )
    finally:
        con.close()
