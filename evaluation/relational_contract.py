"""Schema-only relational constraints for experimental Qwen text-to-SQL grounding.

No LLM, question parsing, benchmark gold, value sampling, guessed joins,
or category-specific patches. Every claim follows from declared SQLite DDL.
Schema identifiers come from the database and are encoded as JSON strings.
Never used by the default SemanticSQLService unless an experiment explicitly
adds the returned context.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from urllib.parse import quote


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _names(values: tuple[str, ...]) -> str:
    return "(" + ", ".join(json.dumps(s, ensure_ascii=False) for s in values) + ")"


@dataclass(frozen=True)
class Key:
    table: str
    columns: tuple[str, ...]
    kind: str


@dataclass(frozen=True)
class Relation:
    source: str
    source_columns: tuple[str, ...]
    target: str
    target_columns: tuple[str, ...]
    target_key_declared: bool
    nullable: bool


@dataclass(frozen=True)
class RelationalMetadata:
    keys: tuple[Key, ...]
    relations: tuple[Relation, ...]
    ignored_partial_or_expression_indexes: int

    def render(self) -> str:
        if not self.keys and not self.relations:
            return ""
        rows = [
            "Declared relational constraints (physical SQLite metadata only; "
            "not inferred from names, values, or examples):"
        ]
        for key in self.keys:
            rows.append(
                f"- DECLARED {key.kind} {_q(key.table)}{_names(key.columns)}"
            )
        for rel in self.relations:
            mark = "declared_unique" if rel.target_key_declared else "unverified"
            rows.append(
                f"- FOREIGN_KEY {_q(rel.source)}{_names(rel.source_columns)}"
                f" -> {_q(rel.target)}{_names(rel.target_columns)}"
                f" target_key={mark} source_nullable={str(rel.nullable).lower()}"
            )
        rows.append(
            "These constraints describe possible joins and uniqueness, NOT which "
            "tables or columns answer the user's question. Use only when relevant."
        )
        return "\n".join(rows)


def inspect_constraints(
    database: str | Path, *, selected_tables: set[str] | None = None,
) -> RelationalMetadata:
    """Fail closed on unknown tables, partial/expression keys and missing FK columns."""
    uri = "file:" + quote(str(Path(database).resolve())) + "?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=15) as connection:
        connection.execute("PRAGMA query_only = ON")
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        physical = {str(row[0]) for row in rows}
        if selected_tables is None:
            selected = physical
        else:
            if not selected_tables <= physical:
                raise ValueError("selected table is absent in physical schema")
            selected = set(selected_tables)
        table_info: dict[str, list[tuple]] = {}
        uniques: dict[str, set[tuple[str, ...]]] = defaultdict(set)
        keys: list[Key] = []
        skipped = 0
        # Key inspection also covers non-selected referenced targets, so a
        # foreign key is not falsely marked unique due to schema pruning.
        for table in sorted(physical):
            cols = connection.execute(f"PRAGMA table_info({_q(table)})").fetchall()
            table_info[table] = cols
            primary = tuple(
                row[1] for row in sorted(cols, key=lambda row: row[5])
                if row[5] > 0
            )
            if primary:
                uniques[table].add(primary)
                if table in selected:
                    keys.append(Key(table, primary, "PRIMARY_KEY"))
            for idx in connection.execute(f"PRAGMA index_list({_q(table)})"):
                # PRAGMA index_list: sequence, name, is_unique, origin, partial
                if not idx[2]:
                    continue
                if len(idx) > 4 and idx[4]:
                    skipped += 1
                    continue
                components = sorted(
                    (row for row in connection.execute(
                        f"PRAGMA index_xinfo({_q(idx[1])})"
                    ) if row[5] == 1), key=lambda row: row[0]
                )
                if not components or any(
                    row[2] is None or row[1] < 0 for row in components
                ):
                    skipped += 1
                    continue
                names = tuple(str(row[2]) for row in components)
                if names in uniques[table]:
                    continue
                uniques[table].add(names)
                if table in selected:
                    keys.append(Key(table, names, "UNIQUE_KEY"))
        relations: list[Relation] = []
        for table in sorted(selected):
            grouped: dict[int, list[tuple]] = defaultdict(list)
            for fk in connection.execute(f"PRAGMA foreign_key_list({_q(table)})"):
                grouped[fk[0]].append(fk)
            for rows_for_fk in grouped.values():
                ordered = sorted(rows_for_fk, key=lambda f: f[1])
                target = str(ordered[0][2])
                if target not in selected or target not in physical:
                    continue
                src = tuple(str(r[3]) for r in ordered)
                dst_parts = [r[4] for r in ordered]
                if not all(c in {x[1] for x in table_info[table]} for c in src):
                    continue
                if any(not col for col in dst_parts):
                    primary = next(
                        (key for key in uniques[target] if
                         len(key) == len(src) and
                         any(k.kind == "PRIMARY_KEY" and k.table == target
                             and k.columns == key for k in keys)),
                        None,
                    )
                    if primary is None:
                        continue
                    dst = primary
                else:
                    dst = tuple(str(col) for col in dst_parts)
                if any(c not in {x[1] for x in table_info[target]} for c in dst):
                    continue
                source_notnull = {row[1]: bool(row[3]) for row in table_info[table]}
                target_unique = any(set(unique) <= set(dst) for unique in uniques[target])
                relations.append(Relation(
                    table, src, target, dst,
                    target_key_declared=target_unique,
                    nullable=any(not source_notnull[col] for col in src),
                ))
        return RelationalMetadata(
            tuple(keys), tuple(relations), skipped,
        )
