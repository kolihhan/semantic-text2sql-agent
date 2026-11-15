"""First-party published Spider FK metadata, validated against physical SQLite.

Spider tables.json schema is an input schema annotation, not a reference
SQL query and not a model response. Do not infer absent relationships.
"""
from __future__ import annotations

from dataclasses import dataclass

from evaluation.schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog


def published_foreign_keys(catalog: DatabaseCatalog, entry: dict) -> tuple[tuple[str,str,str,str], ...]:
    physical = {name.casefold(): name for name in catalog.tables}
    raw_tables = entry.get("table_names_original")
    raw_columns = entry.get("column_names_original")
    raw_pairs = entry.get("foreign_keys")
    if not isinstance(raw_tables,list) or not isinstance(raw_columns,list) or not isinstance(raw_pairs,list):
        raise ValueError("published original Spider FK schema fields missing")
    found = []
    for endpoints in raw_pairs:
        if not (isinstance(endpoints,list) and len(endpoints)==2
                and all(type(x) is int for x in endpoints)):
            raise ValueError("unverified FK pointer format")
        nodes=[]
        for index in endpoints:
            if not 0 <= index < len(raw_columns):
                raise ValueError("published FK index out of bounds")
            table_idx, column = raw_columns[index]
            if not type(table_idx) is int or not 0 <= table_idx < len(raw_tables):
                raise ValueError("invalid source table pointer")
            table = physical.get(str(raw_tables[table_idx]).casefold())
            if table is None:
                break
            columns = {name.casefold(): name for name in catalog.tables[table].columns}
            col = columns.get(str(column).casefold())
            if col is None:
                break
            nodes.append((table,col))
        if len(nodes) == 2 and nodes[0] != nodes[1]:
            found.append((nodes[0][0],nodes[0][1],nodes[1][0],nodes[1][1]))
    return tuple(sorted(set(found)))


def enrichment(catalog: DatabaseCatalog, entry: dict, root, dbid: str, question: str) -> tuple[str,str,dict]:
    original, original_diag = select_schema_context(catalog,root,dbid,question)
    published = published_foreign_keys(catalog,entry)
    old = {frozenset(((a,b),(c,d))) for a,b,c,d in catalog.foreign_keys}
    added = tuple(fk for fk in published
                  if frozenset(((fk[0],fk[1]),(fk[2],fk[3]))) not in old)
    if not added:
        return original,original,{"source_fk_total":len(published),"missing_physical_fk":0,"visible_new_fks":0,"added_tables":[]}
    # Same exact original deterministic selector. The only difference is
    # source-backed foreign keys made available as if declared in SQLite.
    augmented = DatabaseCatalog(
        database=catalog.database,
        tables=catalog.tables,
        foreign_keys=tuple(catalog.foreign_keys)+added
    )
    new_schema, new_diag = select_schema_context(augmented,root,dbid,question)
    added_fact_lines = set(new_schema.splitlines()) - set(original.splitlines())
    visible = [a for a,b,c,d in added
               if f"FK {a}.{b} -> {c}.{d}" in added_fact_lines]
    return original,new_schema,{
        "source_fk_total":len(published),
        "missing_physical_fk":len(added),
        "visible_new_fks":len(visible),
        "added_tables":sorted(
            set(new_diag["selected_tables"])-set(original_diag["selected_tables"])
        ),
        "baseline_fallback":original_diag["fallback_full_schema"],
        "augmented_fallback":new_diag["fallback_full_schema"],
    }
