"""Schema-grounded relational evidence for XiYan Text-to-SQL (pre-registered v2).

No model, labels, SQL golds, or benchmark-specific identifiers in this module.
SQL is read-only. The intended mechanism is to make uniqueness, FK join
direction, composite keys and explicitly quoted value hits visible to the
generator *before SQL generation*, not ask a reviewer to rewrite a guess.

This is structural metadata, never a proof that the foreign-key constraint is
enforced or that a matching value expresses the user's intended predicate.
"""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import re
import sqlite3
from urllib.parse import quote


_TEXT_TYPES = ("CHAR", "CLOB", "TEXT", "VARCHAR")


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _readonly(db_path: Path) -> sqlite3.Connection:
    # SQLite query_only also protects against unexpected attached/mutating SQL.
    uri = "file:" + quote(str(db_path.resolve()), safe="/") + "?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=10)
    con.execute("PRAGMA query_only=ON")
    return con


def _key_sets(con: sqlite3.Connection, table: str) -> tuple[list[dict], list[tuple]]:
    rows = list(con.execute(f"PRAGMA table_info({_identifier(table)})"))
    cols = [str(x[1]) for x in rows]
    keys = []
    pk = tuple(str(row[1]) for row in sorted(rows, key=lambda r:r[5]) if row[5])
    if pk:
        keys.append({"columns":pk,"source":"primary_key","nullable":False})
    for index in con.execute(f"PRAGMA index_list({_identifier(table)})"):
        index_name, is_unique = str(index[1]), bool(index[2])
        is_partial = bool(index[4]) if len(index)>4 else False
        if not is_unique or is_partial:
            continue
        key_columns = [
            x for x in con.execute(f"PRAGMA index_xinfo({_identifier(index_name)})")
            if len(x) >= 6 and x[5] == 1
        ]
        if not key_columns or any(x[2] is None or int(x[1]) < 0 for x in key_columns):
            continue  # expression-based or invalid index: cannot prove physical key
        names = tuple(str(x[2]) for x in sorted(key_columns,key=lambda row:row[0]))
        if names not in {item["columns"] for item in keys}:
            nullable = any((not bool(row[3])) for row in rows if str(row[1]) in names)
            keys.append({"columns":names,"source":"unique_index","nullable":nullable})
    return keys, rows


def relational_metadata(database: str | Path, selected_tables: list[str]) -> dict:
    """Read physical keys and FK direction; no cardinality guesses from samples."""
    db=Path(database)
    if not db.is_file() or not selected_tables or len(selected_tables)!=len(set(selected_tables)):
        raise ValueError("existing DB and distinct selected tables required")
    with _readonly(db) as con:
        discovered={r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
        if any(name not in discovered for name in selected_tables):
            raise ValueError("selected table not present in SQLite")
        tables={}
        for table in sorted(selected_tables):
            keys, columns = _key_sets(con,table)
            tables[table]={"keys":keys,"columns":[{
                "name":str(r[1]),"type":str(r[2]).upper(),"not_null":bool(r[3]),
                "pk_position":int(r[5])
            } for r in columns]}
        foreign_keys=[]
        for source in sorted(selected_tables):
            groups=defaultdict(list)
            for r in con.execute(f"PRAGMA foreign_key_list({_identifier(source)})"):
                groups[int(r[0])].append(r)
            for gid,parts in sorted(groups.items()):
                ordered=sorted(parts,key=lambda x:int(x[1]))
                target=str(ordered[0][2])
                if target not in tables: continue
                source_cols=tuple(str(r[3]) for r in ordered)
                target_cols=tuple(str(r[4]) for r in ordered if r[4] is not None)
                if len(target_cols)!=len(source_cols):
                    pk=next((tuple(item["columns"]) for item in tables[target]["keys"]
                             if item["source"]=="primary_key"),())
                    if len(pk)!=len(source_cols): continue
                    target_cols=pk
                source_unique=any(set(item["columns"])==set(source_cols) and
                                  len(item["columns"])==len(source_cols)
                                  for item in tables[source]["keys"])
                target_unique=any(set(item["columns"])==set(target_cols) and
                                  len(item["columns"])==len(target_cols)
                                  for item in tables[target]["keys"])
                foreign_keys.append({
                    "source":source,"source_columns":source_cols,
                    "target":target,"target_columns":target_cols,
                    "source_unique_declared":source_unique,
                    "target_unique_declared":target_unique,
                    "id":gid,
                    "caution":"schema constraints only; enforcement and actual integrity unverified",
                })
    return {"version":"relational-metadata-v2","tables":tables,"foreign_keys":foreign_keys}


def _quoted_values(question: str, evidence: str) -> list[str]:
    """Only explicitly quoted text spans; no guessed synonyms or DB-specific terms."""
    candidates=[]
    for text in (question,evidence):
        for match in re.finditer(r"""(?<!\w)(?:'([^'\n]{2,48})'|"([^"\n]{2,48})")""", text):
            phrase=(match.group(1) or match.group(2)).strip()
            if phrase and phrase.casefold() not in {x.casefold() for x in candidates}:
                candidates.append(phrase)
    return candidates[:4]


def grounded_value_hits(database: str | Path, metadata: dict, *,
                        question: str, evidence: str = "",
                        max_probes: int = 24, max_hits: int = 4) -> dict:
    if not 1<=max_probes<=64 or not 0<=max_hits<=16:
        raise ValueError("bounded read-only value probe budget required")
    if max_hits==0:
        return {"quotes":[],"hits":[],"probes":0}
    phrases=_quoted_values(question,evidence)
    hits=[]
    probes=0
    with _readonly(Path(database)) as con:
        for phrase in phrases:
            for table,info in sorted(metadata["tables"].items()):
                for col in info["columns"]:
                    if probes >= max_probes or len(hits)>=max_hits:break
                    if not any(t in col["type"] for t in _TEXT_TYPES):continue
                    probes+=1
                    result=con.execute(
                        f"SELECT 1 FROM {_identifier(table)} "
                        f"WHERE {_identifier(col['name'])} = ? COLLATE NOCASE LIMIT 1",
                        (phrase,),
                    ).fetchone()
                    if result is not None:
                        hits.append({
                            "quoted_source_value":phrase,
                            "table":table,"column":col["name"],
                            "match":"exact_case_insensitive",
                            "claim":"value exists; NL predicate binding unknown",
                        })
                if probes >= max_probes or len(hits)>=max_hits:break
            if probes >= max_probes or len(hits)>=max_hits:break
    return {"quotes":phrases,"hits":hits,"probes":probes}


def enrich_mschema(*, base_mschema: str, database: str | Path,
                   selected_tables: list[str], question: str,
                   evidence: str = "") -> tuple[str,dict]:
    """Monotonic schema-context enrichment; never remove baseline M-Schema columns."""
    if not base_mschema.strip():
        raise ValueError("nonempty frozen M-Schema required")
    meta=relational_metadata(database,selected_tables)
    matches=grounded_value_hits(database,meta,question=question,evidence=evidence)
    lines=["Relational grounding (verified SQLite declarations, NOT a query answer):"]
    for table,info in sorted(meta["tables"].items()):
        for key in info["keys"]:
            labels=", ".join(_identifier(x) for x in key["columns"])
            lines.append(f"- {table}: {key['source']}({labels})"
                         + (" [nullable unique keys do not prevent duplicate NULLs]" if key["nullable"] else ""))
    for fk in meta["foreign_keys"]:
        source=", ".join(_identifier(x) for x in fk["source_columns"])
        target=", ".join(_identifier(x) for x in fk["target_columns"])
        if fk["source_unique_declared"] and fk["target_unique_declared"]:
            relation="declared 1-to-1 for matched non-NULL keys"
        elif fk["target_unique_declared"]:
            relation="declared many-to-1 for matched non-NULL keys"
        else:
            relation="cardinality UNKNOWN (target uniqueness not proven)"
        lines.append(
            f"- FK {fk['source']}({source}) -> {fk['target']}({target}): {relation}; "
            "actual FK integrity not checked"
        )
    if matches["hits"]:
        lines.append("Literal-value evidence from explicitly quoted words (existence ONLY):")
        for hit in matches["hits"]:
            lines.append(f"- {hit['table']}.{hit['column']} has an exact case-insensitive match for "
                         + repr(hit["quoted_source_value"])
                         + "; intended predicate remains unknown")
    output=base_mschema+"\n"+"\n".join(lines)
    diagnostics={
        "schema":"relational-grounding-v2",
        "physical_tables":len(meta["tables"]),
        "verified_fk_declarations":len(meta["foreign_keys"]),
        "verified_unique_keys":sum(len(x["keys"]) for x in meta["tables"].values()),
        "literal_hits":len(matches["hits"]),
        "value_probes":matches["probes"],
        "preserves_original_mschema":output.startswith(base_mschema+"\n"),
        "content_sha256":sha256(output.encode("utf-8")).hexdigest(),
    }
    return output,diagnostics
