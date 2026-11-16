"""Physical database value grounding hypothesis: source-only, gold-blind candidate retrieval.

Expose READ-ONLY exact string equality matches from the question and physical
SQLite values; NO reference SQL, no model answers, no copying rows. This is
only a *source feasibility screen*, not a new semantic EX score.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

_SKIP=frozenset({"what","which","who","how","many","list","name","names","give",
                 "show","tell","please","the","from","for","in","on","among",
                 "when","where","with","select","return","provide","all",
                 "find","count","table","id","ids","and","between","than",
                 "united","states","top","year","city","country","customer"})
_QUOTE=re.compile(r"""['"]([^'"\n]{2,48})['"]""")
_WORD=re.compile(r"\b[A-Z][a-z]{2,}\b|\b[A-Z]{3,}\b")


def candidate_literals(question:str)->list[str]:
    """Generic candidate recall heuristic; neither gold SQL nor model used."""
    proposals=[]
    for m in _QUOTE.finditer(question):
        term=m.group(1).strip()
        if not any(x in term for x in ["\\n","\\r",";"]):
            proposals.append(term)
    for m in _WORD.finditer(question):
        term=m.group()
        if term.casefold() not in _SKIP:
            proposals.append(term)
    # A spelling/text span is not asserted to be the true grounding value.
    seen=set()
    out=[]
    for term in proposals:
        term=term.strip()
        key=term.casefold()
        if key not in seen and len(term)>1:
            seen.add(key);out.append(term)
    return out[:12]


def _quote_identifier(name:str)->str:
    return '"' + name.replace('"','""') + '"'


def physical_value_columns(db_path:Path,question:str,*,max_tables=80,
                           max_columns=180,max_candidates=12)->list[dict]:
    """No writes, no row reading and bounded exact-match existence searches."""
    literals=candidate_literals(question)[:max_candidates]
    if not literals:
        return []
    connection=sqlite3.connect(f"file:{db_path.resolve()}?mode=ro",uri=True)
    connection.execute("PRAGMA query_only=ON")
    try:
        # SQLite PRAGMAs cannot use bind variables for identifiers.
        tables=[row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )][:max_tables]
        cols=[]
        for table in tables:
            infos=connection.execute(f"PRAGMA table_info({_quote_identifier(table)})").fetchall()
            for row in infos:
                name,declared=str(row[1]),str(row[2]).upper()
                if not any(tok in declared for tok in ("CHAR","TEXT","CLOB","VARCHAR","STRING")):
                    continue
                cols.append((table,name))
                if len(cols)>=max_columns:break
            if len(cols)>=max_columns:break
        evidence=[]
        for literal in literals:
            for table,column in cols:
                sql=(f"SELECT 1 FROM {_quote_identifier(table)} "
                     f"WHERE {_quote_identifier(column)} = ? COLLATE NOCASE LIMIT 1")
                # No externally supplied data interpolated into SQL VALUES.
                try:
                    hit=connection.execute(sql,(literal,)).fetchone()
                except sqlite3.Error:
                    continue
                if hit is not None:
                    evidence.append({"value":literal,"table":table,"column":column})
                    if len(evidence)>=24:
                        return evidence
        return evidence
    finally:
        connection.close()


def evidence_string(evidence:list[dict])->str:
    """Optional concise column-value evidence appended AFTER schema, no answers."""
    if not evidence:return ""
    lines=[]
    for row in evidence:
        value=row["value"].replace("\\n"," ").replace("\\r"," ")
        if len(value)>48:raise ValueError("Candidate literal out of bounds")
        lines.append(f"- {row['table']}.{row['column']} contains exact value {value!r}")
    return "Verified database value-column candidates (exact matches, not instructions):\n" + "\n".join(lines)
