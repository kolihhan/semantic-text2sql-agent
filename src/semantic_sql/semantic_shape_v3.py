"""Experimental zero-training, intent-shape semantic correction for simple COUNT.

Unlike v2, this addresses the projection/aggregation failure mode. A model
can produce executable SELECT rows for an unambiguous row-count question.
Only act where the question's count unit matches the single physical table,
with one plain nonaggregate output, no joins / grouping / ranking / distinct,
and where the rewritten SELECT COUNT(*) passes read-only preflight plus an
isolated synthetic counterexample.

This is NOT a general semantic verifier or a claim that counting the table's
rows always means counting user entities. Do not promote without an
independent paired BIRD evaluation for regressions, coverage and accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3

from .semantic_v2 import _physical_columns_readonly, _quote_identifier
from .verifier import verify_sql_preflight


@dataclass(frozen=True)
class SemanticShapeDecision:
    sql: str
    changed: bool
    reason: str
    evidence: tuple[str, ...] = ()


_COUNT_INTENT = re.compile(
    r"^\s*(?:how many|what(?:'s| is) the (?:total )?number of|"
    r"give (?:me )?the number of|find the number of)\b",
    flags=re.I,
)
_CONFLICTING = re.compile(
    r"\b(?:distinct|different|unique|various|per|each|group(?:ed)?|"
    r"by|average|mean|median|percent(?:age)?|ratio|"
    r"top|first|last|highest|lowest|maximum|minimum|"
    r"times|ways|types|kinds)\b",
    flags=re.I,
)


def _unit_explicitly_matches(question: str, table: str) -> bool:
    """Only count physical table rows when question explicitly names that unit."""
    words = re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", table))
    if not words or len(words)>3 or words[0] in {"data","info","log","tmp","tbl"}:
        return False
    phrase=" ".join(w.casefold() for w in words)
    variants={phrase}
    if words[-1].endswith("ies"):
        variants.add(" ".join([*words[:-1], words[-1][:-3]+"y"]).casefold())
    elif words[-1].endswith("s") and len(words[-1])>3:
        variants.add(" ".join([*words[:-1], words[-1][:-1]]).casefold())
    else:
        variants.add(" ".join([*words[:-1], words[-1]+"s"]).casefold())
    intent=_COUNT_INTENT.match(question)
    if not intent:
        return False
    rest=re.sub(r"[^a-z0-9]+"," ",question[intent.end():].casefold()).strip()
    # Restrict to the count unit immediately following the quantifier.
    # "How many people attended events?" does NOT request event row count.
    return any(
        bool(re.match(r"^(?:[a-z]+ )?"+re.escape(v)+r"(?:\s|$)",rest))
        for v in variants if len(v)>=3
    )


def _witness(original: str, proposed: str,
             table: str, columns: tuple[str, ...]) -> bool:
    if not 1<=len(columns)<=128 or len({x.casefold() for x in columns})!=len(columns):
        return False
    db=sqlite3.connect(":memory:")
    try:
        db.execute(
            f"CREATE TABLE {_quote_identifier(table)} ("+
            ", ".join(f"{_quote_identifier(c)} TEXT" for c in columns)+")"
        )
        names=", ".join(_quote_identifier(c) for c in columns)
        binds=", ".join("?" for _ in columns)
        for index in range(2):
            values=[f"synthetic_{index}_{colindex}" for colindex,_ in enumerate(columns)]
            db.execute(
                f"INSERT INTO {_quote_identifier(table)} ({names}) VALUES ({binds})",
                values,
            )
        db.execute("PRAGMA query_only = ON")
        before=db.execute(original).fetchmany(3)
        after=db.execute(proposed).fetchmany(3)
        # The expected COUNT output must be exactly one numeric scalar.
        return (before != after and len(after)==1 and len(after[0])==1
                and isinstance(after[0][0], int))
    except sqlite3.Error:
        return False
    finally:
        db.close()


def evidence_gated_count_projection(*,database:str|Path,question:str,sql:str
                                   ) -> SemanticShapeDecision:
    def keep(reason:str)->SemanticShapeDecision:
        return SemanticShapeDecision(sql=sql,changed=False,reason=reason)
    if not _COUNT_INTENT.search(question):
        return keep("no_unambiguous_count_intent")
    if _CONFLICTING.search(question):
        return keep("count_intent_requires_distinct_or_complex_grain")
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError:
        return keep("sql_ast_unavailable")
    try:
        parsed=sqlglot.parse_one(sql,read="sqlite")
    except (ValueError,TypeError,sqlglot.errors.ParseError):
        return keep("sql_ast_unavailable")
    if not isinstance(parsed,exp.Select):
        return keep("not_simple_select")
    if (len(list(parsed.find_all(exp.Select)))!=1
        or len(list(parsed.find_all(exp.Table)))!=1
        or parsed.args.get("joins")
        or any(parsed.args.get(k) is not None
               for k in ("with_","group","having","order","limit","offset",
                         "distinct","qualify","cluster","distribute","sort"))
        or parsed.find(exp.Window) or parsed.find(exp.AggFunc)
        or parsed.find(exp.Subquery) or parsed.find(exp.Union)
        or parsed.find(exp.CTE) or parsed.find(exp.Anonymous)):
        return keep("sql_aggregation_grain_unsafe")
    frm=parsed.args.get("from_")
    table_ast=frm.this if frm else None
    if not isinstance(table_ast,exp.Table) or table_ast.args.get("db") or table_ast.args.get("catalog"):
        return keep("not_one_physical_table")
    if not _unit_explicitly_matches(question,table_ast.name):
        return keep("question_count_unit_not_physical_table")
    if len(parsed.expressions)!=1:
        return keep("multiple_output_columns")
    projection=parsed.expressions[0]
    if isinstance(projection,exp.Alias):
        projection=projection.this
    if not isinstance(projection,(exp.Column,exp.Star)):
        return keep("complex_or_nonrow_projection")
    physical=_physical_columns_readonly(database,table_ast.name)
    if physical is None:
        return keep("physical_table_not_verified")
    table,columns=physical
    if not verify_sql_preflight(database,sql).ok:
        return keep("original_sql_not_verified")
    new=parsed.copy()
    new.set("expressions",[exp.Count(this=exp.Star())])
    candidate=new.sql(dialect="sqlite")
    if candidate.strip()==sql.strip() or not verify_sql_preflight(database,candidate).ok:
        return keep("counterpart_unavailable_or_preflight_failed")
    if not _witness(sql,candidate,table,columns):
        return keep("independent_synthetic_counterexample_failed")
    return SemanticShapeDecision(
        sql=candidate,changed=True,
        reason="explicit_single_table_count_unit_projection_shape_mismatch",
        evidence=(
            "question_explicitly_asks_count_of_physical_table_entities",
            "simple_single_table_ast_without_join_group_or_distinct",
            "only_projection_rewritten_to_count_star",
            "original_and_revised_both_pass_readonly_preflight",
            "independent_two_row_synthetic_counterexample_discriminates_output",
        ),
    )
