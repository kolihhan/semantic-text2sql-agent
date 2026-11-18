"""TRAIN-origin, label-blind *quantifier* compatibility rule for SQL candidates.

This is an intentionally narrow falsifiable semantic probe, NOT general
question-to-SQL semantic verification. The executable selector only sees
question, candidate SQL, and already-known read-only execution validity.
It sees no publisher reference, EX labels, DEV500, or HOLDOUT300.

Rules frozen before checking 4-database BIRD TRAIN-VAL64 Qwen9B outcomes:
- Direct cardinality question -> output SELECT must compute COUNT or SUM.
- Direct arithmetic mean question -> output SELECT must compute AVG, an
  explicit ratio, or project an already named avg/average field.
On uncertain questions abstain (0). When a baseline executable candidate
violates an explicit contract and another executable candidate satisfies
it, constrain the voting pool to contract-satisfying alternatives.
No model calls, SQL rewriting, threshold search, learned model or teacher.
"""
from __future__ import annotations

import re
from typing import Any

from evaluation.kcache_protocol import majority_selector

COUNT_QUESTION = re.compile(
    r"^\s*(?:how\s+many\b|what\s+is\s+the\s+(?:total\s+)?number\s+of\b|"
    r"count\s+(?:the\s+)?(?:number\s+of\b)?)", re.I
)
AVERAGE_QUESTION = re.compile(
    r"^\s*(?:on\s+average\b|(?:what\s+is|what\s+was|"
    r"calculate|compute|find|give|show)\s+(?:the\s+)?"
    r"(?:overall\s+)?(?:average|mean)\b)", re.I
)
COUNT_OUTPUT = re.compile(r"\b(?:COUNT|SUM)\s*\(", re.I)
AVERAGE_OUTPUT = re.compile(r"\bAVG\s*\(|\b(?:AVG|AVERAGE)_[A-Za-z0-9_]+\b|"
                            r"\b[A-Za-z0-9_]+_(?:AVG|AVERAGE)\b", re.I)


def question_only(user_message: str) -> str:
    """Never treat the physical schema as part of the natural-language question."""
    first = user_message.split("\nSchema context:\n",1)[0]
    return first.removeprefix("Question: ").strip()


def contract_for_question(question: str) -> str:
    q = str(question).strip()
    # "On average, how many ..." asks a mean, not a raw cardinality.
    if AVERAGE_QUESTION.search(q):
        return "mean"
    if COUNT_QUESTION.search(q):
        # A compound request for both a count and a list needs a richer
        # multi-output contract. Do not force one aggregate-only SQL.
        if re.search(r"\b(?:list|show|give)\b.{0,55}"
                     r"\b(?:names|titles|identifiers|ids)\b",q,re.I):
            return "abstain"
        return "cardinality"
    return "abstain"


def _mask_sql_literals_and_comments(sql: str) -> str:
    """Retain SQL structure, not text inside quoted values or comments."""
    s = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    s = re.sub(r"--[^\r\n]*", " ", s)
    s = re.sub(r"'(?:''|[^'])*'", "''", s)
    return s


def _outer_projection(sql: str) -> str:
    """Conservative top-level SELECT/FROM extractor (never guess on ambiguity)."""
    s = _mask_sql_literals_and_comments(sql)
    depth = 0
    select_end = -1
    for m in re.finditer(r"\bSELECT\b|\bFROM\b|[()]",s,flags=re.I):
        token=m.group()
        if token=="(":
            depth+=1
        elif token==")":
            depth-=1
            if depth<0:return ""
        elif depth==0 and token.casefold()=="select":
            if select_end==-1:select_end=m.end()
        elif depth==0 and token.casefold()=="from" and select_end!=-1:
            return s[select_end:m.start()]
    if depth!=0:return ""
    return s[select_end:] if select_end!=-1 else ""


def compatibility(question: str, sql: str) -> int:
    """+1 compatible, -1 clear signature violation, 0 undecidable/abstain."""
    contract=contract_for_question(question)
    if contract=="abstain":
        return 0
    projection=_outer_projection(sql)
    if not projection:
        return 0
    if contract=="cardinality":
        return 1 if COUNT_OUTPUT.search(projection) else -1
    if contract=="mean":
        return 1 if (AVERAGE_OUTPUT.search(projection) or
                     re.search(r"(?<!/)/(?!/)",projection)) else -1
    return 0


def select_by_contract(question: str, sqls: list[str],
                       outcomes: list[dict[str,Any]]) -> dict[str,Any]:
    """Choose without reference SQL; fall back to original result plurality."""
    if not sqls or len(sqls)!=len(outcomes):
        raise ValueError("source candidate/outcome mismatch")
    original=majority_selector(outcomes)["index"]
    signal=[compatibility(question,sql) for sql in sqls]
    if signal[original]!=-1:
        return {"index":original,"original_index":original,"changed":False,
                "contract":contract_for_question(question),"signal":signal}
    valid_alternatives=[i for i,(x,s) in enumerate(zip(outcomes,signal))
                        if x.get("valid") and s==1]
    if not valid_alternatives:
        return {"index":original,"original_index":original,"changed":False,
                "contract":contract_for_question(question),"signal":signal}
    masked=[outcome if i in valid_alternatives else
            {"valid":False,"signature":None}
            for i,outcome in enumerate(outcomes)]
    chosen=majority_selector(masked)["index"]
    if chosen not in valid_alternatives:
        raise AssertionError("semantic selection violated executable alternative gate")
    return {"index":chosen,"original_index":original,"changed":chosen!=original,
            "contract":contract_for_question(question),"signal":signal}
