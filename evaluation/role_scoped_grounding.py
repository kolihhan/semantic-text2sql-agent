"""Gold-blind physical-value evidence gated by QUESTION ROLE and SCHEMA ROLE.

Previous source-disjoint BIRD TRAIN16 exact-value-column hints regressed 9→5/16
because identical text referred to different roles (title vs ingredient, race
vs surname). This new SOURCE-ONLY, deterministic, conservative gate only adds
evidence when nearby question role words can be linked to a physical column
or single-component table name. No gold SQL, model outputs, teachers, or writes.

Positive lexical role evidence is NOT semantic proof; proper SQL EX still needs
prospective pinned-model physical-DB-independent validation.
"""
from __future__ import annotations

from collections import defaultdict
import re
from pathlib import Path
from evaluation.physical_value_grounding import (
    physical_value_columns, candidate_literals,
)

STOP = frozenset({
    "what","which","who","when","where","whose","many","list","find",
    "give","show","from","the","this","that","these","those","with",
    "have","does","did","was","were","had","for","are","and","its",
    "all","some","any","most","least","first","last","more","less",
    "many","number","numbers","total","count","names","values","data",
    "record","records","selected","return","using","between","among",
    "than","each","per","above","below","please","explain",
})
GENERIC_COLUMN = frozenset({
    "name","id","value","text","description","details","label","title",
    "first","last","key","code","type",
})
WORD = re.compile(r"[a-zA-Z][a-zA-Z0-9]*")


def norm(s: str) -> str:
    s=s.lower()
    if s.endswith("ies") and len(s)>5:
        s=s[:-3]+"y"
    elif s.endswith("es") and len(s)>5:
        s=s[:-2]
    elif s.endswith("s") and len(s)>4 and not s.endswith("ss"):
        s=s[:-1]
    return s


def identifier_tokens(name: str) -> list[str]:
    name=re.sub(r"([a-z])([A-Z])",r"\1 \2",str(name))
    return [norm(x) for x in WORD.findall(name)]


def nearby_tokens(question: str, literal: str, *, window: int=7) -> set[str]:
    """Only words near an actually mentioned literal are eligible role cues."""
    q=str(question)
    m=re.search(r"(?<!\w)"+re.escape(literal)+r"(?!\w)",q,re.I)
    if m is None:
        return set()
    before=WORD.findall(q[:m.start()])[-window:]
    after=WORD.findall(q[m.end():])[:window]
    return {
        norm(t) for t in before+after
        if norm(t) not in STOP and len(norm(t))>=4
    }


def related_word(a: str,b: str) -> bool:
    """Generic 4-letter prefix allows professor↔prof, not race↔surname."""
    return a==b or (min(len(a),len(b))>=4 and
                    (a.startswith(b) or b.startswith(a)))


def lexical_role_match(question: str, literal: str, table: str, column: str) -> dict|None:
    roles=nearby_tokens(question,literal)
    if not roles:
        return None
    # Generic noun-phrase referent disambiguation: in "ingredients of recipe
    # 'Apricot'", the named value describes the recipe, not ingredients.
    # A directly specified referent head after of/for/named/called/titled
    # overrides more distant question nouns; no hardcoded schema vocabulary.
    mention=re.search(r"(?<!\w)"+re.escape(literal)+r"(?!\w)",question,re.I)
    if mention is not None:
        prefix=question[:mention.start()].rstrip().rstrip(chr(39)+chr(34)).rstrip()
        head=re.search(r"(?i)\b(?:of|for|named|called|titled)\s+(?:the|an|a)?\s*([a-z]{4,})\s*$",prefix)
        if head is not None:
            roles={norm(head.group(1))}
    column_parts=identifier_tokens(column)
    table_parts=identifier_tokens(table)
    # Prefer non-generic column role. Simply sharing 'name', 'id', or
    # 'title' with the question is too weak to disambiguate entities.
    column_role=[c for c in column_parts if c not in GENERIC_COLUMN
                 and len(c)>=4]
    column_hits=sorted({
        r for r in roles for c in column_role if related_word(r,c)
    })
    if column_hits:
        return {"cue":column_hits[0],"basis":"column"}
    # Profession/entity table names are useful for name/ID-valued columns,
    # but compound tables such as "recipe_ingredients" are NOT accepted just
    # because the question said 'recipe' when value is an ingredient.
    if len(table_parts)==1 and any(
            related_word(r,table_parts[0]) for r in roles):
        table_hit=sorted(
            r for r in roles if related_word(r,table_parts[0]))
        if table_hit:
            return {"cue":table_hit[0],"basis":"single_entity_table"}
    return None


def role_scoped_candidates(question: str, evidence: list[dict]) -> list[dict]:
    accepted=[]
    for row in evidence:
        match=lexical_role_match(question,row["value"],row["table"],row["column"])
        if match is None:
            continue
        accepted.append({**row,**match})
    # Only unique role-scoped physical field for the same literal. Ambiguous
    # matches are NOT proof: do not supply hints that could misdirect Qwen.
    grouped=defaultdict(list)
    for r in accepted:
        grouped[r["value"].casefold()].append(r)
    unique=[]
    for _,rows in grouped.items():
        cells={(r["table"],r["column"]) for r in rows}
        if len(cells)!=1:
            continue
        unique.extend(rows[:1])
    return unique[:8]


def evidence_string(evidence: list[dict]) -> str:
    if not evidence:
        return ""
    lines=[]
    for e in evidence:
        v=str(e["value"]).replace("\r"," ").replace("\n"," ")
        if len(v)>48:
            raise ValueError("Candidate literal too long")
        lines.append(
            "- "+str(e["table"])+"."+str(e["column"])+" contains "
            +repr(v)+"; matches nearby question role "+repr(e["cue"])
            +" ("+str(e["basis"])+")"
        )
    return (
        "Physical source values with lexical question-role/column matches "
        "(potential evidence only; do not assume a question filter or join):\n"
        +"\n".join(lines)
    )


def source_only_role_evidence(db: Path, question: str) -> list[dict]:
    hits=physical_value_columns(db,question)
    return role_scoped_candidates(question,hits)
