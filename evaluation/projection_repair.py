"""Conservative structural answer projection edit for top-ranked entity questions.

No gold SQL, no other model calls; never modifies joins, grouping or filters.
"""
from __future__ import annotations
import re

SUPERLATIVE = re.compile(r"\b(most|least|highest|lowest|largest|smallest|top|maximum|minimum|longest|shortest|greatest|youngest|oldest|best|worst)\b", re.I)
ENTITY = re.compile(r"\b(which|who|whose|what|name|find|list|show|identify|give)\b", re.I)
METRIC = re.compile(r"\b(how many|how much|what is the (?:number|count|sum|total|average|maximum|minimum)|return (?:the )?(?:number|count|sum|total|average)|along with (?:the )?(?:number|count|total|sum|average)|(?:and|with) (?:the )?(?:number|count|total|sum|average))\b", re.I)
AGG = re.compile(r"\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(", re.I)
ALIAS = re.compile(r"\s+AS\s+([a-z_]\w*)\s*$", re.I)


def eligible(question: str) -> bool:
    return len(question) > 18 and bool(SUPERLATIVE.search(question) and ENTITY.search(question)) and not METRIC.search(question)


def split_items(text: str) -> list[str] | None:
    chunks, start, depth, quote = [], 0, 0, None
    for i, char in enumerate(text):
        if quote:
            if char == quote:
                quote = None
        elif char in ("'", '"', chr(96)):
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return None
        elif char == "," and depth == 0:
            chunks.append(text[start:i].strip())
            start = i + 1
    chunks.append(text[start:].strip())
    return chunks if depth == 0 and quote is None and all(chunks) else None


def repair(question: str, sql: str) -> tuple[str, str]:
    if not eligible(question):
        return sql, "not_entity_superlative"
    stripped = sql.strip()
    if re.search(r"\b(?:UNION|INTERSECT|EXCEPT|WITH|HAVING)\b", stripped, re.I) or len(re.findall(r"\bSELECT\b", stripped, re.I)) != 1:
        return sql, "complex_sql"
    match = re.match(r"(?is)^SELECT\s+(.+?)\s+FROM\s+(.+)$", stripped)
    if not match or match.group(1).lstrip().upper().startswith(("DISTINCT ", "ALL ")):
        return sql, "not_simple_select"
    parts = split_items(match.group(1))
    if not parts or len(parts) < 2:
        return sql, "single_projection"
    normal, metrics = [], []
    for part in parts:
        if AGG.search(part):
            if not re.match(r"(?is)^\s*(COUNT|SUM|AVG|MIN|MAX)\s*\(", part):
                return sql, "mixed_expression"
            alias = ALIAS.search(part)
            metrics.append((alias.group(1) if alias else None, (part[:alias.start()] if alias else part).strip()))
        else:
            normal.append(part)
    if not normal or not metrics:
        return sql, "no_extra_ranking_metric"
    tail = match.group(2)
    ordering = re.search(r"\bORDER\s+BY\b(.+?)(?=\bLIMIT\b|$)", tail, re.I | re.S)
    if not ordering:
        return sql, "no_ranking_order"
    revised = ordering.group(1)
    used = False
    for alias, expression in metrics:
        if alias:
            outside = tail[:ordering.start()] + tail[ordering.end():]
            if re.search(rf"\b{re.escape(alias)}\b", outside, re.I):
                return sql, "alias_required_outside_order"
            revised, n = re.subn(rf"(?<!\w){re.escape(alias)}(?!\w)", lambda _: expression, revised, flags=re.I)
            used |= bool(n)
        elif expression.casefold() in revised.casefold():
            used = True
        else:
            return sql, "metric_not_ranked"
    if not used:
        return sql, "metric_not_ranked"
    altered_tail = tail[:ordering.start(1)] + revised + tail[ordering.end(1):]
    answer = "SELECT " + ", ".join(normal) + " FROM " + altered_tail
    return answer, "removed_unused_ranking_metrics"
