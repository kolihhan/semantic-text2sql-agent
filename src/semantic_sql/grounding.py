from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import re
from typing import Iterable, Mapping

from .catalog import DatabaseCatalog


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "has", "have", "how", "in", "is", "it", "of", "on", "or",
    "the", "to", "was", "were", "what", "when", "where", "which", "who", "with",
}


def _tokens(value: str) -> set[str]:
    expanded = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)
    words = re.findall(r"[a-z0-9]+", expanded.casefold())
    return {
        word.rstrip("s")
        for word in words
        if word not in _STOPWORDS and len(word.rstrip("s")) > 1
    }


def _normalise_value(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


@dataclass(frozen=True)
class QuestionDecomposition:
    entity: str = ""
    metric: str = ""
    filters: tuple[str, ...] = ()
    ranking: str = ""
    time: tuple[str, ...] = ()
    raw_question: str = ""

    def retrieval_parts(self) -> tuple[tuple[str, str, int], ...]:
        parts: list[tuple[str, str, int]] = []
        if self.entity.strip():
            parts.append(("entity", self.entity.strip(), 3))
        if self.metric.strip():
            parts.append(("metric", self.metric.strip(), 2))
        parts.extend(("filter", value.strip(), 3) for value in self.filters if value.strip())
        if self.ranking.strip():
            parts.append(("ranking", self.ranking.strip(), 1))
        parts.extend(("time", value.strip(), 2) for value in self.time if value.strip())
        if self.raw_question.strip():
            parts.append(("question", self.raw_question.strip(), 1))
        return tuple(parts)


@dataclass(frozen=True)
class IndexedValue:
    table: str
    column: str
    value: str


@dataclass(frozen=True)
class ValueHit:
    table: str
    column: str
    value: str
    score: float


@dataclass(frozen=True)
class GroundingPack:
    context: str
    anchor_tables: tuple[str, ...]
    selected_tables: tuple[str, ...]
    anchor_columns: tuple[str, ...]
    context_columns: tuple[str, ...]
    join_edges: tuple[tuple[str, str, str, str], ...]
    value_hits: tuple[ValueHit, ...]
    fallback_full_schema: bool

    def diagnostics(self) -> dict[str, object]:
        return {
            "selector": "dgsl_v1",
            "fallback_full_schema": self.fallback_full_schema,
            "anchor_tables": list(self.anchor_tables),
            "selected_tables": list(self.selected_tables),
            "anchor_columns": list(self.anchor_columns),
            "context_columns": list(self.context_columns),
            "join_edges": [list(edge) for edge in self.join_edges],
            "value_hits": [
                {
                    "table": hit.table,
                    "column": hit.column,
                    "value": hit.value,
                    "score": hit.score,
                }
                for hit in self.value_hits
            ],
        }


def heuristic_decomposition(question: str) -> QuestionDecomposition:
    """Cheap fail-closed decomposition used when structured decomposition fails."""
    lowered = question.casefold()

    metric_terms = (
        "how many", "count", "number of", "average", "avg", "mean", "sum",
        "total", "maximum", "minimum", "max", "min", "percentage", "percent",
    )
    ranking_terms = (
        "most", "least", "highest", "lowest", "top", "first", "last",
        "largest", "smallest", "fastest", "slowest",
    )

    metric = next((term for term in metric_terms if term in lowered), "")
    ranking = next((term for term in ranking_terms if term in lowered), "")

    quoted = tuple(
        match[1].strip()
        for match in re.findall(r"""(["'])(.+?)\1""", question)
        if match[1].strip()
    )
    time_parts: list[str] = []
    for pattern in (
        r"\bbetween\s+\d{4}\s+and\s+\d{4}\b",
        r"\bfrom\s+\d{4}\s+to\s+\d{4}\b",
        r"\b(?:before|after|since|until|through)\s+\d{4}\b",
        r"\b\d{4}\b",
    ):
        time_parts.extend(match.group(0) for match in re.finditer(pattern, question, flags=re.I))

    return QuestionDecomposition(
        entity=question,
        metric=metric,
        filters=quoted,
        ranking=ranking,
        time=tuple(dict.fromkeys(time_parts)),
        raw_question=question,
    )


def build_value_index(
    catalog: DatabaseCatalog,
    *,
    max_values_per_column: int = 256,
    max_value_chars: int = 120,
) -> tuple[IndexedValue, ...]:
    """Build a bounded in-memory value index once per database."""
    if max_values_per_column <= 0:
        return ()

    rows: list[IndexedValue] = []
    seen: set[tuple[str, str, str]] = set()
    for table in sorted(catalog.tables):
        for column in catalog.tables[table].columns:
            for raw in catalog.sample_values(table, column, limit=max_values_per_column):
                if raw is None or isinstance(raw, (bytes, bytearray, memoryview)):
                    continue
                value = re.sub(r"\s+", " ", str(raw).strip())
                if not value or len(value) > max_value_chars:
                    continue
                key = (table, column, _normalise_value(value))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(IndexedValue(table=table, column=column, value=value))
    return tuple(rows)


def _value_is_usable(value: str) -> bool:
    normalized = _normalise_value(value)
    if not normalized:
        return False
    if normalized.isdigit():
        # Pure numeric literals (years, counts, IDs) create too many accidental
        # value anchors. DGSL v1 grounds them through schema semantics instead.
        return False
    return len(normalized) >= 3


def _value_matches_haystack(needle: str, haystack: str) -> bool:
    if needle.isdigit():
        return re.search(rf"(?<!\d){re.escape(needle)}(?!\d)", haystack) is not None
    return needle in haystack


def match_indexed_values(
    value_index: Iterable[IndexedValue],
    *,
    question: str,
    evidence: str | None = None,
    decomposition: QuestionDecomposition | None = None,
    max_hits: int = 12,
) -> tuple[ValueHit, ...]:
    if max_hits <= 0:
        return ()

    search_chunks = [question]
    if evidence:
        search_chunks.append(evidence)
    if decomposition is not None:
        search_chunks.extend(text for _kind, text, _weight in decomposition.retrieval_parts())
    haystack = _normalise_value(" \n ".join(search_chunks))

    hits: list[ValueHit] = []
    for item in value_index:
        needle = _normalise_value(item.value)
        if not _value_is_usable(needle):
            continue
        if not _value_matches_haystack(needle, haystack):
            continue
        token_bonus = min(4, len(_tokens(item.value)))
        char_bonus = min(2.0, len(needle) / 20.0)
        hits.append(
            ValueHit(
                table=item.table,
                column=item.column,
                value=item.value,
                score=8.0 + token_bonus + char_bonus,
            )
        )

    hits.sort(key=lambda hit: (-hit.score, hit.table, hit.column, hit.value.casefold()))
    deduped: list[ValueHit] = []
    seen: set[tuple[str, str, str]] = set()
    for hit in hits:
        key = (hit.table, hit.column, _normalise_value(hit.value))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(hit)
        if len(deduped) >= max_hits:
            break
    return tuple(deduped)


def _adjacency(
    catalog: DatabaseCatalog,
) -> dict[str, list[tuple[str, tuple[str, str, str, str]]]]:
    graph: dict[str, list[tuple[str, tuple[str, str, str, str]]]] = defaultdict(list)
    for edge in catalog.foreign_keys:
        source, _source_col, target, _target_col = edge
        graph[source].append((target, edge))
        graph[target].append((source, edge))
    return graph


def _shortest_bridge(
    catalog: DatabaseCatalog,
    sources: set[str],
    target: str,
) -> tuple[tuple[str, str, str, str], ...] | None:
    if target in sources:
        return ()
    graph = _adjacency(catalog)
    queue = deque((source, ()) for source in sorted(sources))
    visited = set(sources)
    while queue:
        node, path = queue.popleft()
        for neighbor, edge in graph.get(node, ()):
            if neighbor in visited:
                continue
            next_path = (*path, edge)
            if neighbor == target:
                return next_path
            visited.add(neighbor)
            queue.append((neighbor, next_path))
    return None


def _join_edges_for_tables(
    catalog: DatabaseCatalog,
    selected: set[str],
) -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        edge
        for edge in catalog.foreign_keys
        if edge[0] in selected and edge[2] in selected
    )


def build_grounding_pack(
    catalog: DatabaseCatalog,
    *,
    decomposition: QuestionDecomposition,
    question: str,
    evidence: str | None = None,
    descriptions: Mapping[tuple[str, str], str] | None = None,
    value_index: Iterable[IndexedValue] = (),
    max_anchor_tables: int = 4,
    max_anchor_columns_per_table: int = 6,
    max_value_hits: int = 12,
) -> GroundingPack:
    """Ground question pieces before generation, then connect anchors through FK paths."""
    descriptions = descriptions or {}
    parts = list(decomposition.retrieval_parts())
    if evidence and evidence.strip():
        parts.append(("evidence", evidence.strip(), 1))

    table_scores: dict[str, float] = {table: 0.0 for table in catalog.tables}
    column_scores: dict[tuple[str, str], float] = {}

    for table, info in catalog.tables.items():
        table_tokens = _tokens(table)
        for _kind, text, weight in parts:
            query_tokens = _tokens(text)
            if not query_tokens:
                continue
            table_scores[table] += 4.0 * weight * len(query_tokens & table_tokens)

        for column in info.columns:
            key = (table, column)
            score = 0.0
            column_tokens = _tokens(column)
            description_tokens = _tokens(descriptions.get(key, ""))
            for _kind, text, weight in parts:
                query_tokens = _tokens(text)
                if not query_tokens:
                    continue
                score += 3.0 * weight * len(query_tokens & column_tokens)
                score += 1.5 * weight * len(query_tokens & description_tokens)
            if score > 0:
                column_scores[key] = score
                table_scores[table] += min(score, 12.0)

    value_hits = match_indexed_values(
        value_index,
        question=question,
        evidence=evidence,
        decomposition=decomposition,
        max_hits=max_value_hits,
    )
    for hit in value_hits:
        key = (hit.table, hit.column)
        column_scores[key] = column_scores.get(key, 0.0) + hit.score
        table_scores[hit.table] = table_scores.get(hit.table, 0.0) + hit.score + 4.0

    ranked_tables = [
        table
        for table, score in sorted(
            table_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if score > 0
    ]

    fallback_full_schema = not ranked_tables
    bridge_edges: list[tuple[str, str, str, str]] = []
    if fallback_full_schema:
        anchor_tables = tuple(sorted(catalog.tables))
        selected = set(catalog.tables)
        bridge_edges.extend(catalog.foreign_keys)
    else:
        anchor_tables = tuple(ranked_tables[:max(1, max_anchor_tables)])
        selected = {anchor_tables[0]}
        for table in anchor_tables[1:]:
            bridge = _shortest_bridge(catalog, selected, table)
            if bridge is not None:
                for edge in bridge:
                    source, _source_col, target, _target_col = edge
                    selected.add(source)
                    selected.add(target)
                    if edge not in bridge_edges:
                        bridge_edges.append(edge)
            selected.add(table)

    anchor_columns: list[str] = []
    for table in anchor_tables:
        ranked_columns = [
            (column, score)
            for (candidate_table, column), score in column_scores.items()
            if candidate_table == table and score > 0
        ]
        ranked_columns.sort(key=lambda item: (-item[1], item[0]))
        keep = [column for column, _score in ranked_columns[:max_anchor_columns_per_table]]
        for hit in value_hits:
            if hit.table == table and hit.column not in keep:
                keep.append(hit.column)
        anchor_columns.extend(f"{table}.{column}" for column in keep)

    join_edges = tuple(bridge_edges)
    context_columns = tuple(
        f"{table}.{column}"
        for table in sorted(selected)
        for column in catalog.tables[table].columns
    )

    lines = ["Grounded SQLite schema context:"]
    decomposition_lines = []
    if decomposition.entity.strip():
        decomposition_lines.append(f"- entity/relation: {decomposition.entity.strip()}")
    if decomposition.metric.strip():
        decomposition_lines.append(f"- metric: {decomposition.metric.strip()}")
    if decomposition.filters:
        decomposition_lines.append(f"- filters: {' | '.join(decomposition.filters)}")
    if decomposition.ranking.strip():
        decomposition_lines.append(f"- ranking: {decomposition.ranking.strip()}")
    if decomposition.time:
        decomposition_lines.append(f"- time: {' | '.join(decomposition.time)}")
    if decomposition_lines:
        lines.append("Retrieval decomposition (not SQL):")
        lines.extend(decomposition_lines)

    if value_hits:
        lines.append("Matched database values:")
        lines.extend(
            f"- {hit.table}.{hit.column} = {hit.value!r}"
            for hit in value_hits
        )

    lines.append("Selected schema:")
    for table in sorted(selected):
        lines.append(f"{table}({', '.join(catalog.tables[table].columns)})")

    if join_edges:
        lines.append("Join backbone:")
        lines.extend(
            f"- {source}.{source_col} = {target}.{target_col}"
            for source, source_col, target, target_col in join_edges
        )

    described_columns = set(anchor_columns) | {
        f"{hit.table}.{hit.column}" for hit in value_hits
    }
    description_lines = []
    for qualified in sorted(described_columns):
        table, column = qualified.split(".", 1)
        description = descriptions.get((table, column), "").strip()
        if description:
            description_lines.append(f"- {qualified}: {description}")
    if description_lines:
        lines.append("Grounded column descriptions:")
        lines.extend(description_lines)

    return GroundingPack(
        context="\n".join(lines),
        anchor_tables=anchor_tables,
        selected_tables=tuple(sorted(selected)),
        anchor_columns=tuple(sorted(set(anchor_columns))),
        context_columns=context_columns,
        join_edges=join_edges,
        value_hits=value_hits,
        fallback_full_schema=fallback_full_schema,
    )
