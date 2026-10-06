from __future__ import annotations

from dataclasses import dataclass

from .catalog import DatabaseCatalog


def _norm(value: str) -> str:
    return value.casefold()


def _complete_fk(parts: tuple[object, object, object, object]) -> bool:
    return all(isinstance(value, str) and value.strip() for value in parts)


@dataclass(frozen=True)
class SchemaGraph:
    fk_edges: frozenset[tuple[str, str, str, str]]

    @classmethod
    def from_catalog(cls, catalog: DatabaseCatalog) -> "SchemaGraph":
        return cls(
            frozenset(
                (
                    _norm(source_table),
                    _norm(source_col),
                    _norm(target_table),
                    _norm(target_col),
                )
                for source_table, source_col, target_table, target_col in catalog.foreign_keys
                if _complete_fk((source_table, source_col, target_table, target_col))
            )
        )

    def supports_fk_equality(
        self,
        left_table: str,
        left_column: str,
        right_table: str,
        right_column: str,
    ) -> bool:
        edge = (
            _norm(left_table),
            _norm(left_column),
            _norm(right_table),
            _norm(right_column),
        )
        reverse = (edge[2], edge[3], edge[0], edge[1])
        return edge in self.fk_edges or reverse in self.fk_edges
