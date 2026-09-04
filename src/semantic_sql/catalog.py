from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3


@dataclass(frozen=True)
class TableInfo:
    name: str
    columns: tuple[str, ...]


@dataclass
class DatabaseCatalog:
    database: Path
    tables: dict[str, TableInfo]
    foreign_keys: tuple[tuple[str, str, str, str], ...]

    @classmethod
    def from_sqlite(cls, database: str | Path) -> "DatabaseCatalog":
        database = Path(database)
        con = sqlite3.connect(database)
        try:
            table_names = [
                row[0]
                for row in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
            tables: dict[str, TableInfo] = {}
            foreign_keys: list[tuple[str, str, str, str]] = []
            for table in table_names:
                columns = tuple(row[1] for row in con.execute(f'PRAGMA table_info("{table}")'))
                tables[table] = TableInfo(table, columns)
                for row in con.execute(f'PRAGMA foreign_key_list("{table}")'):
                    foreign_keys.append((table, row[3], row[2], row[4]))
            return cls(database=database, tables=tables, foreign_keys=tuple(foreign_keys))
        finally:
            con.close()

    def sample_values(self, table: str, column: str, limit: int = 50) -> tuple[object, ...]:
        if table not in self.tables or column not in self.tables[table].columns:
            return ()
        con = sqlite3.connect(self.database)
        try:
            rows = con.execute(
                f'SELECT DISTINCT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL LIMIT ?',
                (limit,),
            ).fetchall()
            return tuple(row[0] for row in rows)
        finally:
            con.close()


    def find_value(self, table: str, column: str, value: object) -> object | None:
        if table not in self.tables or column not in self.tables[table].columns:
            return None
        con = sqlite3.connect(self.database)
        try:
            if isinstance(value, str):
                row = con.execute(
                    f'SELECT "{column}" FROM "{table}" WHERE CAST("{column}" AS TEXT) = ? COLLATE NOCASE LIMIT 1',
                    (value,),
                ).fetchone()
            else:
                row = con.execute(
                    f'SELECT "{column}" FROM "{table}" WHERE "{column}" = ? LIMIT 1',
                    (value,),
                ).fetchone()
            return None if row is None else row[0]
        finally:
            con.close()

    def schema_text(self) -> str:
        lines: list[str] = []
        for table in sorted(self.tables):
            lines.append(f"{table}({', '.join(self.tables[table].columns)})")
        for source_table, source_col, target_table, target_col in self.foreign_keys:
            lines.append(f"FK {source_table}.{source_col} -> {target_table}.{target_col}")
        return "\n".join(lines)
