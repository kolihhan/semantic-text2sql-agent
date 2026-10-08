from __future__ import annotations

import sqlite3
from pathlib import Path

from semantic_sql.grounding import build_zero_row_evidence


def test_zero_row_evidence_surfaces_nearby_filter_value(tmp_path: Path) -> None:
    database = tmp_path / "races.sqlite"
    with sqlite3.connect(database) as con:
        con.execute("CREATE TABLE races (name TEXT, year INTEGER)")
        con.execute("INSERT INTO races VALUES ('Singapore Grand Prix', 2010)")

    evidence = build_zero_row_evidence(
        database,
        question="Name the 2010 Singapore Grand Prix.",
        sql="SELECT name FROM races WHERE name = 'Singapore' AND year = 2010",
    )

    assert "races.name" in evidence
    assert "Singapore Grand Prix" in evidence


def test_zero_row_evidence_surfaces_literal_in_different_column(tmp_path: Path) -> None:
    database = tmp_path / "bonds.sqlite"
    with sqlite3.connect(database) as con:
        con.execute("CREATE TABLE bond (bond_id TEXT, bond_type TEXT)")
        con.execute("INSERT INTO bond VALUES ('TR001_2_4', 'single')")

    evidence = build_zero_row_evidence(
        database,
        question="What is the label for bond TR001_2_4?",
        sql="SELECT bond_type FROM bond WHERE bond_type = 'TR001_2_4'",
    )

    assert "bond.bond_id" in evidence
    assert "TR001_2_4" in evidence
