"""Regression coverage for lexical schema grounding; no SQL model needed."""
from pathlib import Path

from evaluation.schema_context import _tokens, select_schema_context
from semantic_sql.catalog import DatabaseCatalog, TableInfo


def test_compound_column_identifiers_split_before_casefold():
    assert {"pay", "frequency"} <= _tokens("PayFrequency")
    assert {"start", "date"} <= _tokens("StartDate")
    assert {"student", "id"} <= _tokens("StudentID")
    assert {"http", "response", "code"} <= _tokens("HTTPResponseCode")


def test_compound_schema_retrieval_avoids_full_schema_fallback(tmp_path: Path):
    catalog = DatabaseCatalog(
        database=tmp_path / "unused.sqlite",
        tables={
            "Payroll": TableInfo("Payroll", ("PayFrequency", "Amount")),
            "Geography": TableInfo("Geography", ("CountryCode", "Latitude")),
            "Inventory": TableInfo("Inventory", ("StockCode", "Quantity")),
        },
        foreign_keys=(),
    )
    result, diagnostics = select_schema_context(
        catalog, tmp_path, "mock", "Which pay frequency occurs most often?"
    )
    assert diagnostics["selected_tables"] == ["Payroll"]
    assert diagnostics["fallback_full_schema"] is False
    assert "Payroll(PayFrequency, Amount)" in result
    assert "Geography(" not in result
