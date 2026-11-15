"""Post-hoc, fixed SQL counterfactual on original Chinook 1.4.5 physical data.

Shows whether a published Qwen wrong-grain query is invariant to adding
purchases to an EXISTING invoice while the correct reference responds.
This proves one failure mechanism, not any generalizable SQL repair gain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
from hashlib import sha256

# Exact previously generated Qwen9B SQL from PR #54, physical run 43703246942:
QWEN_T13 = "SELECT AVG(Quantity) AS average_tracks_per_invoice FROM InvoiceLine;"
# Independent source SQL authored with Chinook tasks (t13):
REFERENCE_T13 = (
    "SELECT ROUND(AVG(n), 2) FROM "
    "(SELECT InvoiceId, SUM(Quantity) AS n FROM InvoiceLine GROUP BY InvoiceId)"
)
SOURCE_TAG = "v1.4.5"
SOURCE_BLOB_SHA = "3989874548c7be6156a49d3b37800c3ee9b2a6d1"


def scalar(connection: sqlite3.Connection, sql: str) -> float:
    return float(connection.execute(sql).fetchone()[0])


def audit(connection: sqlite3.Connection, *, added_lines: int = 100) -> dict:
    if added_lines < 5:
        raise ValueError("intervention must yield a measurable rounded difference")
    original_invoice_count = scalar(connection, "SELECT COUNT(*) FROM Invoice")
    original_total_lines = scalar(connection, "SELECT COUNT(*) FROM InvoiceLine")
    old_correct = scalar(connection, REFERENCE_T13)
    old_wrong = scalar(connection, QWEN_T13)
    row = connection.execute(
        "SELECT InvoiceId, TrackId, UnitPrice FROM InvoiceLine "
        "ORDER BY InvoiceLineId LIMIT 1"
    ).fetchone()
    if row is None:
        raise ValueError("physical Chinook records missing")
    invoice_id, track_id, unitprice = row
    start = int(scalar(connection, "SELECT MAX(InvoiceLineId) FROM InvoiceLine"))
    # Add purchased tracks to an existing invoice; quantity=1, so a wrong
    # line-level AVG(Quantity) must stay fixed despite larger per-invoice count.
    with connection:
        connection.executemany(
            "INSERT INTO InvoiceLine "
            "(InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) "
            "VALUES (?, ?, ?, ?, 1)",
            [(start + i, invoice_id, track_id, unitprice) for i in range(1, added_lines + 1)]
        )
        # Keep invoice accounting consistent with the additional purchased units.
        connection.execute(
            "UPDATE Invoice SET Total = ROUND(Total + ?, 2) WHERE InvoiceId = ?",
            (added_lines * unitprice, invoice_id),
        )
    new_correct = scalar(connection, REFERENCE_T13)
    new_wrong = scalar(connection, QWEN_T13)
    assert scalar(connection, "SELECT COUNT(*) FROM Invoice") == original_invoice_count
    assert scalar(connection, "SELECT COUNT(*) FROM InvoiceLine") == original_total_lines + added_lines
    assert abs(new_wrong - old_wrong) < 1e-12
    assert new_correct > old_correct + .1
    return {
        "protocol": "chinook-t13-real-grain-sensitivity-v1",
        "database": "Chinook 1.4.5 populated original source SQL",
        "original_source_blob_sha": SOURCE_BLOB_SHA,
        "observed_failure_case": "PR#54 Chinook t13",
        "original_pinned_qwen_sql": QWEN_T13,
        "source_reference_sql": REFERENCE_T13,
        "invoice_count": int(original_invoice_count),
        "new_line_items_added_to_existing_invoice": added_lines,
        "baseline": {"reference": old_correct, "qwen_wrong_grain": old_wrong},
        "counterfactual": {"reference": new_correct, "qwen_wrong_grain": new_wrong},
        "reference_sensitive": new_correct > old_correct + .1,
        "qwen_wrong_insensitive": abs(new_wrong - old_wrong) < 1e-12,
        "new_model_calls": 0,
        "new_model_gain": False,
        "decision": "CONFIRMED_GRAIN_FAILURE_NOT_A_REPAIR",
        "caveat": (
            "Post-hoc case already identified as wrong. Not a blinded unseen-DB "
            "validation and not an increase to BIRD100 or external strict EX."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-sql", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    with sqlite3.connect(":memory:") as conn:
        conn.executescript(a.source_sql.read_text(encoding="utf-8"))
        report = audit(conn)
    report["source_sql_sha256"] = sha256(a.source_sql.read_bytes()).hexdigest()
    a.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("SQL_GRAIN_COUNTERFACTUAL", json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
