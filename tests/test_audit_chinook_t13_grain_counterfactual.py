import sqlite3
import pytest
from evaluation.audit_chinook_t13_grain_counterfactual import audit


def make_physical_source():
    c = sqlite3.connect(":memory:")
    c.executescript("""
    CREATE TABLE Invoice(InvoiceId INT PRIMARY KEY, Total REAL NOT NULL);
    CREATE TABLE InvoiceLine(InvoiceLineId INT PRIMARY KEY, InvoiceId INT,
                             TrackId INT, UnitPrice REAL, Quantity INT);
    INSERT INTO Invoice VALUES (1,2),(2,2),(3,2);
    INSERT INTO InvoiceLine VALUES (1,1,2,1,1),(2,1,3,1,1),
     (3,2,4,1,1),(4,2,5,1,1),(5,3,6,1,1),(6,3,7,1,1);
    """)
    return c


def test_wrong_line_average_is_insensitive_to_per_invoice_track_increase():
    with make_physical_source() as c:
        report = audit(c, added_lines=100)
    assert report["baseline"]["qwen_wrong_grain"] == 1.0
    assert report["counterfactual"]["qwen_wrong_grain"] == 1.0
    assert report["reference_sensitive"]
    assert report["new_model_calls"] == 0
    assert report["new_model_gain"] is False


def test_fails_for_too_small_intervention():
    with make_physical_source() as c:
        with pytest.raises(ValueError, match="measurable"):
            audit(c, added_lines=1)
