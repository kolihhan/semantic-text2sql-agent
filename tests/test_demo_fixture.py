import json
import sqlite3
from pathlib import Path


def test_shipped_demo_database_and_questions_exist():
    root = Path(__file__).parents[1]
    db = root / "demo" / "demo.db"
    questions = root / "demo" / "questions.json"
    assert db.exists()
    con = sqlite3.connect(db)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()
    assert {"customers", "orders", "products"}.issubset(tables)
    payload = json.loads(questions.read_text(encoding="utf-8"))
    assert len(payload) >= 5
    assert any("Czech Republic" in row["question"] for row in payload)
