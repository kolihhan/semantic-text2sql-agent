from pathlib import Path
import sqlite3

root = Path(__file__).parents[1]
db = root / "demo" / "demo.db"
if db.exists():
    db.unlink()
con = sqlite3.connect(db)
con.executescript((root / "demo" / "schema.sql").read_text(encoding="utf-8"))
con.close()
print(db)
