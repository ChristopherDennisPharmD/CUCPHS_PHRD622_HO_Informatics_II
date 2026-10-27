"""Regenerate pristine SQLite seed from versioned .sql files. For instructors."""
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data/starter.sqlite"
if OUT.exists():
    OUT.unlink()
with sqlite3.connect(OUT) as conn:
    conn.executescript((HERE / "sql/schema.sql").read_text(encoding="utf-8"))
    conn.executescript((HERE / "sql/seed.sql").read_text(encoding="utf-8"))
    conn.commit()
    print(f"Built {OUT.name}: {conn.execute('SELECT count(*) FROM vaccine_administrations').fetchone()[0]} administrations")
