"""Rx-Mix-a-Lot Vaccine CVX Lab command-line client; Python 3 standard library only."""
import argparse
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen
import json
import shutil
import sqlite3

ROOT = Path(__file__).resolve().parent
STARTER = ROOT / "data" / "starter.sqlite"
WORKSPACE = ROOT / "data" / "workspace.sqlite"
DEFAULT_API = "http://127.0.0.1:8765"


def connect():
    if not WORKSPACE.exists():
        raise SystemExit("No working database. First run: python3 exercise.py reset")
    conn = sqlite3.connect(WORKSPACE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def reset():
    if not STARTER.exists():
        raise SystemExit("Starter database missing; run: python3 build_starter.py")
    shutil.copyfile(STARTER, WORKSPACE)
    print("Working database restored from version-controlled starter.sqlite.")
    print("All terminology mappings have been cleared.")


def print_rows(rows):
    if not rows:
        print("(no rows)")
        return
    names = rows[0].keys()
    widths = [max(len(str(k)), *(len(str(r[k])) if r[k] is not None else 4 for r in rows)) for k in names]
    print(" | ".join(str(k).ljust(w) for k, w in zip(names, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join((str(r[k]) if r[k] is not None else "NULL").ljust(w) for k, w in zip(names, widths)))


def extract():
    with connect() as conn:
        rows = conn.execute("""
            SELECT DISTINCT a.ndc11
            FROM vaccine_administrations a
            LEFT JOIN ndc_cvx_map m ON m.ndc11 = a.ndc11
            WHERE m.ndc11 IS NULL AND a.administration_status = 'given'
            ORDER BY a.ndc11
        """).fetchall()
    print_rows(rows)
    return [r["ndc11"] for r in rows]


def map_ndcs(api_base):
    ndcs = extract()
    if not ndcs:
        print("No new NDCs to request. Reset to start over.")
        return
    with connect() as conn:
        for ndc in ndcs:
            url = api_base.rstrip("/") + "/api/v1/ndc/" + quote(ndc, safe="")
            try:
                with urlopen(url, timeout=5) as response:
                    item = json.load(response)
                if item.get("ndc11") != ndc or not item.get("cvx_code"):
                    raise ValueError("API returned an unexpected or incomplete NDC mapping")
                values = (ndc, str(item["cvx_code"]), item.get("cvx_description"),
                          item.get("vaccine_group"), "mapped", item.get("source", api_base),
                          datetime.now(timezone.utc).isoformat())
                status = "mapped to CVX " + str(item["cvx_code"])
            except HTTPError as exc:
                if exc.code != 404:
                    print(f"{ndc}: HTTP {exc.code}; not stored (retry later)")
                    continue
                values = (ndc, None, None, None, "unmapped", api_base,
                          datetime.now(timezone.utc).isoformat())
                status = "not found; stored as unmapped (needs review)"
            except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                print(f"{ndc}: service error ({exc}); not stored (retry later)")
                continue
            conn.execute("""
                INSERT INTO ndc_cvx_map
                (ndc11, cvx_code, cvx_description, vaccine_group, mapping_status, mapping_source, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ndc11) DO UPDATE SET
                    cvx_code = excluded.cvx_code,
                    cvx_description = excluded.cvx_description,
                    vaccine_group = excluded.vaccine_group,
                    mapping_status = excluded.mapping_status,
                    mapping_source = excluded.mapping_source,
                    fetched_at = excluded.fetched_at
            """, values)
            print(f"{ndc}: {status}")
    print("\nMappings saved. Run: python3 exercise.py report")


def report():
    """Deliberately restricted demo: routine RZV, ages 50+, 2 doses >= 28 days apart.

    NOT a production clinical forecast. No complete vaccination recommendations.
    """
    with connect() as conn:
        records = conn.execute("""
            SELECT p.patient_id, p.display_name, p.date_of_birth,
                   a.administration_id, a.administered_on, a.ndc11,
                   m.cvx_code, m.mapping_status
            FROM patients p
            LEFT JOIN vaccine_administrations a
              ON a.patient_id=p.patient_id AND a.administration_status='given'
            LEFT JOIN ndc_cvx_map m ON m.ndc11=a.ndc11
            ORDER BY p.patient_id, a.administered_on, a.administration_id
        """).fetchall()

    grouped = {}
    for r in records:
        grouped.setdefault(r["patient_id"], []).append(r)
    output = []
    for pid, items in grouped.items():
        given = [r for r in items if r["administration_id"] is not None]
        zoster = [r for r in given if r["cvx_code"] == "187"]
        unresolved = [r for r in given if r["mapping_status"] != "mapped"]
        dates = [date.fromisoformat(r["administered_on"]) for r in zoster]
        birth = date.fromisoformat(items[0]["date_of_birth"])
        # These vignettes are routine immunocompetent adult assessments; dates matter.
        young = any((d.year - birth.year - ((d.month, d.day) < (birth.month, birth.day))) < 50 for d in dates)
        if unresolved:
            status = "REVIEW_UNMAPPED"
        elif young:
            status = "REVIEW_AGE"
        elif len(dates) < 2:
            status = "INCOMPLETE_DEMO" if dates else "NO_RZV_RECORDED"
        elif any((b - a).days >= 28 for i, a in enumerate(dates) for b in dates[i + 1:]):
            status = "MEETS_DEMO_RULE"
        else:
            status = "REVIEW_INTERVAL"
        output.append({"patient": pid, "name": items[0]["display_name"],
                       "rzv_doses": len(dates), "unmapped": len(unresolved),
                       "teaching_assessment": status})
    print("Classroom-only simplified RZV assessment: 2 recorded doses >=28 days apart, given at age >=50.")
    print("NOT a clinical vaccine forecast. The normal recommended interval is 2-6 months.")
    print_rows(output)


def query(statement):
    if not statement.lstrip().lower().startswith(("select ", "with ", "pragma table_info")):
        raise SystemExit("For safety the query command only supports SELECT/WITH/PRAGMA table_info.")
    if not WORKSPACE.exists():
        raise SystemExit("No working database. First run: python3 exercise.py reset")
    # SQLite itself enforces read-only operation, not just the prefix check.
    with sqlite3.connect(f"file:{WORKSPACE.resolve().as_posix()}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        print_rows(conn.execute(statement).fetchall())


def main():
    p = argparse.ArgumentParser(description="Reproducible vaccine NDC-to-CVX data standards lab")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("reset", help="Restore original working SQLite database")
    sub.add_parser("extract", help="Find distinct unmapped NDCs with SQL")
    m = sub.add_parser("map", help="Call HTTP terminology API; save JSON response in SQLite")
    m.add_argument("--api", default=DEFAULT_API, help="HTTP API base URL")
    sub.add_parser("report", help="Apply simplified, non-clinical vaccine-series demo logic")
    q = sub.add_parser("query", help="Run read-only SQL and print results")
    q.add_argument("sql", help="Quoted SQL SELECT query")
    a = p.parse_args()
    if a.command == "reset": reset()
    elif a.command == "extract": extract()
    elif a.command == "map": map_ndcs(a.api)
    elif a.command == "report": report()
    elif a.command == "query": query(a.sql)


if __name__ == "__main__":
    main()
