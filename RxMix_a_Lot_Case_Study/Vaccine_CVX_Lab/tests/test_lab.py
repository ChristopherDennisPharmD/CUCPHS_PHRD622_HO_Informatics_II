"""Offline tests: python3 -m unittest discover -s tests -v

Tests reset the disposable workspace; do not run during class work.
"""
from http.server import ThreadingHTTPServer
from pathlib import Path
import json
import sqlite3
import subprocess
import sys
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api_server import Handler  # noqa: E402


class LabTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "exercise.py"), *args],
                              cwd=ROOT, capture_output=True, text=True, check=True).stdout

    def setUp(self):
        self.run_cli("reset")

    def test_starter_data_no_mapping(self):
        with sqlite3.connect(ROOT / "data/workspace.sqlite") as c:
            self.assertEqual(c.execute("SELECT count(*) FROM patients").fetchone()[0], 7)
            self.assertEqual(c.execute("SELECT count(*) FROM vaccine_administrations").fetchone()[0], 13)
            self.assertEqual(c.execute("SELECT count(*) FROM ndc_cvx_map").fetchone()[0], 0)

    def test_extract_ndcs(self):
        output = self.run_cli("extract")
        self.assertIn("58160-0823-11", output)
        self.assertIn("99999-9999-99", output)

    def test_fixture_matches_seed_ndcs(self):
        fixture = json.loads((ROOT / "data/teaching_crosswalk.json").read_text())
        with sqlite3.connect(ROOT / "data/workspace.sqlite") as c:
            ndcs = {r[0] for r in c.execute("SELECT DISTINCT ndc11 FROM vaccine_administrations")}
        self.assertEqual(ndcs - set(fixture["entries"]), {"99999-9999-99"})

    def test_api_to_sql_to_report(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            api_url = f"http://127.0.0.1:{server.server_address[1]}"
            self.run_cli("map", "--api", api_url)
            with sqlite3.connect(ROOT / "data/workspace.sqlite") as c:
                self.assertEqual(c.execute("SELECT COUNT(*) FROM ndc_cvx_map").fetchone()[0], 7)
                self.assertEqual(c.execute("SELECT COUNT(*) FROM ndc_cvx_map WHERE mapping_status='mapped'").fetchone()[0], 6)
                self.assertEqual(c.execute("SELECT cvx_code FROM ndc_cvx_map WHERE ndc11='58160-0849-43'").fetchone()[0], "187")
            outcome = self.run_cli("report")
            for patient, expected in (("Jordan Lee", "MEETS_DEMO_RULE"),
                                      ("Casey Rivera", "INCOMPLETE_DEMO"),
                                      ("Sam Patel", "REVIEW_INTERVAL"),
                                      ("Avery Chen", "REVIEW_UNMAPPED"),
                                      ("Morgan Blake", "NO_RZV_RECORDED"),
                                      ("Riley Park", "INCOMPLETE_DEMO"),
                                      ("Taylor Singh", "MEETS_DEMO_RULE")):
                self.assertIn(patient, outcome)
                self.assertIn(expected, next(line for line in outcome.splitlines() if patient in line))
            self.assertIn("No new NDCs to request", self.run_cli("map", "--api", api_url))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
