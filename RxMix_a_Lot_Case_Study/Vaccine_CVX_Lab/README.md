# Rx-Mix-a-Lot | Vaccine NDC → CVX lab

A **reproducible, synthetic-data exercise** in relational databases, HTTP APIs, healthcare terminology, and clinical decision support (CDS). Runs in a GitHub Codespace with Python 3 alone; no database server, login, API token, or `pip install` is needed.

> **Teaching demonstration only.** This is not a validated immunization forecaster. Do not use patient data or base treatment decisions on this project.

## Learning sequence

```text
Synthetic patients + administrations (SQLite; NDC only)
        ↓ SQL SELECT DISTINCT / JOIN
Unmapped vaccine NDCs
        ↓ HTTP GET /api/v1/ndc/{ndc11}
CDC-derived, local teaching terminology API (JSON)
        ↓ JSON parsing + SQL UPSERT
NDC-to-CVX mapping table in SQLite
        ↓ LEFT JOIN and illustrative series rule
Patient-level RZV teaching assessment + data-quality review
```

**Why a local API?** The HTTP call is real, but the server uses a small, frozen educational extract of the CDC NDC/CVX crosswalk so class can be repeated even if internet access or the live CDC service changes. It is **not** an official CDC-hosted service. The CDC provides a live vaccine code set REST interface for more advanced work. See [CDC terminology access](https://www.cdc.gov/iis/code-sets/viewpoint-rest.html). Check against current CDC data before using any mapping operationally.

## GitHub Codespaces quick start

1. Add this entire `Vaccine_CVX_Lab/` folder to a GitHub repository and commit it (including `data/starter.sqlite`). Open the repository in **Code → Codespaces**. At the root of your repository, `cd` to this folder; for the example repository layout: `cd Rx_Mix_a_Lot/Vaccine_CVX_Lab`.
2. Open **Terminal A** and run:

   ```bash
   python3 api_server.py
   ```

   The API serves `http://127.0.0.1:8765`; leave Terminal A running. `127.0.0.1` means only this Codespace can use it; there is no need to forward or publish a port.
3. Open **Terminal B** and run:

   ```bash
   python3 exercise.py reset
   python3 exercise.py query "SELECT * FROM vaccine_administrations LIMIT 5"
   python3 exercise.py extract
   python3 exercise.py map
   python3 exercise.py query "SELECT ndc11, cvx_code, mapping_status FROM ndc_cvx_map ORDER BY ndc11"
   python3 exercise.py report
   ```

4. **To run the class again, restore the exact initial state:**

   ```bash
   python3 exercise.py reset
   ```

   This copies `data/starter.sqlite` to a disposable working database (`data/workspace.sqlite`). All mappings and previous student changes in the working copy are removed. You can run reset repeatedly, including in a new Codespace. The original starter database stays unchanged.

5. Stop the API with **Ctrl+C** in Terminal A.

## What students should observe

- Six distinct recognized NDCs plus **one deliberately unknown NDC** are in the data; mapping requests are sent for only those not yet mapped. The client stores HTTP `404` as **unmapped**, not as a fabricated CVX.
- Different NDCs (unit of sale **versus** unit of use) can identify the same vaccine CVX.
- Each NDC is looked up **once** even if multiple patients received it; `ndc_cvx_map` is a normalized reference table joined back to administrations.
- `vaccine_administrations.ndc11` is **TEXT**, so leading zeroes and package distinctions are preserved; never interpret NDC as a numeric quantity or arbitrarily add padding without knowing its original segment structure.
- The `report` shows **MEETS_DEMO_RULE, INCOMPLETE_DEMO, REVIEW_INTERVAL, REVIEW_UNMAPPED,** or **NO_RZV_RECORDED**. `entered-in-error` rows are excluded.

## Database tables

| Table | Role |
| --- | --- |
| `patients` | Synthetic patient demographics |
| `vaccine_administrations` | Source system history, including administration date, NDC, recorded product, and status |
| `ndc_cvx_map` | Terminology results, `mapping_status`, source and lookup timestamp; initially empty |

Example SQL exploration:

```bash
python3 exercise.py query "SELECT p.display_name, a.administered_on, a.ndc11, m.cvx_code, m.mapping_status FROM patients p JOIN vaccine_administrations a USING(patient_id) LEFT JOIN ndc_cvx_map m ON m.ndc11=a.ndc11 ORDER BY p.patient_id, a.administered_on"
```

The `query` command is intentionally read-only. The actual SQL lives in `exercise.py` and the schema is in `sql/schema.sql`. Try `python3 exercise.py --help` to see commands. Students can also open the `.sqlite` file using any compatible VS Code SQLite extension, but **no extension is required**.

## Simplified series assessment — important limitations

For a deliberately limited set of **fictional, immunocompetent patients aged 50+**, the `report` demonstrates a classroom-only rule: **two documented recombinant zoster vaccine (RZV, CVX 187) administrations separated by at least 28 days**. This mirrors only a tiny portion of the real evidence and timing questions. CDC typically recommends two RZV doses **2–6 months apart** for immunocompetent adults aged 50+, with a minimum interval of four weeks. Actual clinical forecasting also considers nuanced eligibility, vaccination history, validity rules, immunocompromise and timing. This program intentionally **does not** compute medical recommendations or assert clinical series completion. A result of `MEETS_DEMO_RULE` is **not** a medical conclusion. See [CDC shingles vaccination recommendations](https://www.cdc.gov/shingles/hcp/vaccine-considerations/index.html).

## Files and instructor maintenance

```text
Vaccine_CVX_Lab/
├── .gitignore                  # never commit students' working database
├── README.md
├── student_activity.md
├── instructor_guide.md
├── api_server.py               # local HTTP JSON terminology service
├── exercise.py                 # reset, extract, map, SQL query, report
├── build_starter.py            # instructor: rebuild clean starter SQLite
├── sql/
│   ├── schema.sql
│   └── seed.sql
├── data/
│   ├── starter.sqlite          # version-controlled, pristine template
│   └── teaching_crosswalk.json # fixed CDC-derived training subset
└── tests/
    └── test_lab.py
```

The starter is derived from SQL sources. To **change the curriculum next year**: edit `sql/schema.sql` and/or `sql/seed.sql`, run `python3 build_starter.py`, run the tests, then commit the refreshed `data/starter.sqlite` and the SQL changes. The check below resets the working database, so do not run it during student work:

```bash
python3 -m unittest discover -s tests -v
```

## Sources and versioning

- [CDC NDC Lookup Crosswalk](https://www2.cdc.gov/vaccines/iis/iisstandards/vaccines.asp?rpt=ndc), including the distinction between unit-of-use and unit-of-sale package NDCs.
- [CDC vaccine code sets](https://www.cdc.gov/iis/code-sets/index.html), 2026-09-18 release (the latest listing checked when this lesson was prepared).
- [CDC vaccine terminology REST access](https://www.cdc.gov/iis/code-sets/viewpoint-rest.html) for advanced discussion of an external service.
- [CDC clinical shingles guidance](https://www.cdc.gov/shingles/hcp/vaccine-considerations/index.html).

The fixture is **not the full crosswalk**, and its selected mapping rows are an instructional snapshot, not a substitute for verifying code life cycles and status against CDC. The intentionally invented NDC `99999-9999-99` is an unmapped data-quality case, not a valid product.
