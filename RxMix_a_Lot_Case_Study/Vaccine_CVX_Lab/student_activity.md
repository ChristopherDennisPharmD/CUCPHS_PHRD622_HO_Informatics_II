# Student activity | From vaccine NDCs to an illustrative immunization assessment

**Time:** approximately 20–30 minutes. **Teams:** 2–4 students. **Prerequisite:** Python 3 Codespace; no SQL or API coding experience assumed. **All patients are synthetic.**

## Learning objectives

By the end of the activity, you should be able to: (1) distinguish product NDC from vaccine CVX; (2) run SQL to retrieve data from a relational table; (3) describe an HTTP request/JSON response; (4) explain how a normalized terminology mapping table is persisted and joined to clinical data; and (5) discuss why mapping is necessary but insufficient for safe series-completion decisions.

## Part A — Before the API

1. In Codespaces, open a terminal in `Vaccine_CVX_Lab` and run:

   ```bash
   python3 exercise.py reset
   python3 exercise.py query "SELECT * FROM vaccine_administrations ORDER BY patient_id, administered_on"
   python3 exercise.py query "SELECT * FROM ndc_cvx_map"
   ```

2. What does an NDC identify? Why is it stored as `TEXT`? How many vaccine administrations are present? Why is `ndc_cvx_map` empty?
3. Run `python3 exercise.py extract`. Predict how many **distinct** NDCs require translation. Why might we request each NDC only once?

## Part B — Connect to terminology

4. In a **second** terminal, run `python3 api_server.py` and leave it running. Open a browser or the terminal and observe a JSON response:

   ```bash
   python3 -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8765/api/v1/ndc/58160-0823-11').read().decode())"
   ```

   The server is local to Codespaces, but it is a **real HTTP API** returning a CDC-derived teaching mapping.
5. In the first terminal, run:

   ```bash
   python3 exercise.py map
   python3 exercise.py query "SELECT ndc11, cvx_code, cvx_description, mapping_status FROM ndc_cvx_map ORDER BY ndc11"
   ```

6. What CVX code does Shingrix map to? How can different package NDCs map to a single CVX? What happened to the intentionally invalid NDC? Why is treating `404` as **unmapped** safer than assigning a default CVX?

## Part C — Convert data into a decision-support input

7. Run:

   ```bash
   python3 exercise.py query "SELECT p.display_name, a.administered_on, a.ndc11, m.cvx_code FROM patients p JOIN vaccine_administrations a USING(patient_id) LEFT JOIN ndc_cvx_map m ON a.ndc11=m.ndc11 WHERE a.administration_status='given' ORDER BY p.patient_id, a.administered_on"
   python3 exercise.py report
   ```

8. Compare **Jordan Lee**, **Casey Rivera**, **Sam Patel**, **Avery Chen**, and **Riley Park**. Why might two raw administrations *not* represent a completed series? Which cases require manual review? What does `entered-in-error` mean for dose counts?
9. The report uses a **simplified teaching rule** (2 recorded RZV doses at age 50+ separated by ≥28 days). Does `MEETS_DEMO_RULE` mean CDC considers the patient clinically up-to-date? Explain the additional information a production-quality vaccination forecaster would need.

## Part D — Reproducibility and change management

10. Run `python3 exercise.py map` a **second** time. Why are no new requests needed?
11. Run:

    ```bash
    python3 exercise.py reset
    python3 exercise.py query "SELECT COUNT(*) AS saved_mappings FROM ndc_cvx_map"
    ```

    Why are the mappings gone? Identify which file is the immutable starting database and which file is the working database.
12. **Extension challenge:** Inspect `sql/schema.sql`, `sql/seed.sql`, `data/teaching_crosswalk.json` and the `map_ndcs` function in `exercise.py`. Identify where the HTTP boundary, JSON processing, SQL insert, and join occur. Suggest one validation or audit measure you would add before deploying this pattern in a health system.

**Reflection:** NDC-to-CVX crosswalks are reference terminology, not a substitute for current clinical vaccination guidance. The toy API has no security model or service-level commitment and must not be used for medical decisions.
