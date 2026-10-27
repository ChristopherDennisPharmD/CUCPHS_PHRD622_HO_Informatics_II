# Instructor guide | Rx-Mix-a-Lot vaccine terminology add-on

## Setup and timing

- Confirm the folder is included in the class repository; opening it in any Python 3 Codespace is sufficient. **Do not replace a repository-level `.devcontainer`**, since this module needs only Python's standard library.
- The class has two terminals: A `python3 api_server.py` (HTTP/JSON), B `python3 exercise.py reset` then `extract`, `map`, `report`. No internet, pip install, or API token needed once the repo is cloned.
- Suggested facilitation: 5 min database, 7 min HTTP mapping, 7 min SQL join/report, 5 min reflection; add 5 min for alternative clinical scenarios.
- Never distribute or commit real patient data. `data/workspace.sqlite` is ignored by Git and should be treated as disposable.

## Answer key / expected teaching outcomes

| Patient | Expected teaching status after mapping | Teaching point |
| --- | --- | --- |
| Jordan Lee (P001) | `MEETS_DEMO_RULE` | Two doses, different NDC packaging, both CVX 187, spaced ~3 months |
| Casey Rivera (P002) | `INCOMPLETE_DEMO` | One recorded RZV dose |
| Sam Patel (P003) | `REVIEW_INTERVAL` | Two RZV records 14 days apart |
| Avery Chen (P004) | `REVIEW_UNMAPPED` | Deliberately unknown NDC means the source history is unresolved |
| Morgan Blake (P005) | `NO_RZV_RECORDED` | HPV9 CVX 165 is not a zoster dose |
| Riley Park (P006) | `INCOMPLETE_DEMO` | Second RZV is `entered-in-error` and should not count |
| Taylor Singh (P007) | `MEETS_DEMO_RULE` | Two different Shingrix NDCs resolve to CVX 187 |

- Source rows = **13** administrations among **7** patients.
- Distinct **7** NDCs in given administrations (six known + one deliberate unknown).
- Prior to mapping, `ndc_cvx_map` has **0** rows; afterward it has **7** rows: **6 mapped, 1 unmapped**.
- Repeat mapping makes **zero** new requests because mappings, including explicit unmapped decisions, have been persisted. A real application must define **refresh policies and terminology versioning**; lack of repeat requests here is a class convenience, not a production cache policy.
- The local API supports `GET /api/v1/ndc/{ndc11}` and `GET /health`. Successful mappings return `200` + JSON; unknown codes return `404` + JSON; HTTP or timeout errors are not silently stored as an unrecognized NDC.

## Discussion prompts

- Product versus concept: why is a specific Shingrix package identifier not the same thing as CVX 187? Discuss CDC's unit-of-sale (carton) and unit-of-use (vial/syringe) distinctions.
- Clinical documentation: How does `LEFT JOIN` preserve unresolved records? What would an `INNER JOIN` inadvertently hide?
- Data quality: How would incorrect NDC padding, a retired NDC, an inaccurate administration status or a duplicate record affect an algorithm?
- Clinical safety: Why is counting two CVX 187 codes not enough to conclude that a series is complete? Discuss patient eligibility, time intervals, dose validity, incomplete external histories, changing guidelines and appropriate human review.
- Data provenance: Why persist mapping source, result status and retrieval timestamp? What other metadata would a production terminology service need (version, effective date, authoritativeness, package type, conflicts, audit)?

## Versioning and refresh

`sql/schema.sql` and `sql/seed.sql` are the editable source of the initial database; `data/starter.sqlite` is the committed, pristine teaching artifact. After editing the SQL, regenerate via `python3 build_starter.py`, verify `python3 -m unittest discover -s tests -v`, and commit all changed files. The crosswalk fixture is deliberately frozen so results are reproducible between semesters. Review it against the current [CDC NDC crosswalk](https://www2.cdc.gov/vaccines/iis/iisstandards/vaccines.asp?rpt=ndc) before presenting it as a contemporary example.

**Live CDC API extension:** CDC offers a terminology REST API documented at https://www.cdc.gov/iis/code-sets/viewpoint-rest.html. The CDC terminology data model separates NDC UoU from UoS, so connecting a production API would require deliberately resolving NDC packaging and navigating related concepts. This bundled exercise does **not** claim the local API is an official live CDC service and intentionally does not include an unverified production-service integration. Students can propose a replacement provider adapter as an advanced assignment.

**Clinical caution:** The 28-day minimum interval in the *demonstration* parallels one narrow aspect of published guidance. CDC generally recommends 2–6 months between Shingrix doses for routine immunocompetent adults aged 50+, and distinguishes other indications. Do not apply this demo to clinical care; full immunization assessment needs maintained CDS algorithms and reliable health histories. Reference: https://www.cdc.gov/shingles/hcp/vaccine-considerations/index.html
