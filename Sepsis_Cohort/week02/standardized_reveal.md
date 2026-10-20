# Standardized Data Reveal

The tables below show how the local representations could be organized for comparison. They illustrate the mapping process without assigning real terminology codes.

## Laboratory observations

| Record | Local representation | Standardized concept | Normalized value | Remaining limitation |
| --- | --- | --- | --- | --- |
| R001 | Lactate 3.6 mmol/L | Lactate measurement, represented with LOINC | 3.6 mmol/L | None apparent in the extract |
| R002 | Lactic Acid 32.4 mg/dL | Lactate measurement, represented with LOINC | Approximately 3.6 mmol/L | Conversion method must be documented |
| R003 | LACT 3.5, no unit | Possible lactate measurement | Cannot normalize safely | Unit and device configuration must be verified |
| R004 | “Lactate elevated” | Narrative statement about lactate | No numeric value | Cannot determine magnitude or threshold from the extract |
| R005 | Lactate 1.7 mmol/L | Lactate measurement, represented with LOINC | 1.7 mmol/L | Result does not meet the listed local elevated threshold |

**Teaching point:** Mapping the test name and normalizing the measurement unit require separate decisions.

## Medications

| Record | Local representation | Standardized concept | Normalized dose | Remaining limitation |
| --- | --- | --- | --- | --- |
| R006 | Ceftriaxone 2 g IV | Ceftriaxone clinical drug, represented with RxNorm | 2,000 mg IV | Product and formulation detail may require additional fields |
| R007 | Rocephin 2,000 mg IV | Ceftriaxone clinical drug, represented with RxNorm | 2,000 mg IV | Brand name differs from normalized clinical concept |
| R008 | CEFTRX 2 GM IV | Ceftriaxone clinical drug after validating the local abbreviation | 2,000 mg IV | Incorrect local mapping would misclassify the drug |
| R009 | NDC-A | Packaged product represented by an NDC-like identifier | Dose unavailable | A crosswalk and administration details are needed |
| R010 | Cefazolin 1 g IV | Cefazolin clinical drug | 1,000 mg IV | Similar spelling does not make it ceftriaxone |

**Teaching point:** RxNorm supports normalization of clinical drug concepts, while NDC identifies packaged products. A crosswalk may connect them, but they serve different purposes.

## Diagnoses and findings

| Record | Local representation | Possible standard representation | Important distinction |
| --- | --- | --- | --- |
| R011 | Pneumonia | SNOMED CT clinical concept | Broad clinical problem-list entry |
| R012 | J18.9 | ICD-10-CM classification | Administrative classification for pneumonia with an unspecified organism |
| R013 | Right lower-lobe infiltrate | Narrative or coded imaging finding | A finding that may support pneumonia but does not independently assert the diagnosis |
| R014 | Bacterial pneumonia | More specific SNOMED CT clinical concept | Contains more clinical detail than the broad pneumonia entry |

**Teaching point:** Related records can differ in meaning, certainty, specificity, and intended use. Standardization should preserve those distinctions.

## Exact-text search compared with standardized retrieval

An exact-text search for `Lactate` and `Ceftriaxone` would retrieve only the records whose local display strings match those terms. It would miss brand names, abbreviations, alternate laboratory names, narrative statements, and product identifiers.

A standards-based approach can retrieve additional relevant records after the organization:

1. validates local-to-standard mappings;
2. normalizes compatible units and doses;
3. preserves missing or uncertain information;
4. retains provenance so reviewers can trace each standardized value to its source.

Standardization improves retrieval and comparison. It does not correct inaccurate source data, supply missing context, or guarantee that a mapping decision is clinically appropriate.
