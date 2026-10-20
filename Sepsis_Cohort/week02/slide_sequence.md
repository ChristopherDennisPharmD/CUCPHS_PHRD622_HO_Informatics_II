# Week 2 Slide Sequence

These slide-ready prompts can be inserted into the existing Data Standards Part 1 lecture. They are designed as short pauses within the lecture rather than a separate presentation.

## Slide 1 The Health System Question

**Title:** Finding Patients Across Multiple Systems

Our health system wants to identify every encounter in which a patient:

- had an elevated lactate; and
- received ceftriaxone.

The data come from several hospitals, laboratories, medication records, and clinical notes.

**Discussion prompt:** What could make this apparently simple search difficult?

## Slide 2 Different Representations of Similar Concepts

**Title:** Local Clinical Representations

Display selected rows from `data/mixed_source_extract.csv`, including:

- Lactate, Lactic Acid, LACT, and “lactate elevated”
- Ceftriaxone, Rocephin, CEFTRX, and NDC-A
- Pneumonia, J18.9, bacterial pneumonia, and right lower-lobe infiltrate

**Discussion prompt:** Which entries belong together, and which only appear related?

## Slide 3 Exact-Text Search

**Title:** What Exact-Text Matching Misses

Searching only for `Lactate` misses:

- alternate test names;
- local abbreviations;
- narrative documentation; and
- coded observations.

Searching only for `Ceftriaxone` misses brand names, local abbreviations, and product identifiers.

**Discussion prompt:** What happens if we make the search too broad?

## Slide 4 Standards by Information Need

**Title:** Different Standards Serve Different Purposes

| Information need | Standard |
| --- | --- |
| Laboratory observation | LOINC |
| Clinical diagnosis or finding | SNOMED CT |
| Administrative diagnosis classification | ICD-10-CM |
| Normalized clinical medication | RxNorm |
| Packaged medication product | NDC |

**Teaching point:** No single code system represents every aspect of the clinical record.

## Slide 5 Mapping and Normalization

**Title:** Standardization Requires More Than a Name

For reliable comparison, the system may need to:

- map local terms to standard concepts;
- normalize compatible units and doses;
- preserve missing or uncertain information; and
- retain the original source and display value.

Use R001 through R009 from `standardized_reveal.md` as examples.

## Slide 6 Remaining Limitations

**Title:** What Standards Do Not Resolve

Standards do not automatically correct:

- inaccurate source data;
- missing units or timestamps;
- ambiguous narrative documentation;
- incorrect local mappings; or
- differences in clinical certainty and specificity.

**Closing prompt:** Did standardization make the data correct, or did it make the data more consistently interpretable?

## Slide 7 Next Information Challenge

**Title:** Exchanging Standardized Information

The health system can represent the concepts more consistently. It must now exchange those data among the laboratory, pharmacy, radiology, and EHR systems without losing structure or meaning.

This introduces clinical documents, messaging standards, and health information exchange.
