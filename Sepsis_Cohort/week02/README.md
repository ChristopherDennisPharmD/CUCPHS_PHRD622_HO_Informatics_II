# Week 2 Demonstration

## Data Standards Part 1

This demonstration continues the longitudinal sepsis case by moving from fragmented records to inconsistent data representations. Students examine a small extract from several source systems and consider why a human can recognize related clinical concepts that a computer may treat as unrelated.

The activity does not require code lookup, programming, or manual unit conversion. Students group related entries, identify barriers to comparison, and consider which standards could preserve the meaning of laboratory, medication, and diagnosis data.

## Learning objectives

By the end of the demonstration, students should be able to:

1. Explain why electronic data may remain difficult to compare or retrieve.
2. Distinguish a local display term from a standardized clinical concept.
3. Match common data domains to relevant standards, including LOINC, SNOMED CT, ICD-10-CM, RxNorm, and NDC.
4. Explain why terminology mapping and unit normalization are separate tasks.
5. Recognize that standardization improves interpretation and retrieval but does not guarantee data quality.

## Module contents

- `facilitation_guide.md` - recommended timing and discussion prompts.
- `student_activity.md` - group instructions and response questions.
- `data/mixed_source_extract.xlsx` - source-system records used in the activity.
- `standardized_reveal.md` - organized representation revealed after discussion.
- `slide_sequence.md` - concise slide-ready prompts and teaching points.

## Central question

> Our health system wants to find every patient with an elevated lactate who received ceftriaxone. Which records would an exact-text search miss, and what would need to change before the search could work reliably?

All patients, identifiers, and source-system values in this module are synthetic. The codes labeled as local or NDC-like identifiers are illustrative placeholders and should not be interpreted as real product identifiers.
