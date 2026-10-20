# Week 2 Facilitation Guide

## Purpose

This demonstration shows why electronic data can remain difficult to retrieve and compare. Students should recognize that humans use clinical knowledge to connect brand names, abbreviations, units, and related findings, while a computer requires explicit representations and mappings.

The activity reinforces the terminology section of Data Standards Part 1. It does not require students to memorize code values or perform programming tasks.

## Recommended placement

Use the activity after introducing terminology standards and before or during the discussion of specific standards such as SNOMED CT, LOINC, NDC, and RxNorm. The diagnosis examples can also support the distinction between SNOMED CT and ICD-10-CM.

## Recommended time

| Segment | Time | Instructor action |
| --- | ---: | --- |
| Reconnect to Week 1 | 2 minutes | Remind students that organizing the records improved review but did not make the data consistent. |
| Introduce the retrieval question | 2 minutes | Ask how the health system could find elevated lactate plus ceftriaxone across all encounters. |
| Group review | 10 minutes | Students complete Parts 1 and 2 of the activity. |
| Standards matching | 6 minutes | Students complete Part 3 after the relevant lecture content. |
| Structured reveal | 7 minutes | Review the standardized tables and exact-text search limitations. |
| Debrief | 5 minutes | Discuss what standardization solves and what remains uncertain. |

The full exercise requires approximately 32 minutes. It can be divided into short segments throughout the lecture.

## Expected observations

Students should notice that:

- R001 through R004 contain lactate-related information, but only R001 and R002 can be normalized confidently from the supplied extract.
- R003 lacks a unit, and R004 lacks a discrete numeric result.
- R006 through R009 may relate to ceftriaxone, but R008 requires validation of the local abbreviation and R009 requires a product crosswalk plus administration details.
- R010 is cefazolin and should not be grouped with ceftriaxone based on spelling or drug class alone.
- R011 through R014 concern related clinical concepts but do not make identical assertions.
- Exact string matching sacrifices recall, while overly broad grouping can sacrifice specificity.

## Standards discussion

Guide students toward these associations:

| Information need | Standard |
| --- | --- |
| Laboratory observation identity | LOINC |
| Clinical diagnosis or finding | SNOMED CT |
| Administrative diagnosis classification | ICD-10-CM |
| Normalized clinical medication | RxNorm |
| Manufacturer product and package | NDC |

Do not require students to locate actual codes. The exercise concerns why the standards exist and how their purposes differ.

## Debrief prompts

1. Which problems could terminology mapping solve?
2. Which problems require unit conversion or additional data?
3. How could an incorrect mapping create a patient-safety or analytics problem?
4. Why should the system retain the original term and source after standardization?
5. What should the system do when it cannot map a record confidently?

## Closing takeaway

> Standards allow systems to connect clinical information that uses different local representations. Reliable reuse also depends on accurate mappings, compatible units, sufficient context, and preserved provenance.

## Transition to Week 3

End with the next problem:

> We can now represent the concepts more consistently. How do the laboratory, pharmacy, radiology, and EHR systems exchange these standardized data without losing their structure or meaning?
