# Week 2 Student Activity

## Can We Reliably Find the Same Clinical Information?

The health system has organized data from multiple hospitals and clinical systems into one repository. It now wants to identify every encounter in which a patient had an elevated lactate, a suspected infection, and received ceftriaxone.

Review `mixed_source_extract.xlsx`. You do not need to know specific terminology codes or calculate unit conversions. Focus on how the information is represented.

## Part 1 Group related records

1. Which records appear to represent a lactate measurement or interpretation?

2. Which records appear to represent ceftriaxone?

3. Which medication record should not be grouped with ceftriaxone solely because its name looks similar?

4. Which records appear related to pneumonia? Do they all represent exactly the same clinical assertion?

## Part 2 Identify comparison barriers

For each issue below, identify at least one example from the extract.

| Potential issue | Example record | Why it matters |
| --- | --- | --- |
| Different names for a similar concept |  |  |
| Different units |  |  |
| Missing unit |  |  |
| Narrative information without a discrete value |  |  |
| Local abbreviation or code |  |  |
| Different levels of clinical detail |  |  |

5. Which lactate results can be compared directly without additional processing?

6. Which records require a mapping table, unit conversion, or additional context?

7. If the system searched for `local_display = "Lactate"`, which relevant records would it miss?

8. If the system searched for `local_display = "Ceftriaxone"`, which potentially relevant records would it miss?

## Part 3 Select the type of standard

Match each information need to the most relevant standard. A standard may be used more than once, and some records may require more than one type of standardization.

| Information need | Relevant standard |
| --- | --- |
| Identify the laboratory observation that was measured |  |
| Represent pneumonia as a clinical concept |  |
| Classify pneumonia for administrative reporting |  |
| Normalize a clinical medication concept across brand and generic names |  |
| Identify a specific manufacturer product and package |  |

Standards to consider: **LOINC, SNOMED CT, ICD-10-CM, RxNorm, and NDC**.

## Part 4 Consider the limitations

9. If all local terms map to standards, what data-quality problems would remain?

10. What could happen if a local term maps to the wrong standard concept?

11. Why might the original display term and source-system provenance still be worth retaining?

12. Complete this statement in your own words:

> Standardized data make cohort retrieval more reliable because...
