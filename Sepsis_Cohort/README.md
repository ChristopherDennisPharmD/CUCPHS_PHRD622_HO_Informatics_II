# Synthetic Sepsis Cohort

This self-contained module supports the longitudinal redesign of PHRD-622 Health Outcomes and Informatics II around sepsis informatics. It defines a public synthetic-cohort specification for 100 hospitalized encounters and provides weekly demonstrations showing how informatics capabilities make clinical information easier to collect, interpret, exchange, and use:

- 4 encounters meeting the course sepsis phenotype
- 16 infections without sepsis
- 20 noninfectious sepsis mimics
- 60 background hospitalizations

The module is a teaching specification, not a clinical decision-support tool. All data described here are synthetic. No real patient information, patient-level ground-truth labels, encounter assignments, or instructor answer keys belong in the public repository.

## Module map

- `docs/cohort_design.md` - cohort structure, longitudinal design, fields, and teaching workflow.
- `docs/phenotype_library.md` - public phenotype concepts, observable signals, and ambiguity rules.
- `config/cohort.yml` - machine-readable cohort totals, encounter schema, and data-governance rules.
- `config/phenotypes.yml` - machine-readable phenotype concepts and signal domains.
- `week01/` - introductory demonstration showing how fragmented information limits clinical review.
- `week02/` - data-standards demonstration showing how local representations affect cohort retrieval.
- `week03/` - interoperability demonstration showing why clinical exchange requires both messaging and document standards.

## Student-facing use

Students use small, curated subsets of the synthetic cohort to experience clinical-information problems before learning the informatics concepts that address them. The initial modules ask students to review fragmented records, compare inconsistent data representations, and identify the different communication needs served by messaging and document standards. Later activities may use the broader cohort to develop phenotype logic, assess missingness, compare rule-based and probabilistic approaches, and report uncertainty.

A future generated data release should be distributed separately from this design module and should not include a truth column or a recoverable answer key in the student fork.

## Public-data boundary

This public repository may contain fictional encounter identifiers and selected synthetic observations needed for demonstrations and student activities. It does not contain real patient information, the full generated cohort with recoverable patient-level assignments, outcome labels, split assignments, generation seeds, or adjudication notes. Any instructor evaluation set or answer key should be stored and distributed through a private assessment workflow.
