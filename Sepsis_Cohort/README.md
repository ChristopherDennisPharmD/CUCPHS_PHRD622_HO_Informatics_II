# Synthetic Sepsis Cohort

This self-contained module supports the longitudinal redesign of PHRD-622 Health Outcomes and Informatics II around sepsis informatics. It defines a public synthetic-cohort specification for 100 hospitalized encounters:

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

## Student-facing use

Students may use the specifications to design data dictionaries, map HL7-style observations, build phenotype logic, assess missingness, compare rule-based and probabilistic approaches, and report uncertainty. A future generated data release should be distributed separately from this design module and should not include a truth column or a recoverable answer key in the student fork.

## Public-data boundary

This repository intentionally contains definitions and aggregate targets only. It does not contain patient IDs, encounter-level observations, outcome labels, split assignments, random seeds, or adjudication notes. Any instructor evaluation set should be stored and distributed through a private assessment workflow.