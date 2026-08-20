# Cohort Design

## Purpose

The cohort is a longitudinal, synthetic inpatient dataset for teaching clinical data modeling and phenotype evaluation. It is designed to make infection, organ dysfunction, treatment response, documentation quality, and competing explanations visible as separate data problems.

## Aggregate composition

The public design target is 100 hospitalizations. The four groups are mutually exclusive for evaluation purposes and are described only in aggregate:

| Public design group | Count | Teaching purpose |
| --- | ---: | --- |
| True sepsis encounters | 4 | Test recognition of infection plus acute organ dysfunction |
| Infection without sepsis | 16 | Test specificity and the difference between infection and sepsis |
| Noninfectious sepsis mimics | 20 | Test confounding, differential diagnosis, and false positives |
| Background hospitalizations | 60 | Provide prevalence context and realistic negative examples |
| **Total** | **100** | |

These group names describe the design strata, not fields that should appear in a student dataset.

## Longitudinal unit of analysis

The primary unit is a hospitalization. Each hospitalization may contain multiple timestamped observations, orders, results, notes, and disposition events. Students should be able to aggregate events into an encounter-level analytic table while retaining event-level provenance.

Recommended time windows are:

- **Baseline:** the first 6 hours after admission or presentation.
- **Early course:** 6 to 24 hours after admission.
- **Hospital course:** after 24 hours until discharge.
- **Follow-up:** a synthetic post-discharge window when an activity requires readmission or mortality modeling.

All timestamps should use one documented timezone and should be internally consistent. Relative hours from admission may be supplied alongside timestamps to make temporal reasoning reproducible.

## Observable domains

The generated data should represent a mixture of structured and narrative-friendly signals:

- Demographics and comorbidity context.
- Admission source, presenting complaint, and vital signs.
- Laboratory results, specimen collection, and culture status.
- Suspected or confirmed infection source and antimicrobial exposure.
- Organ-function measures and supportive interventions.
- Fluid, vasopressor, oxygen, and level-of-care events.
- Documentation, missingness, delayed results, and conflicting signals.
- Disposition and selected utilization outcomes.

Values should include plausible measurement units, reference ranges where useful, and explicit missingness reasons when a value is unavailable. Synthetic noise should include timing differences and documentation gaps without making the intended phenotype impossible to study.

## Student activities supported

The module can support a sequence of activities without changing existing course artifacts:

1. Build a data dictionary and event-to-encounter relational model.
2. Parse and normalize HL7-like event fields.
3. Define transparent rule-based phenotype logic and identify borderline cases.
4. Compare sensitivity, specificity, predictive value, and calibration under class imbalance.
5. Examine missing-not-at-random documentation and temporal leakage.
6. Communicate a phenotype specification, limitations, and an uncertainty-aware result.

Students should submit methods and reasoning, not a guessed patient-level truth table.

## Data-generation guardrails

Generated files should use synthetic encounter keys, not names, dates that could be mistaken for real records, or external identifiers. The public release should not include a column such as `truth_label`, `stratum`, `answer`, or an equivalent encoded assignment. Evaluation labels, generation seeds, and adjudication notes belong in a private instructor workflow.