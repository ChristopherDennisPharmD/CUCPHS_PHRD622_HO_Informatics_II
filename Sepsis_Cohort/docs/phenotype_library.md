# Phenotype Library

This library describes public concepts for students to operationalize. It is intentionally not an adjudication key. The concepts identify signal domains and reasoning constraints; they do not assign any encounter to a phenotype.

## `sepsis_candidate`

Infection evidence and acute organ dysfunction are both relevant to the encounter, with a temporal relationship that a student-defined algorithm must make explicit. The library should expose:

- suspected or confirmed infection source, cultures, and antimicrobial timing;
- acute changes in respiratory, circulatory, renal, neurologic, hepatic, or coagulation measures;
- supportive interventions and escalation of care;
- time ordering, baseline availability, and documentation uncertainty.

Students must state how they operationalize acuity, baseline, attribution, and the observation window. The public module does not state which synthetic encounters satisfy the concept.

## `infection_without_sepsis_candidate`

Evidence supports an infectious process, but the available encounter signals do not establish the required acute organ dysfunction under the student's specified rule. This concept tests whether an algorithm treats infection alone as sepsis.

Important confounders include chronic abnormalities, mild physiologic changes, treatment started before arrival, and delayed or negative cultures. These are observable teaching signals, not labels.

## `noninfectious_mimic_candidate`

The encounter contains acute physiology or organ dysfunction that can resemble sepsis, while a noninfectious explanation is clinically plausible. Example signal domains include trauma, pancreatitis, pulmonary embolism, medication effect, cardiogenic processes, autoimmune inflammation, and postoperative physiology.

Students should examine whether their method overweights nonspecific inflammation, tachycardia, hypotension, leukocytosis, lactate, or oxygen requirement without adequately testing infection evidence.

## `background_hospitalization_candidate`

The encounter provides realistic inpatient context without a strong sepsis pattern. It may include routine postoperative care, chronic disease management, elective procedures, uncomplicated injury, or another admission with ordinary laboratory and vital-sign variation.

Background encounters should still contain plausible missingness, comorbidities, and care transitions so that negative examples are not recognizable solely by data cleanliness.

## Cross-cutting phenotype rules

- Separate observations from interpretations and preserve event timestamps.
- Define whether a signal is present, absent, unknown, or not measured.
- Distinguish a suspected infection from a confirmed infection.
- Make baseline selection explicit and avoid using post-outcome information.
- Treat competing noninfectious explanations as a modeling question, not an automatic exclusion.
- Record provenance for derived features so another student can reproduce the method.
- Report indeterminate cases and sensitivity analyses rather than forcing every case into a confident category.

The counts in the cohort specification are aggregate design requirements only. Patient-level assignments and instructor adjudication remain outside the public module.