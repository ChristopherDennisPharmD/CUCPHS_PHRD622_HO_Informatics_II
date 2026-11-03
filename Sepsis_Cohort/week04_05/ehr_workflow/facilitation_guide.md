# EHR Workflow Facilitation Guide

## Purpose

This activity moves from exchange to use. Students encounter three linked design problems: matching information to the correct patient, assessing whether imported data retain meaning, and presenting relevant information within a pharmacist's workflow.

## Recommended time

| Segment | Combined format | Standalone format |
| --- | ---: | ---: |
| Introduce the workflow problem | 2 minutes | 3 minutes |
| Patient-matching case | 5 minutes | 7 minutes |
| Review information inventory | 4 minutes | 6 minutes |
| Design pharmacist view | 7 minutes | 9 minutes |
| Report out and debrief | 4 minutes | 5 minutes |

## Opening prompt

> The organization is permitted to use this information for the patient's care. That does not guarantee safe use. Before the pharmacist acts, the EHR must associate the information with the correct patient, preserve what it means, and present it at the right point in the workflow.

## Stage 1: Patient matching

Do not ask students to calculate a matching score. Ask what evidence increases or decreases confidence and what should happen when confidence is insufficient.

Expected observations:

- MRN 008421 is the strongest candidate because name, date of birth, address, and phone align.
- MRN 011903 may be a duplicate with an incorrect birth day, or it may represent another person using shared contact information.
- Names and phone numbers alone are insufficient.
- Both a false merge and an unresolved duplicate can cause harm.
- An uncertain match should follow a defined manual-review or identity-management workflow.

## Stage 2: Imported information

Invite groups to identify where successful transport still leaves ambiguity:

- the transfer summary is readable but time-bound;
- the lactate is discrete and trendable but arrived later;
- the medication list does not preserve order-versus-administration status;
- the allergy entry is readable but uncertain and conflicting;
- the radiology report is available while the image is not;
- the culture result is provisional and may change; and
- the outside identifier requires mapping rather than display as a local MRN.

Use the foundational, structural, and semantic labels after students describe what the systems can and cannot do.

## Stage 3: Pharmacist view

There is no single correct layout. Strong designs usually:

- keep identity, source, and recency visible;
- place actual administrations near orders;
- show lactate, renal function, vital signs, and oxygen as trends;
- preserve the transfer summary for context without making it the only source;
- distinguish pending, missing, external, and unverified information;
- elevate the allergy conflict as a reconciliation task; and
- support an action such as documenting review, recommending therapy, or assigning follow-up.

## Debrief prompts

- What did you remove from immediate view, and why?
- Which value could be technically correct but clinically misleading?
- How would the design fail if provenance disappeared?
- Who owns a pending result after transfer?
- How could too many warnings make the important warning less visible?

## Transition to clinical decision support

> The EHR now gives the pharmacist an appropriate and usable view once the chart is opened. Our next question is whether the system can identify which patients need review before the pharmacist opens the chart.
