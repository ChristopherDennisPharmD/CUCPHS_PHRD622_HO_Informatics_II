# EHR Workflow Debrief

## Information is not useful merely because it arrived

| Requirement | Why it matters in this case |
| --- | --- |
| Correct patient | A result attached to the wrong record can drive unsafe treatment; duplicate records can hide relevant history. |
| Preserved meaning | A medication order is not the same event as an administration; a pending culture is not a final negative result. |
| Provenance | Clinicians need to know who created the information and whether it is local or external. |
| Time and status | Collection, result, receipt, and update times answer different clinical questions. |
| Visible uncertainty | Conflicting allergy entries and unavailable images require reconciliation, not silent selection of one value. |
| Appropriate presentation | Trends, summaries, details, and warnings serve different tasks. |
| Workflow connection | Important information needs an owner, an action, and a way to document completion. |

## No single display is best for every user

The pharmacist may prioritize antimicrobial administrations, allergies, renal function, cultures, and response to therapy. A nurse or physician may need a different arrangement. Good EHR design begins with the user's task rather than displaying every available field equally.

## Missing is not normal

The interface should distinguish among:

- not measured;
- measured but not received;
- received but not reconciled;
- pending;
- unavailable from the source; and
- measured and within the expected range.

## Closing statement

> Interoperability delivers information. Patient matching, preserved meaning, thoughtful presentation, and workflow integration make it safe to use.
