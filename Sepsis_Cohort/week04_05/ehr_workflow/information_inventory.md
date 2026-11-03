# Imported Information Inventory

Assume the identity question has been sent for appropriate review. The following items are available to the receiving EHR.

| Item | Representation and status | Source | Clinical issue |
| --- | --- | --- | --- |
| Transfer summary | Readable clinical document finalized at 13:00 | Community hospital | Coherent clinical course, but created before the repeat lactate resulted |
| Repeat lactate | Discrete result: 4.2 mmol/L, collected 12:50, resulted 13:25 | Community laboratory | Time-sensitive result that arrived after the summary |
| Medication list | Structured list containing ceftriaxone, azithromycin, and normal saline | Community pharmacy system | Does not clearly distinguish active orders from completed administrations |
| Medication administrations | Ceftriaxone at 10:42 and azithromycin at 10:47 | Community medication record | Needed to determine timing and next doses |
| Allergy history | Free text: `PCN - rash?`; date and verification status absent | Prior community clinic record | Conflicts with local entry of no known medication allergies |
| Blood cultures | Collected 09:35; no growth to date; final result pending | Community microbiology system | Status may change after admission |
| Chest radiograph | Report available; image not available | Community radiology system | Report supports pneumonia, but local team cannot review the image |
| Vital signs | Discrete observations through 12:40 | Community nursing flowsheet | No observations documented during transport |
| Outside patient identifier | CH-447190 | Community registration system | Not the same identifier as the local MRN |

## Interoperability lens

For selected items, consider which level has been achieved:

- **Foundational:** information can be sent and received.
- **Structural:** the format and organization are predictable.
- **Semantic:** the receiving system can preserve and interpret the intended meaning.

An item may meet one level for one use and still be inadequate for another. A readable transfer summary, for example, may support human review without supplying discrete values for trending or automated logic.
