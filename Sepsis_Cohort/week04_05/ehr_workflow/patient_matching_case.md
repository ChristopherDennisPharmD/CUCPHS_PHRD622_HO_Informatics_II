# Patient-Matching Case

The receiving EHR searches for a local record that corresponds to the patient transferred from the community hospital.

## Information received from the sending hospital

| Field | Value |
| --- | --- |
| Name | James Robert Morgan |
| Date of birth | 1958-04-18 |
| Administrative sex | Male |
| Address | 214 Oak Ridge Drive, Cedar Falls |
| Mobile phone | 555-014-8821 |
| Sending-hospital identifier | CH-447190 |

## Possible local records

| Local record | Name | Date of birth | Address | Phone | Additional information |
| --- | --- | --- | --- | --- | --- |
| MRN 008421 | James R. Morgan | 1958-04-18 | 214 Oak Ridge Dr., Cedar Falls | 555-014-8821 | Prior cardiology visit two years ago |
| MRN 011903 | James Robert Morgan | 1958-04-08 | 214 Oak Ridge Drive, Cedar Falls | 555-014-8821 | Created during telephone scheduling last month |
| MRN 019774 | James Morgan Jr. | 1986-09-22 | 52 Pine Street, Cedar Falls | 555-019-3370 | Emergency visit last year |

## Discuss before opening the information inventory

1. Which fields support a possible match?
2. Which discrepancies require investigation?
3. Is the evidence sufficient for automatic matching, manual review, or creation of a new record?
4. Could MRN 008421 and MRN 011903 be duplicates? What harm could result from merging them incorrectly? From failing to merge true duplicates?
5. Which identifiers should appear prominently when a clinician opens the imported information?

Do not assume that similar names or a single matching field establish identity.
