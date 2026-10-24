# Week 3 Student Activity Answer Key

## Designing a Safe Information Handoff

This guide supports discussion of `week03/student_activity.md`. The activity allows more than one defensible response. Evaluate whether students identify the communication requirement, explain the clinical consequence, and recognize the need for both timely updates and a coherent summary.

## Part 1 Information requirements

The table below represents reasonable responses. Some decisions depend on local workflow and system capability.

| Information | Needed before arrival? | May change after transfer starts? | Must populate a discrete EHR field? | Needs narrative or clinical context? |
| --- | --- | --- | --- | --- |
| Patient identity | Yes | Usually no, although corrections are possible | Yes | Limited context may be needed for patient matching |
| Transfer destination | Yes | Yes | Yes | Usually limited |
| Vital signs and oxygen requirement | Yes | Yes | Preferably | Yes, particularly the trend and response to treatment |
| Lactate results | Yes, including the latest available result | Yes | Yes | Yes, including unit, collection time, result time, reference range, and trend |
| Blood-culture status | Yes | Yes | Yes | Yes, including collection time, source, preliminary status, and prior antibiotics |
| Medication orders | Yes | Yes | Preferably | Yes, including whether the order remains active and its intended plan |
| Medication administrations | Yes | Yes | Yes | Yes, including dose, route, and administration time |
| Allergy information | Yes | Yes if reconciled or corrected | Yes | Yes, especially because the available records conflict |
| Chest radiograph interpretation | Yes | Possibly, if amended or finalized | Preferably | Yes; the report provides context beyond a single coded finding |
| Reason for transfer and clinical course | Yes | May be updated | Not completely | Yes; this is primarily contextual information |
| Pending tests and follow-up needs | Yes | Yes | Preferably when supported | Yes, including who is responsible for follow-up |

### Discussion emphasis

“Must populate a discrete field” should not imply that narrative information lacks value. Several items need both computable fields and narrative context. Students may reasonably select “preferably” rather than “must” for information that systems cannot always import discretely.

## Part 2 Design the handoff

### 1. Which information should reach the receiving organization as soon as an event occurs?

Reasonable examples include:

- transfer acceptance, departure, arrival, or destination changes;
- new or corrected laboratory and microbiology results;
- changes in vital signs or oxygen support that affect preparation for arrival;
- medication administrations, discontinuations, or major order changes;
- allergy reconciliation or other safety-critical corrections; and
- amendments to imaging interpretations.

These items represent events that can occur after someone creates the transfer summary.

### 2. Which information should remain together as a complete clinical summary?

The summary should preserve:

- the reason for admission and transfer;
- the working clinical assessment;
- the course of deterioration and response to treatment;
- active problems;
- relevant allergies and the uncertainty surrounding them;
- medications and treatments provided;
- important results and trends;
- pending tests;
- follow-up responsibilities; and
- authorship, source organization, and time of creation.

The summary communicates relationships among the facts rather than presenting isolated data points.

### 3. Which information belongs in both categories?

Examples include laboratory results, microbiology updates, allergies, medication administrations, imaging results, vital signs, oxygen support, and transfer status.

For example, the receiving system may need a new lactate as a discrete update while the transfer summary should also describe the lactate trend and its clinical significance.

### 4. What context must accompany a laboratory value?

Students should identify most of the following:

- patient and encounter identity;
- test identity;
- numeric or categorical result;
- unit;
- specimen type when relevant;
- specimen collection time;
- result or verification time;
- reference range or abnormal flag when appropriate;
- result status, such as preliminary, final, corrected, or amended;
- performing laboratory or source organization; and
- relationship to prior results when the trend matters.

### 5. What distinctions must the medication information preserve?

Important distinctions include:

- order versus administration;
- ordered, verified, dispensed, started, completed, held, discontinued, or refused status;
- generic or brand medication identity;
- dose, concentration, route, frequency, and administration time;
- one-time dose versus ongoing therapy;
- intended duration or next dose when relevant; and
- source and time of the information.

In this case, the receiving team must know that ceftriaxone and azithromycin were actually administered rather than merely ordered.

### 6. What should happen when information changes after the transfer summary is completed?

The sending organization needs a process for communicating new or corrected events independently of the completed summary. The receiving system should associate the update with the correct patient and encounter, identify it as new or corrected information, and route it to an accountable clinician or workflow.

Depending on the information and local process, the organization might also issue an amended document. Amending a document alone may be insufficient when the receiving team needs immediate notification.

### 7. How should the receiving team know the source and responsible organization?

The exchange should preserve provenance, including:

- sending organization and source system;
- author or responsible clinician when applicable;
- creation, collection, and result times;
- document or result status;
- contact information or escalation pathway; and
- responsibility for pending-result review and follow-up.

## Part 3 Communication complications

### Card 1 A Result Arrives After the Summary

**Problem:** The summary is a point-in-time representation and has become incomplete.

**Readable by a clinician:** The original summary is readable, but it lacks the new result.

**Computable:** The new result could be computable if transmitted with the required structure and context.

**Primary categories:** Content currency, event communication, transport, and workflow responsibility.

**Possible consequence:** The receiving team may underestimate the patient’s deterioration or fail to respond promptly to the rising lactate.

**Needed requirement:** Event-triggered communication of the result, correct patient matching, clear result status, and routing to an accountable clinician. An amended summary may supplement the update.

### Card 2 A Value Without Enough Context

**Problem:** The value lacks the metadata required for interpretation.

**Readable by a clinician:** The clinician can read `4.2` but cannot interpret it confidently.

**Computable:** The computer cannot safely normalize, trend, or evaluate the result without knowing the test identity, unit, and time.

**Primary categories:** Content and structure.

**Possible consequence:** Incorrect comparison with prior results, missed deterioration, or an inappropriate clinical response.

**Needed requirement:** Test identity, value, unit, collection time, result time, reference information, status, patient identity, and source.

### Card 3 Orders and Administrations Are Combined

**Problem:** The exchange does not preserve the status and timing of medication events.

**Readable by a clinician:** The list is readable but ambiguous.

**Computable:** A receiving system cannot reliably determine prior exposure or calculate the next appropriate dose.

**Primary categories:** Content, structure, and workflow.

**Possible consequence:** Duplicate therapy, omitted therapy, incorrect timing, or inaccurate medication reconciliation.

**Needed requirement:** Separate representations of orders and administrations with medication identity, dose, route, status, and timestamps.

### Card 4 Readable but Not Computable

**Problem:** The document supports human review but does not populate the receiving system’s structured fields.

**Readable by a clinician:** Yes.

**Computable:** Limited or no computable use beyond document storage and retrieval.

**Primary categories:** Structure, receiving-system capability, and workflow.

**Possible consequence:** Manual transcription errors, delayed reconciliation, duplicate work, and failure of clinical decision support to use the external data.

**Needed requirement:** Structured or coded entries that the receiving system can interpret, along with reconciliation workflows. The original document should remain available for context.

**Key insight:** Successful document delivery does not establish full semantic or process interoperability.

### Card 5 Discrete Data Without the Clinical Story

**Problem:** Individual data elements arrive without the relationships and reasoning needed to interpret the episode of care.

**Readable by a clinician:** Each item is readable, but the overall story remains unclear.

**Computable:** The receiving system can store and potentially process the discrete elements.

**Primary categories:** Context, wholeness, and clinical workflow.

**Possible consequence:** Repeated assessments, incorrect assumptions about prior treatment response, incomplete follow-up, or failure to appreciate uncertainty in the allergy and infection-source assessments.

**Needed requirement:** A coherent clinical summary that preserves the reason for transfer, clinical course, assessment, response to treatment, pending issues, and follow-up responsibilities.

## Final reflection

### Some clinical information should be communicated as individual events because...

A strong response explains that events and results continue to occur after a summary is created. The receiving system may need timely, discrete updates that can populate the record, trigger workflows, and support automated processing.

### Some clinical information should remain together as a complete summary because...

A strong response explains that isolated data elements do not fully communicate the clinical course, reasoning, relationships among findings, treatment response, uncertainty, pending issues, and responsibility for follow-up.

## Overall takeaway

Messaging standards support timely communication of events and discrete data. Document standards preserve a persistent, coherent, and human-readable clinical record. Safe transitions of care often require both approaches, plus accurate patient matching and workflows that ensure the receiving team reviews and acts on the information.
