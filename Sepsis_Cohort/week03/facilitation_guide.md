# Week 3 Facilitation Guide

## Purpose

This opening activity creates the need for messaging and document standards before students learn the technical terminology. It should focus on communication requirements, information loss, and clinical workflow. Save HL7 v2 structure and parsing for the post-lecture group activity.

## Recommended time

| Segment | Time | Instructor action |
| --- | ---: | --- |
| Reconnect to the longitudinal case | 2 minutes | Summarize the patient’s deterioration and planned transfer. |
| Review the information inventory | 2 minutes | Emphasize that the information resides in different systems. |
| Design the handoff | 5 minutes | Groups complete the first two parts of `student_activity.md`. |
| Introduce complications | 5-7 minutes | Reveal cards sequentially or assign one card to each group. |
| Debrief and transition | 3-4 minutes | Organize responses into the two communication needs and introduce the lecture. |

## Opening prompt

Tell students:

> The patient is moving from one organization to another. The receiving ICU must prepare for the patient and continue care without repeating every assessment or losing important information. Design the information handoff based on what the receiving team needs. You do not need to know a technical standard.

## Facilitation approach

Avoid introducing the terms `HL7`, `CDA`, `CCD`, `message`, or `document standard` before the initial group discussion. Students should first identify the underlying needs.

When groups report, organize their recommendations into two unlabeled columns:

- information that should arrive as individual events or updates; and
- information that should remain together as a complete clinical summary.

Some items belong in both columns. For example, a laboratory result may need to arrive as a discrete update while also appearing in a transfer summary.

## Expected observations

Students should recognize that:

- the transfer summary represents the information available when someone creates it and may become outdated;
- new results require a way to reach the receiving organization after the summary is complete;
- values need identifiers, units, timestamps, provenance, and clinical status to remain meaningful;
- medication orders and medication administrations represent different events;
- a readable document may not supply data that the receiving system can use computationally;
- discrete data do not always communicate clinical reasoning or the full course of care;
- technical delivery does not ensure patient matching, reconciliation, review, or action.

## Introducing the terminology

After the debrief, label the two columns:

- **Messaging standards:** timely communication of events, results, and transactions in an expected structure.
- **Document standards:** persistent communication of a coherent clinical record that preserves context and remains human-readable.

Clarify that a message or exchange service may transport a clinical document. The distinction concerns how the clinical content is organized and used, not simply the network pathway that carries it.

## Transition into the lecture

Use this transition:

> You have described two different but complementary interoperability needs. We will now examine the standards healthcare developed to address them, beginning with the technical layers that allow systems to communicate and then considering clinical documents, messages, and exchange networks.

## Connection to the post-lecture HL7 activity

Before the later group exercise, return to the case:

> Earlier, you said the receiving system needed predictable fields for the patient identity, result, value, unit, and time. The HL7 v2 activity will show one way a messaging standard organizes those elements so software can locate and use them.

The parser exercise should demonstrate the solution to a problem students have already encountered rather than function as a separate technical demonstration.
