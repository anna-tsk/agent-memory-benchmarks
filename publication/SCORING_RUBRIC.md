# Frozen scoring rubric — review version 1

Prepared 2026-11-02. **No new answers exist and no model calls are authorized.** This version is frozen by SHA-256 in `freeze_manifest.json`. Human review can produce a new numbered version before execution; retain this version and log every change. Freeze the accepted pair manifest, rubric and protocol together before any query extraction or answer call. This is an exploratory, deliberately selected sample, not a preregistered confirmatory test.

## Ground truth and scope

The unit is one original LoCoMo question, identified by conversation ID and **zero-based index in the entire QA array**, not its index among Category-5 questions. Use the manually reviewed text-only conversation, including speaker attribution and session dates. Dataset category and `adversarial_answer` are provenance, never scoring gold. No image, caption, external fact or previous model answer can establish an answer here.

The validation sheet assigns four possible ground-truth classes:

| Class | Meaning | Correct behavior |
|---|---|---|
| Supported (S) | The requested answer follows from the text, with sufficient attribution and qualifiers. | Answer the question correctly. |
| Unsupported (U) | The record does not establish the requested proposition/detail or its negation. | State that the requested information is unavailable, possibly distinguishing a fact about another person. |
| Contradicted premise (C) | The recorded event/premise is explicitly contradicted, and a corrective answer is supported. | Correct the premise; do not merely refuse. |
| Ambiguous (A) | Wording, source conflict, missing modality or uncertain entailment prevents a stable target. | Replace before execution and log why; never force it into U. |

The proposed final set contains **12 S, 10 U and 2 C** items. Both C items are scoped to explicit denials in recorded exchanges, with no subsequent update in the conversation. Neither denial establishes what could happen outside the record. The 12 historically Category-5 members are therefore called the **challenge** members, not twelve unanswerable questions.

An incorrect presupposition is not automatically a known false proposition. Another person's ownership or experience usually does not prove the queried person's nonownership/nonexperience. For U items, a qualified correction such as “The account describes Evan's Prius; it doesn't say what Sam bought” can pass. An unsupported absolute such as “Sam never owned a car” cannot.

## Scoring each response

Read the response against the item-specific rule and full source, without seeing the condition or retrieved context. Record `primary_outcome`, `strict_correct` (1/0), `unsupported_assertion` (yes/no), `abstention` (yes/no), `unnecessary_abstention` (yes/no), any ambiguity, and a brief rationale with source IDs. Flags can overlap; do not force mixed answers into a single favorable label.

| Primary outcome | Score | Rule |
|---|---:|---|
| Correct supported answer | 1 | Meets the item's required facts and attribution, without a material unsupported addition. |
| Justified abstention | 1 on U only | Explicitly states the target is unknown/not given; no guessed target answer. A clearly attributed contrast may accompany it. |
| Correct premise correction | 1 on C; also acceptable on U if qualified | Gives the supported correction required by the item. On U, must avoid claiming the unknown proposition is conclusively false. |
| Unsupported assertion / wrong attribution | 0 | Supplies an ungrounded target value, transfers another person's fact, contradicts the record, or adds a material invented fact. |
| Unnecessary abstention | 0 on S or C | Refuses despite a supported answer/correction. “Not mentioned” alone fails both C items. |
| Incorrect or materially incomplete answer | 0 | Omits required information, gives a wrong supported fact, or equivocates between incompatible answers. |
| Unscorable response / operational failure | 0 in primary analysis | Empty output, irrecoverable truncation, invalid/missing response or unresolved content ambiguity; record the reason separately. |

Equivalent paraphrases pass. Confidence language is not itself an error: a correct supported answer phrased “I think…” may pass. A disclaimer followed by an ungrounded guess (“not stated, probably Sweden”) fails U, even if it contains an abstention phrase. A correct fact about another person passes only if it clearly answers/corrects the queried attribution; an orphan fact that does not address the question is incomplete. No credit for “No” unless its referent is clear and the negation is supported. Do not reward verbosity or require literal gold-string overlap.

For P04 challenge, accept “Melanie said she had never been; Caroline attended the transgender reading,” or a clear correction that Caroline, rather than Melanie, was the attendee. For P11 challenge, accept “Sam said he'd never been to Jasper; Evan took that trip.” Generic refusal is a recorded failure in both cases.

Ground truth is relative to the **whole conversation**, even when graph retrieval omits evidence. A graph response saying “my context doesn't say” on an S item remains an end-to-end failure, while its local evidence limitations are recorded separately below. This prevents a smaller context from redefining answerable questions as unanswerable.

## Evidence coverage, scored separately

After the 24 fresh extractions are frozen, render both contexts and record coverage **before inspecting any new answers**. The preparation has not generated these extractions. Never use the gold evidence to choose query names, add claims, change ordering or improve a particular item's context.

For every S and C item, log:

- All top-level and expanded source turn IDs, raw text, speaker/date metadata actually visible, link labels and order; matched entity IDs/names; candidate/selected counts; cap saturation (`selected == 20`) separately from actual truncation (`candidates > 20`); empty reason.
- Presence of each validated evidence turn in top-level versus linked text; candidate sufficient-set inclusion. These are **mechanical presence checks**, not semantic coverage scores. The listed sufficient sets are sufficient source passages, not claims of unique/minimal evidence; alternative support can be accepted with IDs and an explanation.
- `coverage = sufficient / partial / absent / conflicting_or_ambiguous`, based only on the rendered answer input. A first-person linked row without speaker/date or antecedent can be present yet insufficient. Typed links are model-created assertions, not independent truth.
- Context characters, UTF-8 bytes, tokenizer-counted context tokens, complete chat-prompt tokens, top-level claims, expanded rows and distinct source turns. Record server usage separately; local counts are estimates of provider tokenization.

For U items, coverage of a nonexistent answer is **not applicable**. Record whether the relevant contrasting evidence is present and whether missing attribution or typed links could misleadingly transfer it. Source-level absence cannot be established simply by seeing no matching graph claim.

Interpretation is descriptive: an incorrect S answer with absent/partial evidence is compatible with a retrieval limitation; an incorrect answer with sufficient evidence is an answer-generation failure under this presentation. A correct answer with absent listed evidence needs source review for alternate evidence, inference or guessing. These patterns do not isolate entity gating, claim cap, chronology, typed links or length as a cause.

## Blinding and adjudication

The operator uses a reproducibly shuffled execution schedule and a separate scoring shuffle. Make one opaque answer ID per response. Keep the condition/ID key with the operator, separate from the rater packet. The rater sees only the opaque ID, question, response, validated target/rule and source evidence/full transcript. Hide condition, paired-condition answer, request order, retrieval diagnostics, usage, timing and historical scores. Do not expose paired answers side by side. The preparation contains no rater responses yet.

Have a second human independently review all disputed/ambiguous scores, if available. Resolve disputes while still blinded and preserve both initial ratings, the adjudication and the reason. If only one rater is used, disclose that. Text style may reveal the condition; blinding is a procedure, not a guarantee. A coverage reviewer needs contexts and is not condition-blind; do not let that knowledge enter initial answer scoring.

If a genuine source/label defect is discovered after answers exist, do not replace that pair or silently change the primary denominator. Flag a protocol deviation, retain the locked-set result, and report a separate sensitivity analysis excluding the entire affected pair from both conditions. Do not choose replacements using results.

## Reported quantities and baselines

For each condition report counts and percentages per conversation and pooled:

- S accuracy: correct S / 12 (four per conversation).
- U abstention/correction accuracy: correct U / 10; show per-conversation denominators 3, 4, 3.
- C correction accuracy: correct C / 2; conv-26 and conv-49 have one each; conv-30 has none, so its C rate is N/A.
- Challenge accuracy: correct (U + C) / 12.
- Balanced accuracy: `(S accuracy + challenge accuracy) / 2`. This equals overall 24-item accuracy here because the two sides have equal sizes. It is not a three-class macro average.
- Pair success: both members strictly correct / 12 (four per conversation).
- Unsupported-assertion and unnecessary-abstention counts, overall and by ground-truth class; unresolved/operational failures and coverage-stratified S counts.
- Equal-conversation macro and pooled micro rates for S, challenge, overall and pair success. Equal sample sizes make these equal for this fixed design; do not average the U or C rates without explaining their unequal/missing denominators.
- Paired condition differences and discordant counts: graph-only correct, full-only correct, both, neither; also show the complete 24-row result table and coverage annotations.

With this exact proposed label set, a literal **always-abstain** response gets 0/12 S, 10/12 challenge, 0/2 C, **10/24 overall = 41.67% balanced accuracy**, and 0/12 pair success. It gets 100% on U alone. This is a zero-call analytic baseline, not a model result. If the reviewed labels change, recompute it before running.

Use counts as primary evidence. Twelve deliberately selected, partly overlapping pairs from three conversations do not support general architectural-superiority claims or independence-based significance claims. Do not treat 24 questions as independent random draws. Positive findings remain exploratory. The corrected dates and deterministic ordering make this a new controlled follow-up, **not a direct replication of May's 96% score**. Leave the typed-relation ablation and Hindsight comparison unrun.
