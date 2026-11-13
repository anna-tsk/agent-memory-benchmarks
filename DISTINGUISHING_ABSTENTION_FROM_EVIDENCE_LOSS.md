# Distinguishing Abstention from Evidence Loss in Agent Memory Evaluation

*Independent exploratory case study · Version 3, public presentation · 12 November 2026*

## Abstract

Can an agent-memory system withhold unsupported answers while preserving supported answering? An initial comparison on 111 LoCoMo Category-5 questions appeared to favor a graph configuration, with a 96.1% equal-conversation judge score. An artifact audit reproduced the scores but identified a blanket abstention-gold override, at least five answerable questions, inconsistent judgments, and unequal settings. An exploratory follow-up used twelve source-reviewed supported/challenge pairs and matched answer-generation settings; it was not a direct replication of May. Under one blinded AI review, with 11 flagged judgments and no independent human adjudication, full context answered 11/12 supported questions correctly and graph answered 0/12; both scored 9/12 on challenge questions. A frozen retrieval trace found all sixteen distinct prespecified evidence turns stored and indexed, but ten failed entity matching and five were removed by recency-based selection. The remaining turn did not supply a complete answer. The later matched-pairs experiment exposed evidence loss that the original abstention-oriented evaluation had not measured. This case applies established distinctions between source answerability, retrieved evidence sufficiency, and response correctness to a specific local failure trace. It does not introduce a new evaluation principle or establish a general property of graph memory.

## 1. Research question

A refusal can reflect two different situations: the conversation never established the requested fact, or the memory system failed to retrieve a fact that the conversation established. Only the first supports a source-level claim that the answer is unavailable. An evaluation that rewards withholding without measuring supported answering cannot distinguish these situations.

The investigation progressed from a May comparison of three context-construction conditions, through an audit, to a November matched-pairs experiment and frozen retrieval trace. It evaluates complete configurations, not graph structure in isolation.

## Related work and contribution

The distinction between retrieval insufficiency and failure to use available evidence precedes this case study. Joren et al., *Sufficient Context* (ICLR 2025), analyze answer errors according to whether retrieved context can support an answer [11]. Wu et al., *LongMemEval* (ICLR 2025), evaluate abstention alongside other memory abilities and separate indexing, retrieval and reading [12]. These provide relevant evaluation concepts; this project does not propose a new benchmark or discover those distinctions.

Recent diagnostic work makes the overlap more specific. *EvalMem* separates checks of stored facts, retrieved evidence and generation from oracle evidence [13]. Roshchupkin and Khodorchenko's *Where Did the Evidence Go?* (6 November 2026 preprint) traces annotated support sets through retrieval and final-context selection, distinguishing evidence recall from answer correctness [14]. Our frozen trace concerns one local implementation and uses prespecified sufficient-evidence sets; it does not include an oracle-evidence answering intervention. These works are cited to position the completed investigation, not to retroactively describe its design process.

The LoCoMo benchmark is credited to Maharana et al. [9]. Its issue tracker also predates this audit's annotation findings: `dial481`'s 4 April 2026 comment in issue #34 identifies the explicit “No” bowl and Oscar answers and questions blanket Category-5 scoring; issue #43, filed by `kepler-somnium` on 24 August 2026 and co-signed by Lei Shi, reports Caroline's answerable bowl-reminder and instrument questions [15,16]. These are prior community reports, not independent adjudication of our outcomes. We claim neither first discovery of those defects nor originality for separating retrieval and generation failures. Hindsight is cited at the inspected upstream revision; the local experimental patch is separately identified [2,10].

The incremental contribution is the documented chain from a misleading local evaluation to a corrected question design and a trace of exactly where stored evidence failed to reach the answerer. The evidence supports this implementation-specific account, including the limits of its provisional scoring.

## 2. Initial experiment: an apparent abstention advantage

### 2.1 Design and reconstructed results

The May comparison used all 111 Category-5 questions from conversations `conv-26`, `conv-30` and `conv-49`: 47, 24 and 40 questions respectively. These are the three shortest conversations in the inspected LoCoMo release [9]. Conditions supplied the answer model with the complete text conversation, memory retrieved through a locally adapted Hindsight implementation [10], or claims retrieved from an experimental graph. All retained answer runs identify `Qwen/Qwen2.5-72B-Instruct`. The following results concern questions without a multiple-choice wrapper. [1]

The graph stored each conversation turn as one claim, with its raw text and speaker metadata. An extractor identified entities and free-text entity-to-entity predicates; a separate classification step added typed links between claims sharing indexed entities (`COEXISTS`, `CONFLICTS`, `SAME_AS`, or `NEEDS_CLARIFICATION`). Retrieval matched extracted query entity names exactly after lowercasing, selected claims by descending stored timestamp, and expanded linked claims. The answer prompt displayed turn text and claim-link labels, not the extracted entity predicates. Neither these stored relations nor retrieval verified whether a queried proposition was true or supported. [3]

**Table 1. May scores reconstructed from saved judgments. “Accepted” means agreement with the local evaluation target, not independently verified abstention or factual correctness.**

| Condition | conv-26 | conv-30 | conv-49 | Equal-conversation mean | Pooled accepted / questions |
|---|---:|---:|---:|---:|---:|
| Full context | 42/47 | 18/24 | 33/40 | 82.29% | 93/111 (83.78%) |
| Locally adapted Hindsight | 32/47 | 20/24 | 21/40 | 67.97% | 73/111 (65.77%) |
| Graph | 45/47 | 24/24 | 37/40 | 96.08% | 106/111 (95.50%) |

The headline 82.3/68.0/96.1 figures are macro averages: each conversation receives equal weight. Pooled scores weight each question equally. Graph leads in each conversation, but the full ordering is not constant: Hindsight exceeds full context on `conv-30`. [1]

### 2.2 Why these scores do not establish discrimination

The local harness assigned **“Not mentioned in the conversation.” to every Category-5 question**, overriding any explicit answer. A local patch enabled this behavior in the Hindsight benchmark runner; the inspected unpatched Hindsight revision excluded Category 5. The override is therefore a property of this experiment, not a finding about published Hindsight evaluation. Dataset annotation problems and the experimental decision to impose a universal abstention target are separate issues. [2]

The audit identified at least five source-supported questions: Caroline’s bowl reminder (`conv-26:qa[161]`), whether she made a black-and-white bowl (`qa[167]`), whether Oscar was Melanie’s pet (`qa[178]`), Caroline’s instrument (`qa[184]`), and Gina’s dance-contest award (`conv-30:qa[95]`). The black-and-white bowl question (“Did Caroline make the black and white bowl in the photo?”, `conv-26:qa[167]`) and Oscar pet question (“Is Oscar Melanie's pet?”, `conv-26:qa[178]`) already had explicit “No” answers in the source data. The trophy question also appeared in Category 4 with the answer “a trophy.” All five graph responses received credit for withholding; several supported full-context answers were rejected. These five cases are a demonstrated minimum, not validation of the remaining questions. [2]

The answerer and judge used the same named Qwen model. Judges received the question, gold and response without source evidence. Their permissive rubric accepted uncertainty about details without explicitly prohibiting subsequent speculation. The audit found at least three cross-condition grading inconsistencies; saved outputs cannot establish an exhaustive error rate. Under the intended synthetic-gold rule, an always-abstain answerer could receive **111/111**. That analytic reference was not run. [2]

The comparison also bundled material differences. Full context and graph requested plain text at temperature 0 with a 100-token output limit; Hindsight requested JSON answer/reasoning fields and left answer temperature and output limit unspecified. Judge serialization and output limits differed. Hindsight ingested session JSON including image metadata, while the other conditions used text. Graph timestamps reflected ingestion time rather than conversation dates, and equal timestamps produced unstable selection at the cutoff. Exact served model revisions were not recorded. Separate MCQ runs also changed retrieval queries and are excluded from this account’s comparisons. [1–3]

The original architectural explanation was unsupported: the explicit empty-context marker appeared on only **1/111** questions, already accepted in all three conditions. Retrieval tested entity-name matches, not whether a queried proposition was supported. Excluding the five clearly answerable items left graph at **101/106 (95.28%)**, full context at **91/106 (85.85%)**, and Hindsight at **71/106 (66.98%)** under the remaining original judgments. This sensitivity calculation preserves the descriptive ordering but does not repair the evaluation construct. [2,3]

## 3. Methodological correction: matched answerability

### 3.1 Questions and scoring targets

The November preparation selected four pairs per conversation. Each pair uses an original Category-4 question and a closely related Category-5 question differing in person attribution and associated grammar. Whole-conversation source review by the assistant, followed by user approval before execution, produced **12 supported (S), 10 unsupported (U), and 2 contradicted-premise (C)** targets. Approval authorized execution; it did not constitute independent human adjudication of all source labels. Five candidate pairs were replaced for ambiguity, date/event mismatch or dependence on excluded image information; the reasons and exact original question IDs were recorded before new answers existed. This was deliberate selection informed by the preceding investigation, not random sampling or a confirmatory preregistration. [4]

The frozen rubric requires a supported answer on S, justified withholding or a qualified correction on U, and a supported correction on C. In the Jasper challenge, Sam says he has never been there; identifying Evan as the traveler can correct the premise. Generic refusal fails C, and a disclaimer followed by an unsupported target guess fails U. An S answer remains required even if retrieval omits its evidence: smaller context does not redefine source-level answerability. [4]

### 3.2 Conditions and execution

The comparison included full context and graph only. Both used the original unwrapped question, the same answer prompt, and `Qwen/Qwen2.5-72B-Instruct` explicitly routed to DeepInfra through Hugging Face. Answer settings were temperature 0, `top_p=1`, maximum 256 output tokens, zero presence/frequency penalties, and plain-text output. The common prompt asks for direct evidence but also permits logical inference; it does not enforce abstention. [5]

The graph reused the existing cache. It matched fresh query entities to stored names exactly after lowercasing, united their claim lists, sorted by source datetime and numeric session/turn, retained twenty claims, and expanded up to five eligible stored links per claim without recursion. Query predicates did not constrain matching. Top-level rows included speaker and date; expanded rows did not. Linked targets could repeat under different parents. Source-date repair and deterministic ordering made this a **new controlled follow-up, not a direct replication of May**. [5]

Both members of every pair received fresh query extraction. The 24 extractions were frozen before context construction. The assistant then assessed rendered evidence sufficiency—including attribution and alternative support—before any answer call. Contexts were not repaired using those assessments. Forty-eight answer requests followed in a frozen shuffled schedule; all 72 extraction/answer calls completed normally without retries. Requests, responses, usage, query outputs, contexts and coverage records were retained. [5]

### 3.3 Evaluation and estimands

Human scoring was planned but replaced, with user authorization, by one condition-blinded AI review attributed to **ChatGPT GPT-6**. The reviewer reportedly saw all 48 anonymized responses, targets, rubric and source passages without the condition mapping. The external review process and exact model revision were not independently verified. Ratings were checked against the frozen packet, preserved and hashed before unblinding. There is **no independent human adjudication or inter-rater agreement**. [6]

Eleven responses were flagged for possible adjudication. The scoring amendment retained their submitted scores, including three passes carrying unsupported-assertion flags. These materiality judgments remain unresolved. The flags concern scoring interpretation, not source-label changes. [6]

Reported quantities are S, U and C accuracy; combined challenge accuracy on U+C; and pair success, requiring both members correct. Balanced accuracy is

`½ × (S correct / 12 + challenge correct / 12)`.

It equals overall 24-question accuracy here, not a three-class macro average. The analytic always-abstain reference passes all ten U items and fails all S and C items: **10/24 = 41.67% balanced accuracy and 0/12 complete pairs**. [4]

## 4. Findings from the November comparison

### 4.1 Provisional AI-judged outcomes

**Table 2. Preserved judgments from one blinded AI reviewer, with 11 flagged judgments and no independent human adjudication.**

| Measure | Full context | Graph | Analytic always-abstain |
|---|---:|---:|---:|
| Supported answering | 11/12 (91.67%) | 0/12 (0%) | 0/12 (0%) |
| Unsupported abstention/correction | 7/10 (70%) | 8/10 (80%) | 10/10 (100%) |
| Contradicted-premise correction | 2/2 (100%) | 1/2 (50%) | 0/2 (0%) |
| Challenge questions, U+C | 9/12 (75%) | 9/12 (75%) | 10/12 (83.33%) |
| Overall = balanced accuracy | 20/24 (83.33%) | 9/24 (37.50%) | 10/24 (41.67%) |
| Both members correct | 9/12 (75%) | 0/12 (0%) | 0/12 (0%) |

Full context had higher overall scores in all three conversations: **7/8 versus 3/8**, **5/8 versus 4/8**, and **8/8 versus 2/8** for `conv-26`, `conv-30` and `conv-49`. Equal-conversation and pooled averages coincide for S, challenge, overall and pair success because their per-conversation denominators are equal. U and C denominators differ; Table 2 reports their pooled counts. Paired question outcomes were full-only correct **13/24**, graph-only **2/24**, both **7/24**, and neither **2/24**. [6]

Graph responses carried abstention flags on **23/24**, including all twelve S questions, but also unsupported-assertion flags on **10/24**; full context had these flags on **5/24** and **5/24**, respectively. Flags overlap and are the supplied reviewer’s judgments. The graph therefore exhibited neither literal always-abstention nor uniformly cautious answers. Its one-question U advantage was offset by one fewer C correction, leaving no aggregate challenge advantage in these saved judgments. Equal challenge scores do not establish equal underlying capability. [6]

### 4.2 Sensitivity to flagged ratings

Six flagged ratings belong to full context and five to graph. Flipping each score individually does not reverse the overall ordering. Even the extreme assigning every flagged decision in graph’s favor yields **full context 17/24 (70.83%) versus graph 11/24 (45.83%)**, and **7/12 versus 0/12** complete pairs. Graph’s twelve S failures are unflagged in this review. These are mechanical bounds over the eleven supplied flags, not adjudicated alternatives or confidence intervals; unflagged ratings and source labels may also be wrong. This sensitivity analysis cannot establish robustness to those unknown errors. The smaller U comparison, C comparison and graph’s one-answer deficit to always-abstain are sensitive to the flags. [7]

## 5. Retrieval diagnosis: stored evidence was not delivered

The pre-answer coverage assessment classified ten graph S contexts as absent and two as partial, with **0/12 sufficient**; full context retained support for all twelve. These are assistant assessments, not independent human annotations. On November 9, a diagnosis traced the prespecified source evidence through ingestion, indexing, candidates, ranking, expansion and rendering. No retrieval rule or query was changed. Reconstructed graph contexts matched all 24 saved contexts and their actual sent prompts. [8]

**Worked example: Caroline’s adoption-agency choice.** P02 asks, “Why did Caroline choose the adoption agency?” (`conv-26:qa[87]`). Caroline’s source turn `D2:12` answers: “I chose them 'cause they help LGBTQ+ folks with adoption. Their inclusivity and support really spoke to me.” The graph stored this text but indexed it under `I`, `they`, `LGBTQ+ folks` and `adoption`, without resolving the pronouns to Caroline or the agency. The query extracted `Caroline` and `adoption agency`; neither matched this turn’s indexed names. The turn therefore never entered the candidate pool, and no link from a selected claim recovered it. [8]

The resulting graph answer (`R011`) began, “The provided memories do not explicitly state why Caroline chose a specific adoption agency.” It then speculated about reliability, support and a thorough process, omitting the stated LGBTQ+ inclusion rationale. This was not a clean refusal: the preserved review flagged both unnecessary abstention and unsupported assertion. The information was unavailable in the retrieved context, although the original conversation explicitly supplied it. Inserting the missing turn was not tested, so a repaired answer is not established. The [standalone case study](case_studies/caroline_adoption_agency.md) supplies the exact response and identifiers. [6,8]

All **1,297/1,297** cached turns matched source text and speaker. The twelve S questions’ prespecified sufficient sets contain **sixteen distinct turns**, appearing seventeen times because two questions share one turn. Every required turn was preserved as raw text, indexed under some extracted entities, and matched to historical extraction logs. This does not imply indexing under the correct speaker or a semantically useful name. These sets include antecedent/event context as well as answer values; they are not asserted to be uniquely necessary evidence. Supplementary relevant passages were also traced. [8]

**Table 3. Direct retrieval fate of the sixteen distinct prespecified evidence turns. These are turn-level diagnostic counts, not independent question-level failure rates.**

| Stage | Turns | Observation |
|---|---:|---|
| Failed query-entity matching | 10/16 | Stored and indexed, but no extracted query name matched an indexed entity of the turn |
| Entered candidates, then removed | 5/16 | Ranked 70, 74, 127, 128 or 131, beyond the top-20 cutoff |
| Retained directly | 1/16 | Poetry attendance at rank 15; the reading’s subject remained absent |

**Speaker and phrase matching.** Ingestion extracted entities from raw text without its speaker or surrounding dialogue. None of the sixteen required turns was indexed under its own speaker’s name. Jon’s flooring evidence (`conv-30:D2:8`) used `Marley flooring` and `dance studios`, which did not match `flooring` or `dance studio`. Unresolved pronouns and exact phrase boundaries excluded these turns before ranking. All twenty top-level turns in each of the four Evan-positive contexts were spoken by Sam: matching a name can select speech addressing that person. [8]

**Selection and expansion.** Relevant turns that did match were still displaced by newer candidates. Gina’s furnishing rationale ranked **74/82**, and Evan’s new-Prius statement ranked **131/132**. Of the fifteen omitted required turns, seven had links from selected claims, but every route exceeded the five-link cap; the earliest eligible positions ranged from **52 to 114**. The other eight had no stored link from a selected parent. These link-stage counts overlap the direct-stage losses and must not be added to Table 3. [8]

**Rendering and context reuse.** No complete required S evidence set arrived and then became insufficient solely through removal of attribution. Required answer content was missing upstream, although unlabeled expanded rows introduced additional attribution risks. Some positive evidence appeared in the corresponding negative prompt instead: Evan’s new-Prius statement was repeated **17 times** in the Sam-negative context, and his alternative painting-history statement appeared there once. Across 24 graph questions there were **13 distinct context texts**. No matched positive/negative pair shared graph context, but all four Evan-positive questions shared one. Every retrieval was nonempty and selected twenty top-level claims. [8]

Graph supplied fewer input tokens overall—**115,734 versus 346,344** across 24 answer calls per condition, a **66.58% reduction**—yet included 95–100 additional, sometimes repeated, linked rows per question. Evidence volume alone is therefore an incomplete description of the change: identity, event binding, ordering and duplication also differed. [6,8]

## 6. Discussion

In this configuration, frequent withholding coincided with failure to retrieve sufficient evidence and poor supported answering. The saved judgments showed no gain in combined challenge accuracy; this is an observed association, not a demonstrated causal tradeoff. A response can accurately report missing information in its immediate context while failing the task defined over the stored conversation. An abstention-oriented target can reward that behavior.

The trace establishes where particular evidence was lost; the coverage review assesses remaining support; and the provisional AI ratings assess responses against whole-conversation targets. Together they document failure to preserve supported answering. They do not isolate a causal benefit or harm from typed relations, entity gating, recency, cap size or context length.

There is no S graph-context stratum with sufficient evidence from which to estimate accuracy conditional on successful retrieval. Full context’s one S failure occurred with evidence available. Graph’s failed Jasper correction had inferential attribution cues judged sufficient at moderate confidence, and its response score was flagged. Retrieval limitation and answer-generation error therefore remain distinct. [6]

This does not attribute May’s entire numerical difference to evidence loss or establish that explicit abstention mechanisms are unnecessary. It shows why the original score could not establish discrimination and why refusal alone would have hidden the follow-up’s answerability failure.

## 7. Limitations and future work

The sample consists of twelve deliberately selected pairs from three short conversations. P01/P02 and P07/P08 share source evidence; other items share conversations and retrieved contexts. Questions are not independent random samples. No significance tests, independence-based intervals, or population-level architectural claims are warranted.

One user-reported blinded AI reviewer supplied the outcomes, eleven were flagged, and none received independent human adjudication. Source validation and coverage review also depended on assistant judgments. Blinding concealed condition labels but could not guarantee that response style concealed the condition. The preserved scoring amendment leaves passing/error-flag tensions unresolved.

November changed the question set, targets, output limit, graph chronology and ordering, and evaluation procedure. Its percentages cannot be interpreted as a direct decline from May’s 96.1%. Although November matched answer settings, it compared bundled context-construction configurations, not individual retrieval components. No repeated generations measured variability; exact remote weight revisions and serving fingerprints were unavailable. Some caches, logs and sibling-repository files are local artifacts, so a clean public checkout alone is insufficient for reproduction.

Future experiments could separately test **speaker-aware indexing**, **relevance-based retrieval ranking under a fixed evidence budget**, and **evidence-sufficiency assessment against matched S/U/C targets**. A speaker-aware index might still lose older facts at the cap, and a larger cap might still miss unmatched entities. These are hypotheses, not demonstrated repairs. Each intervention should preserve the other conditions and log evidence coverage before evaluating answers. Independent scoring and wider source validation are also needed. No such ablation or additional model experiment is reported here.

## References and provenance

References 1–8 identify the experimental record; 9–16 identify primary sources and prior reports. Question IDs use zero-based indices in each conversation's complete QA array. The [selected evidence package](publication/README.md) supplies the public-facing records and identifies original files by hash. Excluded originals are named below as provenance identifiers, not links to files that a public checkout would lack.

1. **May reconstruction.** [Saved no-MCQ counts and averages](publication/may_results.json), extracted from the original audit's `computed_summary.json`. Full/graph verdicts came from saved judged reports, not rough-match flags; Hindsight verdicts came from saved result JSON. The original `AUDIT_REPORT.md`, 333-row `audit_question_level.csv` and complete source manifest remain local; audit and summary hashes are in [provenance](publication/provenance.json).
2. **Evaluation validity.** Local `EVAL_AUDIT.md` records the judge rubric, five answerable cases and grading inconsistencies; its hash is preserved in the provenance record. The local patched Hindsight runner SHA-256 is `380953c90d8bf5abaf8aa8380284291e5fd9bae088108467b72538a55dd66914`. Its [unpatched upstream base](https://github.com/vectorize-io/hindsight/blob/89c34c31357ecd80e2ab2ed3244d9e37515cb5f8/hindsight-dev/benchmarks/common/benchmark_runner.py) is commit `89c34c31357ecd80e2ab2ed3244d9e37515cb5f8`; this link does not supply the local patch or its historical Hindsight results.
3. **Original graph mechanism.** [Graph representation](2026-04-22-graph-memory/graph.py), [extractor](2026-04-22-graph-memory/extractor.py), and [May retrieval/rendering path](2026-04-27-locomo/graph_qa.py). Local `MECHANISM_AUDIT.md` supplies the historical candidate, timestamp and MCQ-query analyses, with its hash in the provenance record. The dated code is not the repaired November runner.
4. **November preparation.** [Original frozen scoring rubric](publication/SCORING_RUBRIC.md); SHA-256 `bf47f0de65ce41c6714c1105c432e90a86f6fad8676e39360b2414f65a6015b2`. The original pair manifest, validation sheet and replacement decisions remain local; their relevant hashes are recorded. The rubric's planned human-scoring workflow was superseded by reference 6.
5. **Execution.** [Selected frozen settings and exact prompts](publication/execution_settings.json); source/code freeze and run-manifest hashes are in the provenance record. Execution checkout HEAD was `9aeeb1986539314e829eb42ed6b8dcd2fa3de588`; uncommitted runner/preparation files are identified by freeze hashes, not HEAD alone. LoCoMo data SHA-256: `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`. Local tokenizer revision: `495f39366efef23836d0cfae4fbe635880d2be31`, which does not pin remote weights. Raw request/response logs are not bundled.
6. **Provisional outcomes.** [Saved results](publication/november_results.json), [48 selected score/flag records](publication/november_scores.json), and [original AI-scoring amendment](publication/SCORING_AMENDMENT.md). Original ratings SHA-256: `5437099d754ed29264a8d2eda8d304d478bfe182bfcd8d06f4c605a97513e161`. The score extract omits source passages and most model responses; it is not a replacement for independent adjudication or the complete frozen packet.
7. **Scoring uncertainty.** [Preserved sensitivity calculations](publication/scoring_sensitivity.json), including individual flips and extreme scenarios. The original analysis-manifest hash is preserved. No scores were changed for publication.
8. **Frozen diagnosis, 9 November.** [Diagnostic summary](publication/retrieval_summary.json) and [Caroline trace extract](publication/caroline_trace.json). Full traces, contexts, coverage freeze, caches and historical extraction log remain local; their recorded verification is not independently repeatable from these extracts alone. The original May PDF was already absent from its recorded workspace path when the follow-up ran; the May account relies on the preserved local audit and underlying run artifacts.
9. **LoCoMo.** Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. 2024. [Evaluating Very Long-Term Conversational Memory of LLM Agents](https://aclanthology.org/2024.acl-long.747/). *Proceedings of ACL 2024, Volume 1: Long Papers*, pp. 13851–13870. DOI: `10.18653/v1/2024.acl-long.747`. [Official repository](https://github.com/snap-research/locomo), inspected commit `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`.
10. **Hindsight.** Vectorize and Hindsight contributors. [Hindsight implementation documentation](https://github.com/vectorize-io/hindsight/blob/89c34c31357ecd80e2ab2ed3244d9e37515cb5f8/README.md) [software], inspected revision `89c34c31357ecd80e2ab2ed3244d9e37515cb5f8` (29 April 2026). The local experimental adaptation is not the published upstream evaluation.
11. Hailey Joren, Jianyi Zhang, Chun-Sung Ferng, Da-Cheng Juan, Ankur Taly, and Cyrus Rashtchian. 2025. [Sufficient Context: A New Lens on Retrieval Augmented Generation Systems](https://proceedings.iclr.cc/paper_files/paper/2025/hash/33dffa2e3d2ab74a783d1a8c292f66d9-Abstract-Conference.html). *ICLR 2025*.
12. Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. 2025. [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813). *ICLR 2025*; preprint first posted 14 October 2024, revised 4 March 2025.
13. Zeyu Liu, Jian Zhong, Rongduo Han, Ziyang Wu, Shunye Tang, Chenghao He, Yaxuan Yang, Yihang Qiu, Ailing Wang, Xiao Liang, Guohuan Xie, Xiaokang Xue, Gongchen Li, Haining Zhang, and Wei Wang. 2026. [EvalMem: An Operation-Level Diagnostic Framework for Long-Term Memory Systems](https://arxiv.org/abs/2609.22231). arXiv:2609.22231, first posted 3 September 2026; inspected v2, 2 November 2026. The arXiv record reports acceptance to Findings of EMNLP 2026.
14. Bogdan Roshchupkin and Maria Khodorchenko. 2026. [Where Did the Evidence Go? Diagnosing Proof-Set Loss in Long-Term Conversational Memory](https://arxiv.org/abs/2611.10170). arXiv:2611.10170v1, 6 November 2026; preprint.
15. `dial481`. 4 April 2026. Comment on [LoCoMo issue #34](https://github.com/snap-research/locomo/issues/34), “EXIA GHOST Full Pipeline: 95.27% overall (cats 1-5), 100% Category 5.” Cited for its evaluation critique and two explicit “No” examples, not endorsement of the issue's original performance claims.
16. `kepler-somnium`, co-signed by Lei Shi. 24 August 2026. [LoCoMo issue #43](https://github.com/snap-research/locomo/issues/43), “conv-26: two mislabeled category-5 (adversarial) questions — answerable in the transcript, complementing the cat 1-4 audits in #27/#35.” The filing identifies itself as AI-agent-authored with a human collaborator.

**Availability and scope.** The selected package supports inspection of saved counts, scoring flags and the documented case. It does not make the full historical experiment reproducible: omitted source data, graph caches, original logs and the patched Hindsight environment remain necessary. The [artifact guide](publication/README.md) separates runnable offline record checks from unprovided historical replay. No original dataset file or upstream Hindsight code is redistributed in this package. Brief LoCoMo quotations are attributed under the inspected CC BY-NC 4.0 terms.

This public presentation preserves the earlier drafts and frozen experimental artifacts. It introduces no model call, rescoring, ablation or change to the graph implementation.
