# Agent Memory Evaluation: Abstention, Evidence Loss, and Retrieval Failure

An exploratory case study of how an agent-memory retrieval pipeline failed to deliver answer-bearing evidence, and how an abstention-oriented evaluation initially obscured that failure.

This independent project investigates one local graph-memory prototype using LoCoMo conversations. It follows an initial comparison with full context and a locally adapted Hindsight implementation, an audit of that comparison, a matched-question follow-up, and a trace of the evidence reaching the answering model. Its contribution is a documented implementation failure and correction of the project's own evaluation—not a new benchmark or general theory of memory.

**[Read the research note](DISTINGUISHING_ABSTENTION_FROM_EVIDENCE_LOSS.md)** · **[Inspect the Caroline case study](case_studies/caroline_adoption_agency.md)**

## Main finding

A system can withhold an answer because retrieval missed the evidence, even when that evidence exists in its stored conversation.

Caroline explicitly explained that she chose an adoption agency for its LGBTQ+ inclusion and support. The graph stored her statement under `I`, `they`, `LGBTQ+ folks` and `adoption`, while the query extracted `Caroline` and `adoption agency`; exact matching missed the stored turn. The model then said the reason was unavailable and speculated about alternatives.

The distinction is between information missing from the **retrieved context** and information absent from the **original conversation**. Inserting the omitted evidence was not tested as a repair.

## Evidence at a glance

**November follow-up: provisional judgments from one blinded AI reviewer, with 11 flagged judgments and no independent human adjudication.** The deliberately selected sample contains 12 supported/challenge pairs from three conversations: 12 supported, 10 unsupported and 2 contradicted-premise questions.

| Measure | Full context | Graph |
|---|---:|---:|
| Supported answering | 11/12 | 0/12 |
| Unsupported abstention/correction | 7/10 | 8/10 |
| Contradicted-premise correction | 2/2 | 1/2 |
| All challenge questions | 9/12 | 9/12 |
| Overall / balanced accuracy | 20/24 | 9/24 |
| Both members of a pair correct | 9/12 | 0/12 |

The equal challenge scores do not demonstrate equal underlying capability. Scoring sensitivity covers only the 11 supplied flags; unknown scoring and source-label errors remain possible.

## What the investigation contains

- **Initial evaluation:** three context conditions on 111 LoCoMo Category-5 questions. A local blanket abstention target, answerable items, source-blind judging and unequal settings limited what its scores measured.
- **Methodological correction:** the November comparison matched answer model, settings and question text across full context and graph. It used 24 fresh query extractions and 48 answer calls, with no Hindsight condition.
- **Frozen diagnosis:** all 16 distinct prespecified evidence turns were stored as text and indexed under some entities. Ten failed exact entity matching; five entered candidates but fell below the recency-first top-20 cutoff; one survived without providing the complete answer. These are turn-level counts, not independent question-level failure rates.

None of the 12 supported graph contexts was sufficient under the assistant's assessment made before answers were generated. That assessment also lacks independent human validation. November changed the protocol and does not directly replicate May.

## Reproducibility and artifacts

The [selected evidence package](publication/README.md) includes saved score records, summary statistics, sensitivity results, diagnostic summaries, the original scoring rubric and its AI-only amendment, and an extract for the Caroline trace. The package records source hashes and marks omissions explicitly.

From a checkout containing this package, run:

```sh
python3 publication/check_records.py
```

This offline check validates included hashes and arithmetic. It does not rerun models, independently validate the labels, or reconstruct retrieval. The [artifact guide](publication/README.md) explains dataset acquisition and the limits of the selected records. Complete prompts, caches, ingestion logs and the patched Hindsight experiment remain outside this package; full historical reproduction is not claimed.

The original [graph representation](2026-04-22-graph-memory/graph.py), [extractor](2026-04-22-graph-memory/extractor.py) and [May retrieval code](2026-04-27-locomo/graph_qa.py) remain available for inspection. They are historical code, not a turnkey runner for the corrected November experiment.

## Scope and attribution

This is a small, deliberately selected, partly overlapping sample. No ablation isolated typed relations, matching, ranking or context size. The results do not establish general superiority or inferiority of graph memory, statistical generalizability, or a novel evaluation principle. The note positions the case alongside prior work on context sufficiency and memory diagnostics, and credits earlier LoCoMo issue reports.

LoCoMo is from [Maharana et al., ACL 2024](https://aclanthology.org/2024.acl-long.747/); its inspected repository carries [CC BY-NC 4.0 terms](https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/LICENSE.txt). Hindsight is upstream software from [Vectorize and contributors](https://github.com/vectorize-io/hindsight), under its [MIT license](https://github.com/vectorize-io/hindsight/blob/89c34c31357ecd80e2ab2ed3244d9e37515cb5f8/LICENSE). Local adaptations and the graph prototype are distinguished from upstream code in the artifact guide. This repository does not yet assign a general reuse license to its own code and writing.
