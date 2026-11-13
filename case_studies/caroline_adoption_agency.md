# Caroline’s adoption-agency choice: from stored evidence to a missing answer

This is a trace of one supported question in the frozen November experiment, not a new test or repaired retrieval run. It distinguishes an answer missing from the model's retrieved context from an answer absent from the original conversation.

[Research note](../DISTINGUISHING_ABSTENTION_FROM_EVIDENCE_LOSS.md) · [Machine-readable trace extract](../publication/caroline_trace.json) · [Artifact scope and provenance](../publication/README.md)

## Question and source

- **Question:** “Why did Caroline choose the adoption agency?”
- **Dataset identifier:** `conv-26:qa[87]`, a zero-based index in the complete LoCoMo QA array; matched pair `P02`, supported member.
- **Validated target:** the agency's inclusivity and support for LGBTQ+ people. Source review was assistant-led and approved for execution; it was not independent human adjudication of every label.
- **Speaker and source:** Caroline, dialogue `D2:12`, session 2, 25 May 2023.

> I chose them 'cause they help LGBTQ+ folks with adoption. Their inclusivity and support really spoke to me.

This question and source passage are verbatim excerpts from LoCoMo by Maharana et al.; see the [paper](https://aclanthology.org/2024.acl-long.747/) and [license at the inspected revision](https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/LICENSE.txt) (CC BY-NC 4.0). The source text is unchanged; the diagnostic annotations describe this project's local implementation. No complete conversation or dataset file is bundled here.

## Stored claim and query

The graph preserved the passage verbatim as claim `e4f7c468-74a4-4b71-b5ce-530537061676`, with Caroline stored separately as speaker metadata. The historical extractor saw the raw turn text without that speaker context.

| Field | Recorded value |
|---|---|
| Indexed entity names for this turn | `I`, `they`, `LGBTQ+ folks`, `adoption` |
| Stored entity predicates | `I — chose → they`; `they — help → LGBTQ+ folks`; `they — with → adoption` |
| Fresh query entities | `Caroline`, `adoption agency` |
| Fresh query predicate | `Caroline — choose → adoption agency`; ignored for retrieval gating |
| Graph entities matched by the query | `Caroline`, `adoption agency` |

Both query names matched entities elsewhere in the graph. Neither matched an entity indexing this source turn. Stored predicates did not resolve “I” to Caroline or “they” to the agency, and did not independently test proposition support.

## Retrieval and final context

The gate united claims indexed under **any** exactly matching entity name after lowercasing. Here `{caroline, adoption agency}` and `{i, they, lgbtq+ folks, adoption}` have no common name.

| Stage | Frozen observation |
|---|---|
| Ingestion and indexing | Source text equals cached raw text; all four listed entity indexes contain the claim. |
| Candidate matching | The required turn was absent from the 130-claim candidate pool. |
| Ranking and top-20 cap | Twenty other claims were selected. This turn was lost before ranking, rather than displaced at the cutoff. |
| Linked expansion | No selected parent had a stored link to this turn; the expansion did not recover it. |
| Final rendering | Zero occurrences of `D2:12`; pre-answer semantic coverage was `absent`. The context was nonempty, so the explicit empty-context branch did not apply. |

The November 9 diagnostic reconstructed the graph context and checked it against the actual sent messages. Context SHA-256: `78082167f6d0c865e00102facebbbd96dae19c66392d5a9b30befecb5c77c416`. The extract includes the 130 ranked candidate identifiers and every rendered source identifier, allowing inspection of the recorded omission without reproducing the full transcript.

Historical records are identified as `calls/extraction/extraction-03.validated.json` and `calls/answer/answer-14.request.json` within the November run. The extract and [provenance record](../publication/provenance.json) retain their identifiers and hashes. Full cache, raw sent context and original logs are not part of this selected package; the prompt-match check is a recorded result of the frozen diagnostic, not an independently reproducible check from this extract alone.

## Exact answer-model response

Answer `R011`, `Qwen/Qwen2.5-72B-Instruct`, graph condition:

> The provided memories do not explicitly state why Caroline chose a specific adoption agency. However, based on the conversation, it can be inferred that Caroline is knowledgeable about the adoption process and is supportive of the idea. She advises Melanie to do research, find an adoption agency or lawyer, and gather necessary documents and emotional preparation. This suggests that Caroline likely chose an adoption agency based on similar criteria—reliability, support, and a thorough process to ensure the well-being of the children and the adoptive parents.

The preserved blinded AI review assigned score **0** and flagged both unnecessary abstention and unsupported assertion. This individual score was not one of the review's 11 uncertainty flags. It remains an AI judgment, without independent human adjudication.

## What this example establishes

**Observed:** the source supplied the reason; its text was stored and indexed; exact entity matching excluded it; linked expansion did not recover it; the final context lacked the designated answer; the model expressed uncertainty and then speculated.

**Interpretation:** the model's statement about its provided memories is consistent with a retrieval omission. It is not evidence that the original conversation lacked the answer. The stored speaker field alone did not make the turn retrievable by the speaker's name.

**Untested:** inserting the source turn, adding speaker-aware indexing, changing ranking, or removing typed links might change the response, but none was tested here. This example does not establish that any proposed repair would succeed or isolate a general causal effect of graph memory.
