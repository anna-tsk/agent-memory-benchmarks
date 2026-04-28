# Sequential Claim Interpretation in Frozen Instruct Models: Two Failure Modes

*Anna Tskhovrebov · April 2026*

## 1. Motivation

Agent memory systems are designed to store and recall facts accurately, but often collapse ambiguous or conflicting information into confident, singular answers. Responses in the shape of "I'm not sure" or "multiple things may be true" are already under-rewarded in LLMs, and sometimes become impossible due to the destructive nature of external memory stores or editing mechanisms. This has become a pressing alignment issue, since a system that cannot properly hold conflicting information is more likely to silently resolve ambiguity in ways humans cannot inspect.

Before designing alternatives, we first ask: what do raw LLMs do when encountering sequential claims in context, with no external memory layer? In particular, can they reliably determine how claims relate to one another — or do they impose unstable assumptions about the structure of those relationships? If the model's ability to properly assess claim relations is limited, any memory system built on top may inherit these limitations.

This note reports an empirical probe of that question. I tested two frozen instruct-tuned LLMs on 14 minimal scenarios involving sequential claims from a single speaker. The results reveal two distinct failure modes: a surface-level recency bias, and a more subtle but more concerning ontological collapse — where the model implicitly decides what kind of relationship it is dealing with (e.g. singular, plural, or time-varying) rather than recognizing that this may itself be ambiguous. These decisions are often inconsistent across prompt conditions for the same model and the same observations.

## 2. Methodology

### Probe design

Each probe presents two timestamped first-person statements from the same speaker, followed by a question. The 14 probes cover a range of predicates: kinship roles (sister, mother), professional relationships (boss, doctor), social relationships (best friend), preferences (favorite person, favorite city), self-descriptions (home, hometown, diet, first language, name preference), and life-state attributes (age, relationship status). No probe includes expected answers. The probes are deliberately underspecified — whether the second claim replaces, coexists with, or contradicts the first is not noted. A representative probe:

```text
All observations are statements from the same speaker.

Observations:
- 2026-01-01T09:00:00Z | Maya is my sister.
- 2026-02-15T12:00:00Z | Leila is my sister.

Question: Who is the speaker's sister?
```

### Models and controls

Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct, run locally through Hugging Face `transformers` with greedy decoding. The system prompt was fixed at *"You answer questions."* to avoid biasing the model toward any resolution policy. Exact prompt messages were logged and inspected for each run. The same-speaker rendering was added after a pilot in which the model incorrectly parsed names inside observations as different speakers ('Leila is Maya's sister').

### Prompt conditions

Each probe was run under three framings: "minimal" (question only), "analytic" (structured output: direct answer, interpretations, assumptions, missing information, memory handling), and "conversational" (same structure, but asking the model to produce a "clarifying question" instead of identifying "missing information"). The prompt format was deliberately varied because pilot runs showed that it changed the model's ontological assumptions, not just response length.

## 3. Failure Mode 1: Recency Collapse

Both models default to treating the more recent claim as the current state. This is appropriate for some probes — such as age — but the same heuristic is applied indiscriminately. On the doctor probe, the 7B model under all three conditions produces some variant of "Dr. Alvarez is the speaker's current doctor," despite the fact that having more than one doctor is unremarkable. On the mother probe, both models overwrite Sara with Lina, despite the possibility of two mothers. On the best friend probe, the 7B model in the analytic condition resolves to "Tina is the current best friend" even after explicitly enumerating, in its own "possible interpretations" response, that both could be best friends simultaneously.

This produces a characteristic contradictory shape within a single structured answer: a careful enumeration of interpretations followed by a "direct answer" that ignores them. Recency wins by default, even when the model has just demonstrated that it can see alternatives.

## 4. Failure Mode 2: Ontological Collapse

The more important failure is deeper than recency bias. The model is not only choosing which claim wins — it is also implicitly deciding what kind of relationship it is dealing with: a single-possibility relation, a plural-compatible relation, or a temporally-indexed singular slot. These decisions are unstable across models and across prompt conditions for a fixed model.

The sister probe is the clearest illustration. The 3B model under the minimal condition reports that "Maya is my sister" and "Leila is my sister" contradict each other — a category error resulting from an implicit singular assumption. The 7B model fixes this under minimal (correctly answering that both are sisters), but the same 7B model under the analytic condition answers "Leila is the speaker's sister" and notes that "the speaker refers to 'sister' in the singular form, implying there is only one sister mentioned." The structured prompt reintroduced singularity where the minimal prompt had not.

### Selected results across model size and prompt condition

| Probe | 3B Minimal | 7B Minimal | 7B Analytic |
|---|---|---|---|
| Sister | Contradiction | Both are sisters | Leila only (recency) |
| Best Friend | Contradiction | Tina (recency) | Tina (recency) |
| Mother | Contradiction | Lina overwrites Sara | Lina overwrites Sara |
| Doctor | Dr. Chen only (inverted) | Dr. Alvarez (recency) | Dr. Alvarez (recency) |
| Home | New York (recency) | Both homes | New York (recency) |
| Favorite City | Tbilisi (recency) | Tbilisi (recency) | Alternation noted |
| Rel. Status | Alternates / unclear | Married | Married |
| First Language | Contradiction | Contradiction | Russian (recency) |
| Diet | Not strictly vegetarian | Not strictly vegetarian | Not strictly vegetarian |

*Table 1. Selected results from 14 probes. Each cell summarizes the model's resolution of two sequential claims about the same predicate. The table illustrates not just incorrect answers, but instability: the same model shifts its interpretation of the relationship depending on prompt framing.*

### Prompt-sensitive ontology selection

The pattern across the probe set is that the prompt format changes the model's ontological commitment for the same observations. The 7B model treats "home" as plural-compatible under minimal but applies recency under analytic. It treats "sister" as plural under minimal but singular under analytic. It treats "hometown" as singular-default under minimal but allows two hometowns under analytic. These are not phrasing differences — they are different ontological assumptions triggered by whether the model is asked to produce structured reasoning. There is no stable relation policy. The model defines ontology on the fly, contingent on surface features of the prompt.

### Scaling: 3B vs. 7B

The 3B model produces frequent low-level failures (calling sister a contradiction, misidentifying recency, experiencing speaker-identity confusion). The 7B model eliminates many of these simple errors, but scaling does not resolve the ontology-selection problem. The 7B model still treats doctor as replacement, still overwrites mother, and still temporalizes best friend and favorite city. The failure mode shifts from low-level parsing confusion to higher-level ontology choice, which brings us closer to the research problem.

## 5. Implications for Memory Architecture

These findings have direct implications for the design of agent memory systems. Frozen LLMs are currently the interpreters in most memory pipelines — deciding how new claims relate to existing ones — and our probes show they perform this operation unreliably, perpetuating incomplete ontological assumptions from training and varying those assumptions based on prompt content. Any memory system that delegates claim-relation classification to an LLM will inherit these problems.

But the deeper issue is not that LLMs do this badly — it is that treating claim interpretation as a classification to be resolved at write time and discarded may be the wrong approach. Whether two claims coexist, revise one another, conflict, or are genuinely ambiguous is relational structure that should persist in the memory, remain inspectable, and inform how the memory is read. A memory system that stores only the winning interpretation has destroyed information that may later matter, and has done so based on an assumption that was never clarified.

## 6. Toward Relation-Aware Memory Evaluation

This points toward memory architectures where the relationships between claims are first-class objects in the substrate: not labels written by an LLM, but structural properties of how claims are stored and connected — properties that could emerge from learned mechanisms rather than being imposed by a classifier. Crucially, the relational structure of memory should have its own dynamics — updating, consolidating, and reorganizing as new claims arrive — rather than being fixed at the moment of storage.

This also reframes the evaluation question. Instead of asking whether a system retrieves the correct answer, we should ask whether the memory structure itself reliably represents the relational state of the claims it has received, including when that state is ambiguous, plural, or still evolving.

Existing benchmarks do not evaluate this. Memory retrieval benchmarks — LoCoMo (Maharana et al., ACL 2024), LongMemEval (ICLR 2025), LifeBench (2026) — treat user facts as static ground truth and ask whether the system can recall them. LoCoMo's adversarial (unanswerable) question category comes closest to testing ambiguity preservation, but it was excluded from evaluation by at least one major system (MemO) precisely because ground truth was unavailable — illustrating how the field's evaluation norms resist ambiguity rather than measure it. Factual editing benchmarks — CounterFact, ZsRE, and the multi-hop extension MQuAKE — assume singular correct answers and test whether an edit sticks without disturbing unrelated facts; plurality is outside their scope. Belief-revision benchmarks — Belief-R (Wilie et al., EMNLP 2024) and BeliefShift (Myakala et al., 2026) — test whether a model can update its reasoning when new premises arrive, but treat this as a single-turn logic problem rather than a property of stored memory structure. None of these benchmarks ask whether a system can represent the relational history of claims it has received, and that is the evaluation gap this project targets.

## 7. Limitations and Next Steps

The probes reported here are diagnostic, not a benchmark: 14 minimal scenarios tested qualitatively on two models from one family, with no inter-rater agreement. The contribution is identifying and naming the two failure modes, not claiming statistical coverage of their prevalence. Next steps: (i) extend to a different model family to separate Qwen-specific tendencies from general instruct-model behavior; (ii) replace qualitative coding with a structured rubric and report agreement on a subset; (iii) use the failure modes documented here to define the relation-typed graph schema in a follow-up prototype.