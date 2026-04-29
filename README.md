# Agent Memory Benchmarks

Building agent memory systems that preserve ambiguity, history, and contradiction + evaluating how existing pipelines flatten them into premature certainty.

## Experiments

### [2026-04-17-initial-probes/](2026-04-17-initial-probes/) — sequential claims in raw context
Probes how Qwen2.5 (3B and 7B) interprets subsequent claims in a
conversation when the context does not explicitly mark the relation
as revision, contradiction, or clarification (e.g. "Marie is my best
friend" followed days later by "Tina is my best friend"). See run
transcripts under [runs/](2026-04-17-initial-probes/runs/).

Two failure modes emerged: recency collapse (defaulting to later
claim regardless of context) and ontological collapse (the model
making assumptions about whether a qualifier like "best friend" can have more than one value, with those assumptions changing across prompt
formats for the same model and the same observations).

Writeup: [sequential-claim-interpretation.md](2026-04-17-initial-probes/sequential-claim-interpretation.md)

### [2026-04-21-embedding-noise-floor/](2026-04-21-embedding-noise-floor/) — embedding noise-floor diagnostic
One design idea explored was to store claims as representations in the
model's embedding space, retrieve via similarity, and inject retrieved
memories as soft tokens during query. We characterized Qwen2.5-3B's
pooled embeddings across three methods (`last_token`, `mean`,
`mean_midlayer`) to assess whether the space offers sufficient dynamic
range for this approach.

`mean_midlayer` showed near-zero variance across all pairs
(0.996–1.000). `mean` exhibited name-dominance bias — pairs sharing
an entity name clustered regardless of relationship type. `last_token`
had the widest dynamic range (0.88–0.98), but critical pairs like
"Anna lives in LA" vs "Anna lives in SF" (0.982) were
indistinguishable from paraphrases (0.961). This motivated moving
back towards token-level storage including the graph-based approach
below.

Report: [diagnostics/noise_floor_report.txt](2026-04-21-embedding-noise-floor/diagnostics/noise_floor_report.txt)

### [2026-04-22-graph-memory/](2026-04-22-graph-memory/) — text → entity/relation extraction + visualisation
Extracts claims into an entity/relation graph (`EntityNode`,
`RelationLink`, `ClaimNode` with `coexists` / `conflicts` / `same_as`
/ `needs_clarification` links) using either a local Qwen model. Every LLM call is logged to
[logs/extractor_calls.jsonl](2026-04-22-graph-memory/logs/extractor_calls.jsonl)
so failures can later become training data for a learned classifier.
[demo_visualize.py](2026-04-22-graph-memory/demo_visualize.py)
renders the graph as an interactive HTML file.

### [2026-04-27-locomo/](2026-04-27-locomo/) — LoCoMo benchmark loader
Thin loader for the [LoCoMo](https://github.com/snap-research/locomo)
long-term conversational QA dataset (Maharana et al., ACL 2024). 
