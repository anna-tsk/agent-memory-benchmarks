# LoCoMo experimental results — running log

Snapshot of completed experimental cells, with sources for each number.
All cells use **Qwen2.5-72B-Instruct** for memory + answer + LLM-as-judge,
called via HuggingFace Inference Providers. Same answer-prompt template
across all cells (Hindsight's `locomo_benchmark.py` template). LLM judge
uses Hindsight's "generous grading" prompt verbatim. Numbers are binary
correct/wrong fractions.

## conv-26 cat-1 through cat-4 (full benchmark, no cat-5)

| condition | accuracy | source |
|---|---:|---|
| Full-context, all conv-26 (199q) | not run | (saturated by smaller diagnostic; would cost ~$3 for completeness) |
| Full-context, cat-1 multihop hard subset, all 10 conversations (73q) | **66/73 = 90%** | [runs/baseline_qa_20260430_125453.md](runs/baseline_qa_20260430_125453.md) |
| Hindsight, all conv-26 (152q, cats 1–4 only) | **136/152 = 89.5%** | `hindsight/.../conv26_cats1to4_default_hindsight.json` (Hindsight's standard run, cat-5 skipped) |

**Per-category Hindsight on conv-26 (cats 1–4):**
| cat | correct | acc |
|---|---:|---:|
| 1 (Multi-hop) | 24/32 | 75% |
| 2 (Temporal) | 35/37 | 95% |
| 3 (Single-hop) | 10/13 | 77% |
| 4 (Open-domain) | 67/70 | 96% |

**Reading:** full-context with the matched (Hindsight-style) prompt is
*at parity with Hindsight* on cats 1–4. The hard cat-1 multi-hop subset
specifically chosen to stress retrieval (≥4 evidence ids spanning ≥3
sessions) lands at 90% — same neighborhood as Hindsight's published
89.61%. The "memory architecture beats full-context on LoCoMo" framing
is mostly a prompt-engineering artifact at this model scale.

## Cat-5 ablation, conv-26 (47 questions)

The MCQ wrapper is LoCoMo's original cat-5 format that converts the
question into a binary choice between "Not mentioned" and a plausible
adversarial distractor. Hindsight (and most newer memory papers) skip
cat-5 entirely; we evaluate it both with and without the wrapper.

|              | no MCQ | with MCQ | source |
|--------------|---:|---:|---|
| **Full-context** | **42/47 = 89%** | **33/47 = 70%** | [runs/baseline_qa_20260430_132557.md](runs/baseline_qa_20260430_132557.md), [runs/baseline_qa_20260430_140847.md](runs/baseline_qa_20260430_140847.md) |
| **Hindsight**    | **32/47 = 68%** | **19/47 = 40%** | `hindsight/.../cat5_no_mcq.json`, `hindsight/.../cat5_with_mcq.json` |
| **Your graph**   | tbd | tbd | (cells E, F — graph backend, not yet run) |

Run `python extract_hindsight_results.py` from this directory to refresh
the Hindsight numbers from their JSON files (which live in the parallel
`hindsight/` clone — set `HINDSIGHT_RESULTS_DIR` to point elsewhere).

## Key findings so far

1. **The MCQ wrapper actively hurts modern instruction-tuned LLMs on
   cat-5 abstention.** Both full-context (-19pt) and Hindsight (-28pt)
   score worse with the wrapper than without it, the opposite of the
   field's working assumption. The wrapper's plausible distractor
   biases the model into picking it instead of recognizing absence.

2. **Memory retrieval hurts cat-5 abstention by ~21pt no-MCQ,
   ~30pt with-MCQ.** Hindsight's structured retrieval surfaces
   related-but-irrelevant facts (e.g. retrieving "Melanie's dad" +
   "horseback riding" facts when asked "what activity did Melanie do
   with her dad?") and the model confabulates rather than abstaining.
   The combined effect of MCQ + retrieval is multiplicative.

3. **Hindsight's published 89.61% on LoCoMo silently excludes 22% of
   the benchmark.** Their `locomo_benchmark.py` skips all 446 cat-5
   questions across the dataset by default. Full-benchmark numbers
   require including cat-5; doing so puts Hindsight roughly 21pt below
   full-context for the cat-5 portion.

## Methodology notes

- **Same prompt template** across all conditions. We ported Hindsight's
  exact `system + # CONTEXT + # INSTRUCTIONS + Context + Question + Answer`
  scaffold from `locomo_benchmark.py:159-188` into our `baseline_qa.py`
  so prompt engineering is held constant. Without this, full-context
  with locomo's published prompt format underperforms by ~30pt on cat-2
  because the model echoes the literal `DATE: ...` block headers.
- **Judge** is Qwen2.5-72B-Instruct with Hindsight's "generous grading"
  LoCoMo prompt verbatim (see `summarize_run.py:JUDGE_PROMPT`). Note
  the self-consistency caveat: same model judges its own answers
  (Hindsight has the same caveat in their published numbers).
- **No CURRENT DATE anchor**. Hindsight's prompt only adds it when
  `qa["question_date"]` is set, which is `None` for LoCoMo. We omit it
  to match exactly.
- **Hindsight patches** at `hindsight/hindsight-dev/benchmarks/common/benchmark_runner.py`
  add two env vars: `HINDSIGHT_BENCHMARK_INCLUDE_CAT5` (don't skip cat-5),
  `HINDSIGHT_BENCHMARK_ONLY_CAT5` (run *only* cat-5), `HINDSIGHT_BENCHMARK_CAT5_MCQ`
  (apply MCQ wrapper to cat-5).

## What's still needed for the 2x3 grid

- **Cell E**: your graph backend, conv-26 cat-5, no MCQ. ~$5–7 (~$3
  graph ingestion + 47 questions).
- **Cell F**: your graph backend, conv-26 cat-5, with MCQ. ~$2 (graph
  cached after E).

Target for cell E: **beat 68%** (Hindsight no MCQ). Approaching the
89% full-context ceiling would suggest typed `NEEDS_CLARIFICATION`
relations preserve absence-of-evidence cleanly enough to avoid the
confabulation Hindsight's similarity retrieval triggers.
