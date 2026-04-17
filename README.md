# First Memory Benchmark

Research scaffold for studying how persistent memory systems interpret
sequential natural-language observations.

The current question is not just whether a model can retrieve remembered facts.
It is how a model or memory system decides whether later claims should replace
earlier claims, coexist with them, remain ambiguous, or trigger clarification.

## Current Focus

The active probe track uses raw timestamped observations without explicit update
labels such as `revise`, `supersedes`, or `conflict`.

Example:

```text
2026-01-01T09:00:00Z | Marie is my best friend.
2026-02-15T12:00:00Z | Tina is my best friend.
```

The point is to observe what relation structure a model imposes on its own:
revision, coexistence, contradiction, ambiguity, category correction, or some
other interpretation.

Early runs suggest that small instruct models can be brittle and
prompt-sensitive. The larger research direction is an ambiguity-preserving
claim-to-belief layer between raw observations and active belief.

## Repository Layout

```text
data/examples/
  raw_observation_probes.json
  raw_observation_probes_v2.json

docs/session-notes/
  dated notes from exploratory sessions

runs/
  dated model output logs

scripts/
  probe_raw_context_qwen.py
  run_context_only_qwen.py

src/memory_benchmark/
  package code and placeholder utilities
```

## Running Raw Probes

Install model dependencies separately, then run from the repository root:

```bash
python3 scripts/probe_raw_context_qwen.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --condition all \
  --show-context \
  --show-prompt \
  --max-new-tokens 768 \
  --log-name all_probes_qwen25_7b_all_conditions
```

Run one targeted probe:

```bash
python3 scripts/probe_raw_context_qwen.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --probe-id probe_sister \
  --condition all \
  --show-context \
  --show-prompt
```

## Notes

This repo is an active research workspace. File formats, scripts, and prototype
APIs are expected to change.

Keep the README as a stable map. Put chronological working notes in
`docs/session-notes/` and model outputs in dated subfolders under `runs/`.
