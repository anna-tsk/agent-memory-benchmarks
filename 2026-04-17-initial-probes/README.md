# 2026-04-17 Initial Probes

Initial raw-observation probe track for the memory benchmark project.

This experiment asks how a model interprets later claims when the context does
not explicitly label them as revisions, contradictions, or clarifications.

## Layout

```text
data/
  examples/
  schema/

docs/
  session-notes/

runs/
  2026-04-17-initial-probe-runs/

scripts/
  probe_raw_context_qwen.py
  run_context_only_qwen.py

src/
  memory_benchmark/

tests/
```

## Running

From the repository root:

```bash
python3 2026-04-17-initial-probes/scripts/probe_raw_context_qwen.py --limit 1 --show-context
```

Or from inside this directory:

```bash
python3 scripts/probe_raw_context_qwen.py --limit 1 --show-context
```
