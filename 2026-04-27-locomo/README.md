# 2026-04-27-locomo — LoCoMo benchmark loader

Thin wrapper around the [LoCoMo](https://github.com/snap-research/locomo)
question-answering benchmark (Maharana et al., ACL 2024). The locomo
repo is **not vendored** here — only a loader that reads its dataset
from a configurable path.

## Why a loader, not their eval scripts

LoCoMo's `task_eval/` ships with pinned conda dependencies (Linux+CUDA,
Python 3.9, openai 0.28, anthropic 0.32, transformers 4.35). Grafting
that onto this repo would freeze us to 2024 SDKs. We instead read just
the dataset and will write our own QA harness when we need one — using
current SDKs and shaping the evaluation around the memory probes in
this repo. Their [task_eval/evaluate_qa.py](../../locomo/task_eval/evaluate_qa.py)
and [task_eval/gpt_utils.py](../../locomo/task_eval/gpt_utils.py)
remain as references for scoring fidelity.

## Pointing at the data

The loader looks in this order:
1. `$LOCOMO_DIR` if set — should point at the directory containing `locomo10.json`.
2. The sibling folder `../locomo/data/` (default for this checkout).

Clone or already have the locomo repo somewhere else? Just:

```bash
export LOCOMO_DIR=/path/to/locomo/data
```

## Usage

```python
from loader import load_samples, iter_qa

samples = load_samples()              # list[Sample], 10 conversations
for sample_id, qa in iter_qa(samples):
    ...                               # 1986 QA pairs total
```

A `Sample` has `sample_id`, `speaker_a`, `speaker_b`, `sessions`
(chronological, with `date_time`), and `qa`. Use
`sample.turn_by_dia_id("D1:3")` to resolve `evidence` references.

## Dataset shape (verified by `test_smoke.py`)

- 10 conversations (`conv-26`, `conv-30`, `conv-41`–`conv-44`, `conv-47`–`conv-50`)
- 19–32 sessions per conversation, 369–689 turns per conversation
- 5882 turns and 1986 QA pairs total

QA categories (per LoCoMo's eval code):
| cat | count | meaning                                               |
|-----|------:|-------------------------------------------------------|
| 1   |   282 | single-hop fact lookup                                |
| 2   |   321 | temporal — eval prepends "Use DATE of CONVERSATION..." |
| 3   |    96 | multi-hop / preference inference                      |
| 4   |   841 | open-domain / commonsense                             |
| 5   |   446 | adversarial / unanswerable (has `adversarial_answer`) |

## Full-context baseline ([baseline_qa.py](baseline_qa.py))

Upper-bound reference: feed the **entire conversation** to Qwen2.5-Instruct
and ask each QA in turn. This is what our graph memory retrieval needs
to be measured against.

**Two backends, same prompt, same outputs:**

- `hf_api` (default) — HuggingFace Inference Providers, OpenAI-compatible
  router. Calls Qwen2.5-7B-Instruct via whichever provider HF routes to
  (Together, Fireworks, etc.). Pass-through pricing (no markup); ~$10
  per full LoCoMo eval. Uses `HF_TOKEN` env var.
- `local` — `transformers.AutoModelForCausalLM` on this machine, defaults
  to Qwen2.5-3B-Instruct (smaller for tractable CPU iteration). Slower
  but no API key required. Reproducibility safety net.

The prompt format mirrors locomo's `task_eval/gpt_utils.py` — each
session is rendered as `DATE: ...\nCONVERSATION:\n<turns>` so the
category-2 wrapper ("Use DATE of CONVERSATION...") aligns with the
explicit `DATE:` labels in the prompt. No separate system prompt; the
QA instruction goes at the end of the user message.

### Setup (HF API)

```bash
pip install openai
export HF_TOKEN=hf_...   # get from https://huggingface.co/settings/tokens
```

Verified prefix sizes (Qwen tokenizer) — all fit Qwen2.5's 32k context:

| sample  | sessions | turns | qa  | prefix tokens |
|---------|---------:|------:|----:|--------------:|
| conv-26 | 19       | 419   | 199 | 14,737        |
| conv-30 | 19       | 369   | 105 | 11,519        |
| conv-41 | 32       | 663   | 193 | 22,124        |
| conv-42 | 29       | 629   | 260 | 19,153        |
| conv-43 | 29       | 680   | 242 | 21,460        |
| conv-44 | 28       | 675   | 158 | 21,131        |
| conv-47 | 31       | 689   | 190 | 20,539        |
| conv-48 | 30       | 681   | 239 | 19,768        |
| conv-49 | 25       | 509   | 196 | 16,440        |
| conv-50 | 30       | 568   | 204 | 20,656        |

```bash
# Smoke run first — 5 questions on one sample, validates the pipeline:
python baseline_qa.py --sample-id conv-30 --max-questions 5

# Full run (1986 questions; ~10-30 min via HF API, hours locally):
python baseline_qa.py

# Force the local backend (no API key, slower):
python baseline_qa.py --backend local --max-questions 5
```

Output is one JSONL row per question in `runs/baseline_qa_<ts>.jsonl`
with `{sample_id, category, question, gold, prediction, evidence,
n_input_tokens, ...}`. Scoring is deferred to a separate script — eyeball
the JSONL first to sanity-check the predictions.

The baseline re-encodes the conversation prefix per question (no
KV-cache reuse). For 200 questions × 20k tokens × 10 conversations
that adds up; if it gets painful, the next optimization is sharing
`past_key_values` across questions in the same sample.

## Known data quirks

The loader splits multi-id evidence strings (`"D8:6; D9:17"`,
`"D9:1 D4:4 D4:6"`) on whitespace/`;`/`,` since some entries jam several
dia_ids into one string. Four genuinely malformed evidence ids remain
in the source — `D:11:26`, `D30:05`, `D10:19`, `D4:36` — and won't
resolve. The smoke test prints them; treat them as known bad rows.
