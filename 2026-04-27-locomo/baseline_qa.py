"""Full-context QA baseline on LoCoMo using Qwen2.5-Instruct.

For each conversation, format every session+turn into one block of text,
then loop the conversation's QA pairs and ask the model with the entire
conversation as prefix. This is the upper-bound baseline our graph
memory retrieval needs to be measured against.

Two backends, same prompt format, same outputs:
- "hf_api"  (default): HuggingFace Inference Providers (OpenAI-compatible
                       router), uses HF_TOKEN env var. Pass-through
                       provider pricing, no markup. Fast.
- "local":             Local transformers + AutoModelForCausalLM. Slower
                       but reproducible without an API key.

Output: runs/baseline_qa_<timestamp>.jsonl, one line per question.

Usage:
    python baseline_qa.py                              # default: hf_api, 7B
    python baseline_qa.py --backend local              # local 3B on CPU
    python baseline_qa.py --max-questions 5            # smoke run
    python baseline_qa.py --sample-id conv-26          # one conversation
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from loader import QA, Sample, load_samples


CATEGORY_LABELS = {
    1: "Multi-hop",
    2: "Temporal",
    3: "Single-hop",
    4: "Open-domain",
    5: "Adversarial",
}


# Which model the backend should call. The local default is smaller
# (3B) for tractable CPU iteration; the API default is 7B since cost
# and speed are the same either way.
LOCAL_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
HF_API_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# OpenAI-compatible router. HF passes through provider pricing 1:1.
HF_API_BASE_URL = "https://router.huggingface.co/v1"

MAX_NEW_TOKENS = 100
MAX_INPUT_TOKENS = 30_000  # Qwen2.5 native context is 32k; leave room for output

# Answer-prompt scaffolding mirrors hindsight-dev/benchmarks/locomo/locomo_benchmark.py
# (LoComoLLMAnswerGenerator.generate_answer, lines 159-188 of that file) verbatim,
# so all three of our experimental conditions (full-context, Hindsight memory, our
# typed-relation memory) wrap their context in the *same* prompt template. Only
# the {context} block content varies, isolating the memory variable from
# prompt-engineering differences.
#
# Notes on what differs from the original locomo paper's methodology:
# - No separate cat-2 wrapper ("Use DATE of CONVERSATION..."): Hindsight doesn't
#   apply it. We drop it for parity. This makes our locomo numbers slightly
#   different from the published locomo-paper format but directly comparable
#   to Hindsight's published 89.61%.
# - No cat-5 MCQ wrapper. Hindsight skips cat-5 questions entirely; we do too
#   by default (--skip-cat-5 is on). Pass --include-cat-5 to evaluate them
#   without an MCQ wrapper (their abstention is judged via the prompt's
#   instruction #7 about logical reasoning).
# - No # CURRENT DATE block: Hindsight only adds it when qa["question_date"]
#   is set, which it is *not* for LoCoMo. We omit it to match exactly.
ANSWER_SYSTEM_PROMPT = (
    "You are a helpful expert assistant answering questions from "
    "lme_experiment users based on the provided context."
)

ANSWER_USER_TEMPLATE = """
# CONTEXT:
You have access to facts and entities from a conversation.

# INSTRUCTIONS:
1. Carefully analyze all provided memories
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information or multiple instances of an event, say them all
5. Always convert relative time references to specific dates, months, or years.
6. Be as specific as possible when talking about people, places, and events
7. If the answer is not explicitly stated in the memories, use logical reasoning based on the information available to answer (e.g. calculate duration of an event from different memories).

Context:

{context}

Question: {question}
Answer:

"""


_local_model = None
_tokenizer = None
_api_client = None


def _get_tokenizer(model_id: str):
    """Always loaded — used for token-budget checks regardless of backend."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(model_id)
    return _tokenizer


def _get_local_model():
    global _local_model
    if _local_model is None:
        from transformers import AutoModelForCausalLM

        print(f"Loading {LOCAL_MODEL_NAME} (first call only)...")
        # Loads on CPU on this machine — no device_map="auto" because
        # MPS on macOS 13 can't handle bf16 / long-context matmul.
        _local_model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_NAME,
            torch_dtype="auto",
        )
        # Suppress "generation flags not valid" warnings under do_sample=False.
        for k in ("temperature", "top_p", "top_k"):
            if hasattr(_local_model.generation_config, k):
                setattr(_local_model.generation_config, k, None)
    return _local_model


def _get_api_client():
    global _api_client
    if _api_client is None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit(
                "HF_TOKEN not set. Get one at "
                "https://huggingface.co/settings/tokens, then "
                "`export HF_TOKEN=...` before running."
            )
        from openai import OpenAI
        _api_client = OpenAI(base_url=HF_API_BASE_URL, api_key=token)
    return _api_client


_PUNCT_RE = re.compile(rf"[{re.escape(string.punctuation)}]")


def _normalize(text) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Accepts non-string inputs (some LoCoMo gold answers are ints, e.g.
    'How many sessions...') and stringifies before normalizing.
    """
    if text is None or text == "":
        return ""
    text = str(text).lower()
    text = _PUNCT_RE.sub(" ", text)
    return " ".join(text.split())


def _evidence_session_count(qa: QA) -> int:
    """Number of distinct sessions referenced by a QA's evidence dia_ids.

    Dia_id format is 'D<session>:<turn>' (e.g. 'D14:7'). Used by the
    cat1_multihop filter to pick questions whose evidence is spread
    across many sessions — those are the ones where retrieval should
    plausibly help full-context (long-context attention dilution).
    """
    sessions: set[int] = set()
    for dia_id in qa.evidence:
        m = re.match(r"D(\d+):", dia_id)
        if m:
            sessions.add(int(m.group(1)))
    return len(sessions)


# Diagnostic filters for picking specific question types. Used to spend
# little money on questions where the comparison is most informative,
# instead of running thousands of questions where full-context already wins.
FILTERS = {
    "cat1_multihop": (
        lambda q: q.category == 1
        and len(q.evidence) >= 4
        and _evidence_session_count(q) >= 3
    ),
    "cat5": (lambda q: q.category == 5),
}


def is_rough_match(qa: QA, prediction: str) -> bool:
    """First-cut substring/abstain heuristic. Not the final scoring metric.

    - cat 5 (adversarial): correct iff the prediction text contains "not
      mentioned" — i.e. the model's *content* says it's abstaining,
      regardless of which option-letter it labelled the answer with
      (the model is often confused about (a) vs (b) anyway).
    - others: correct iff the gold (normalized) substring-overlaps with the
      prediction (normalized), or vice versa. Lenient — catches short
      paraphrases, misses long ones. Use it to spot patterns, not numbers.
    """
    pred_norm = _normalize(prediction)
    if not pred_norm:
        return False
    if qa.category == 5:
        return "not mentioned" in pred_norm
    if qa.answer is None:
        return False
    gold_norm = _normalize(qa.answer)
    if not gold_norm:
        return False
    return gold_norm in pred_norm or pred_norm in gold_norm


def format_conversation(sample: Sample) -> str:
    """Render conversation with session timestamps as soft markers.

    Avoids the literal 'DATE: ... CONVERSATION:' block headers we used
    before — those primed the model to copy them verbatim as date answers.
    A bracketed timestamp + speaker turn keeps the temporal information
    available without the echo-bait pattern.
    """
    blocks: list[str] = []
    for session in sample.sessions:
        block_lines = [f"[{session.session_id} — {session.date_time}]"]
        for turn in session.turns:
            block_lines.append(f"{turn.speaker}: {turn.text}")
        blocks.append("\n".join(block_lines))
    return "\n\n".join(blocks)


def format_question(qa: QA) -> str:
    """Pass the question through unchanged.

    We deliberately do NOT apply locomo's cat-2 hint ("Use DATE of
    CONVERSATION...") or cat-5 MCQ wrapper here — Hindsight applies
    neither, and we keep parity. Cat-5 questions are skipped at the
    sample-iteration level by default (--skip-cat-5).
    """
    return qa.question


def build_messages(sample: Sample, question_text: str, context: str) -> list[dict]:
    """Hindsight-equivalent answer prompt: system + user with sections.

    `context` is injected verbatim into the user message's `Context:` block.
    For full-context: pass `format_conversation(sample)`.
    For graph backend: pass the typed-claim render from graph_qa.format_context.
    """
    user_msg = ANSWER_USER_TEMPLATE.format(context=context, question=question_text)
    return [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def count_input_tokens(messages: list[dict], tokenizer) -> int:
    """Approximate token count using the local Qwen tokenizer.

    For the API backend this is approximate (the provider may apply a
    slightly different chat template), but it's tight enough for the
    MAX_INPUT_TOKENS safety check.
    """
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return len(tokenizer(text).input_ids)


def answer_one(messages: list[dict], backend: str, tokenizer) -> str:
    if backend == "hf_api":
        client = _get_api_client()
        resp = client.chat.completions.create(
            model=HF_API_MODEL,
            messages=messages,
            max_tokens=MAX_NEW_TOKENS,
            temperature=0,
        )
        return (resp.choices[0].message.content or "").strip()

    # local backend
    import torch
    model = _get_local_model()
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer_ids = gen[:, inputs.input_ids.shape[1]:]
    return tokenizer.batch_decode(answer_ids, skip_special_tokens=True)[0].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=("hf_api", "local", "graph"), default="hf_api",
        help=(
            "hf_api: HuggingFace Inference Providers, full conversation in context (default). "
            "local: local transformers model. "
            "graph: typed-relation graph memory retrieval (condition 3)."
        ),
    )
    parser.add_argument(
        "--sample-id", type=str, default=None,
        help="Restrict to a single sample (e.g. conv-26).",
    )
    parser.add_argument(
        "--max-questions", type=int, default=None,
        help="Cap questions per sample (for smoke runs). Applied AFTER cat-5 skipping.",
    )
    parser.add_argument(
        "--include-cat-5", action="store_true",
        help="Evaluate cat-5 (adversarial) questions too. Default: skip them, "
             "matching Hindsight's locomo benchmark which excludes cat-5 entirely.",
    )
    parser.add_argument(
        "--filter", type=str, default=None, choices=tuple(FILTERS.keys()),
        help=(
            "Restrict to a diagnostic question subset: "
            "cat1_multihop = cat-1 questions with ≥4 evidence ids spanning ≥3 sessions "
            "(stress-tests retrieval vs full-context); "
            "cat5 = cat-5 adversarial only (forces --include-cat-5)."
        ),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path(__file__).parent / "runs",
    )
    args = parser.parse_args()

    samples = load_samples()
    if args.sample_id:
        samples = [s for s in samples if s.sample_id == args.sample_id]
        if not samples:
            raise SystemExit(f"sample_id {args.sample_id!r} not found")

    args.out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out_dir / f"baseline_qa_{timestamp}.jsonl"

    model_id = HF_API_MODEL if args.backend in ("hf_api", "graph") else LOCAL_MODEL_NAME
    tokenizer = _get_tokenizer(model_id)
    if args.backend == "local":
        _get_local_model()
    else:
        _get_api_client()  # validates HF_TOKEN exists for both hf_api and graph backends

    if args.backend == "graph":
        from graph_qa import build_graph, retrieve, format_context
        print(f"backend=graph (retrieval model={model_id})")
    else:
        print(f"backend={args.backend}, model={model_id}")
    print(f"writing to {out_path}")

    cat_hits: dict[int, int] = defaultdict(int)
    cat_total: dict[int, int] = defaultdict(int)
    cat_skipped: dict[int, int] = defaultdict(int)

    # The cat5 filter implies --include-cat-5; otherwise the filter would
    # match nothing.
    include_cat_5 = args.include_cat_5 or args.filter == "cat5"

    for sample in samples:
        # Match Hindsight: skip cat-5 (adversarial) by default. Their
        # benchmark logs "Skipping 47 category=5 questions for conv-26".
        eligible_qa = [q for q in sample.qa if include_cat_5 or q.category != 5]
        if args.filter:
            predicate = FILTERS[args.filter]
            eligible_qa = [q for q in eligible_qa if predicate(q)]
        n_skipped_cat5 = sum(1 for q in sample.qa if q.category == 5) if not include_cat_5 else 0
        questions = eligible_qa[: args.max_questions] if args.max_questions else eligible_qa
        notes = []
        if n_skipped_cat5:
            notes.append(f"skipped {n_skipped_cat5} cat-5")
        if args.filter:
            notes.append(f"filter={args.filter}")
        notes_str = f" ({', '.join(notes)})" if notes else ""
        print(f"\n=== {sample.sample_id} ({sample.speaker_a} & {sample.speaker_b}) "
              f"— {len(questions)} questions{notes_str} ===")
        if not questions:
            print("  no questions match the filter on this sample, skipping.")
            continue

        # graph backend: build the memory graph once per sample before QA
        sample_graph = None
        if args.backend == "graph":
            sample_graph = build_graph(sample)

        for i, qa in enumerate(questions):
            question_text = format_question(qa)

            if args.backend == "graph":
                results = retrieve(sample_graph, question_text)
                context = format_context(sample_graph, results)
            else:
                context = format_conversation(sample)
            messages = build_messages(sample, question_text, context)

            n_input = count_input_tokens(messages, tokenizer)

            if n_input > MAX_INPUT_TOKENS:
                cat_skipped[qa.category] += 1
                print(f"  [{i+1}/{len(questions)}] cat={qa.category} SKIP — {n_input} tokens > {MAX_INPUT_TOKENS}")
                entry_skip = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "sample_id": sample.sample_id,
                    "category": qa.category,
                    "question": qa.question,
                    "gold": qa.answer,
                    "evidence": list(qa.evidence),
                    "prediction": None,
                    "skipped": True,
                    "skip_reason": f"input {n_input} > {MAX_INPUT_TOKENS}",
                    "n_input_tokens": n_input,
                    "backend": args.backend,
                    "model": model_id,
                }
                with out_path.open("a") as f:
                    f.write(json.dumps(entry_skip) + "\n")
                continue

            answer = answer_one(messages, args.backend, tokenizer)
            match = is_rough_match(qa, answer)
            cat_total[qa.category] += 1
            if match:
                cat_hits[qa.category] += 1

            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sample_id": sample.sample_id,
                "category": qa.category,
                "question": qa.question,
                "gold": qa.answer,
                "adversarial_answer": qa.adversarial_answer,
                "evidence": list(qa.evidence),
                "prediction": answer,
                "rough_match": match,
                "n_input_tokens": n_input,
                "backend": args.backend,
                "model": model_id,
            }
            with out_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")

            mark = "✓" if match else "✗"
            gold_display = qa.answer if qa.category != 5 else f'"Not mentioned" (decoy: {qa.adversarial_answer})'
            pred_display = answer.replace("\n", " ").strip()
            print(f"  [{i+1}/{len(questions)}] cat={qa.category} {mark} ({n_input} tok)")
            print(f"     Q: {qa.question[:120]}")
            print(f"     gold: {gold_display}")
            print(f"     pred: {pred_display[:200]}")

    # End-of-run summary
    total_hits = sum(cat_hits.values())
    total_count = sum(cat_total.values())
    total_skipped = sum(cat_skipped.values())

    print("\n" + "=" * 60)
    print("SUMMARY (rough substring match — first-cut heuristic, not final scoring)")
    print("=" * 60)
    print(f"{'category':<22} {'hits':>5} / {'total':<5}  {'≈acc':>5}  {'skip':>4}")
    for cat in sorted(set(list(cat_total.keys()) + list(cat_skipped.keys()))):
        label = f"cat {cat} ({CATEGORY_LABELS.get(cat, '?')})"
        hits = cat_hits.get(cat, 0)
        total = cat_total.get(cat, 0)
        skipped = cat_skipped.get(cat, 0)
        acc = f"{hits/total*100:>4.0f}%" if total else "  n/a"
        print(f"{label:<22} {hits:>5} / {total:<5}  {acc:>5}  {skipped:>4}")
    print("-" * 60)
    overall_acc = f"{total_hits/total_count*100:>4.0f}%" if total_count else "  n/a"
    print(f"{'overall':<22} {total_hits:>5} / {total_count:<5}  {overall_acc:>5}  {total_skipped:>4}")
    print(f"\nresults in {out_path}")


if __name__ == "__main__":
    main()
