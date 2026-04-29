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
HF_API_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# OpenAI-compatible router. HF passes through provider pricing 1:1.
HF_API_BASE_URL = "https://router.huggingface.co/v1"

MAX_NEW_TOKENS = 100
MAX_INPUT_TOKENS = 30_000  # Qwen2.5 native context is 32k; leave room for output

# Prompt scaffolding mirrors locomo/task_eval/gpt_utils.py — no separate
# system prompt, everything goes in the user message. The DATE: /
# CONVERSATION: labels are what the cat-2 wrapper ("Use DATE of
# CONVERSATION...") expects to see in the prompt.
CONV_START_PROMPT = (
    "Below is a conversation between two people: {a} and {b}. "
    "The conversation takes place over multiple days and the date of each "
    "conversation is written at the beginning of the conversation.\n\n"
)
QA_PROMPT_TAIL = (
    "\n\nBased on the above context, write an answer in the form of a short "
    "phrase for the following question. Answer with exact words from the "
    "context whenever possible.\n\nQuestion: {q} Short answer:"
)


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


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return " ".join(text.split())


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
    """Locomo-style: each session as 'DATE: ...\\nCONVERSATION:\\n<turns>'."""
    blocks: list[str] = []
    for session in sample.sessions:
        turns_text = "\n".join(f"{t.speaker}: {t.text}" for t in session.turns)
        blocks.append(f"DATE: {session.date_time}\nCONVERSATION:\n{turns_text}")
    return "\n\n".join(blocks)


def format_question(qa: QA) -> str:
    """Apply LoCoMo's category-specific question wrappers.

    Mirrors locomo/task_eval/gpt_utils.py:243-256, except we don't
    randomize cat-5 option order (deterministic baseline).
    """
    if qa.category == 2:
        return qa.question + " Use DATE of CONVERSATION to answer with an approximate date."
    if qa.category == 5:
        adversarial = qa.adversarial_answer or "(no adversarial answer provided)"
        return (
            qa.question
            + f" Select the correct answer: (a) Not mentioned in the conversation, (b) {adversarial}."
        )
    return qa.question


def build_messages(sample: Sample, question_text: str) -> list[dict]:
    user_msg = (
        CONV_START_PROMPT.format(a=sample.speaker_a, b=sample.speaker_b)
        + format_conversation(sample)
        + QA_PROMPT_TAIL.format(q=question_text)
    )
    return [{"role": "user", "content": user_msg}]


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
        "--backend", choices=("hf_api", "local"), default="hf_api",
        help="hf_api: HuggingFace Inference Providers (default). local: transformers on this machine.",
    )
    parser.add_argument(
        "--sample-id", type=str, default=None,
        help="Restrict to a single sample (e.g. conv-26).",
    )
    parser.add_argument(
        "--max-questions", type=int, default=None,
        help="Cap questions per sample (for smoke runs).",
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

    model_id = HF_API_MODEL if args.backend == "hf_api" else LOCAL_MODEL_NAME
    tokenizer = _get_tokenizer(model_id)
    if args.backend == "local":
        _get_local_model()  # eager load so the first sample's first question doesn't pause
    else:
        _get_api_client()  # validates HF_TOKEN exists

    print(f"backend={args.backend}, model={model_id}")
    print(f"writing to {out_path}")

    cat_hits: dict[int, int] = defaultdict(int)
    cat_total: dict[int, int] = defaultdict(int)
    cat_skipped: dict[int, int] = defaultdict(int)

    for sample in samples:
        questions = sample.qa[: args.max_questions] if args.max_questions else sample.qa
        print(f"\n=== {sample.sample_id} ({sample.speaker_a} & {sample.speaker_b}) — {len(questions)} questions ===")

        for i, qa in enumerate(questions):
            question_text = format_question(qa)
            messages = build_messages(sample, question_text)
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
