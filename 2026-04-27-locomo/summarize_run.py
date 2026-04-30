"""Re-score a baseline_qa_*.jsonl with one of two scorers.

Reads an existing JSONL run and prints a per-category summary, optionally
also writing a sibling .md file with human-readable per-question rows.

Two scoring paths:
  - default:  rough substring match (`is_rough_match` from baseline_qa).
              Mechanical, no LLM calls, fast, strict on paraphrases.
  - --judge:  LLM-as-judge. Calls Qwen2.5-72B via HF Inference Providers
              with Hindsight's "generous grading" LoCoMo prompt — same
              scoring approach as Hindsight's published 89.61%, so the
              numbers become directly comparable. ~$1-2 for a full eval.

Usage:
    python summarize_run.py runs/baseline_qa_<timestamp>.jsonl
    python summarize_run.py runs/baseline_qa_<timestamp>.jsonl --judge
    python summarize_run.py runs/baseline_qa_<timestamp>.jsonl --misses
    python summarize_run.py runs/baseline_qa_<timestamp>.jsonl --no-md
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from baseline_qa import CATEGORY_LABELS, is_rough_match
from loader import QA


# --- LLM-as-judge configuration ---
# Mirrors Hindsight's default LoCoMo eval — same model family, same
# prompt, so re-scoring our runs produces numbers comparable to their
# published 89.61%. Override JUDGE_MODEL via env var to use a stronger
# (and less self-consistency-biased) judge for publication numbers.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-72B-Instruct")
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "https://router.huggingface.co/v1")

# Verbatim from hindsight-dev/benchmarks/common/benchmark_runner.py
# (the default LoCoMo path, lines 309-323).
JUDGE_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
        (1) a question (posed by one user to another user),
        (2) a 'gold' (ground truth) answer,
        (3) a generated answer
    which you will score as CORRECT/WRONG.

    The point of the question is to ask about something one user should know about the other user based on their prior conversations.
    The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
    Question: Do you remember what I got the last time I went to Hawaii?
    Gold answer: A shell necklace
    The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.
    There's an edge case where the actual answer can't be found in the data and in that case the gold answer will say so (e.g. 'You did not mention this information.'); if the generated answer says that it cannot be answered or it doesn't know all the details, it should be counted as CORRECT.
"""


_judge_client = None


def _get_judge_client():
    global _judge_client
    if _judge_client is None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit(
                "HF_TOKEN not set — required for --judge. Export it first: export HF_TOKEN=hf_..."
            )
        from openai import OpenAI
        _judge_client = OpenAI(base_url=JUDGE_BASE_URL, api_key=token)
    return _judge_client


def judge_one(qa: QA, prediction: str) -> tuple[bool | None, str]:
    """Run Hindsight's LoCoMo judge prompt on one (question, gold, prediction).

    Returns (is_correct, reasoning). is_correct is None if the judge call
    failed (counted as 'invalid' in the summary, not as wrong).

    For cat 5 the gold answer is None — pass the abstain phrasing the
    benchmark wrapper presented to the model so the judge can tell the
    model 'not mentioned' was the correct option.
    """
    client = _get_judge_client()

    if qa.category == 5:
        gold_for_judge = "Not mentioned in the conversation."
    else:
        gold_for_judge = str(qa.answer) if qa.answer is not None else ""

    user_msg = (
        f"{JUDGE_PROMPT}\n\n"
        f"Question: {qa.question}\n"
        f"Gold answer: {gold_for_judge}\n"
        f"Generated answer: {prediction}\n\n"
        'Reply with JSON only: {"reasoning": "one sentence", "correct": true|false}'
    )
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=200,
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return None, f"judge_error: {type(e).__name__}: {e}"

    parsed = _parse_judgment(raw)
    if parsed is None:
        return None, f"judge_parse_error: {raw[:200]!r}"
    return bool(parsed.get("correct")), str(parsed.get("reasoning", ""))


def _parse_judgment(raw: str) -> dict | None:
    text = raw.strip()
    # strip code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    # try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # fallback: pull first {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def write_markdown_report(out_path: Path, rows: list[dict], scored: list[dict],
                          cat_hits: dict, cat_total: dict, cat_skipped: dict) -> None:
    """Write a sibling .md file with summary table + per-question rows.

    `scored` parallels `rows` but with `match: bool` and a normalized
    `category_label` already attached for unskipped rows.
    """
    if not rows:
        return
    sample_ids = sorted({r["sample_id"] for r in rows})
    models = sorted({r.get("model", "?") for r in rows})
    backends = sorted({r.get("backend", "?") for r in rows})
    timestamps = sorted(r.get("timestamp", "") for r in rows)
    first_ts = timestamps[0] if timestamps else "?"

    total_hits = sum(cat_hits.values())
    total_count = sum(cat_total.values())
    total_skipped = sum(cat_skipped.values())

    lines = [
        f"# baseline_qa — {', '.join(sample_ids)}",
        "",
        f"- **timestamp:** {first_ts}",
        f"- **backend:** {', '.join(backends)}",
        f"- **model:** {', '.join(models)}",
        f"- **rows:** {len(rows)} ({total_skipped} skipped)",
        "",
        "## Summary",
        "",
        "| category | hits / total | ≈ acc |",
        "|---|---:|---:|",
    ]
    for cat in sorted(set(list(cat_total.keys()) + list(cat_skipped.keys()))):
        label = f"cat {cat} ({CATEGORY_LABELS.get(cat, '?')})"
        hits = cat_hits.get(cat, 0)
        total = cat_total.get(cat, 0)
        acc = f"{hits/total*100:.0f}%" if total else "n/a"
        lines.append(f"| {label} | {hits} / {total} | {acc} |")
    overall_acc = f"{total_hits/total_count*100:.0f}%" if total_count else "n/a"
    lines.append(f"| **overall** | **{total_hits} / {total_count}** | **{overall_acc}** |")
    lines += [
        "",
        "> Rough substring match — first-cut heuristic, not final scoring.",
        "",
        "## Questions",
    ]

    # group by category in numeric order
    by_cat: dict[int, list[dict]] = defaultdict(list)
    for entry in scored:
        by_cat[entry["category"]].append(entry)

    for cat in sorted(by_cat):
        lines.append("")
        lines.append(f"### cat {cat} — {CATEGORY_LABELS.get(cat, '?')}")
        for entry in by_cat[cat]:
            mark = "✓" if entry["match"] else "✗"
            gold = entry.get("gold")
            adv = entry.get("adversarial_answer")
            gold_display = gold if cat != 5 else f'"Not mentioned" (decoy: {adv})'
            evidence = ", ".join(entry.get("evidence", []) or []) or "—"
            pred = (entry.get("prediction") or "").strip()
            lines += [
                "",
                f"#### {mark} Q: {entry['question']}",
                f"- **gold:** {gold_display}",
                f"- **pred:** {pred}",
                f"- **evidence:** {evidence}",
            ]
            if "judge_reasoning" in entry:
                lines.append(f"- **judge:** {entry['judge_reasoning']}")

    out_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="path to a baseline_qa_*.jsonl")
    parser.add_argument(
        "--judge", action="store_true",
        help="use Hindsight-style LLM-as-judge instead of substring match. "
             "Calls JUDGE_MODEL via HF API; cost ~$1-2 for full eval.",
    )
    parser.add_argument(
        "--misses", action="store_true",
        help="also list every miss for inspection",
    )
    parser.add_argument(
        "--no-md", action="store_true",
        help="don't write a sibling .md report (default: write one)",
    )
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.path.open()]

    cat_hits: dict[int, int] = defaultdict(int)
    cat_total: dict[int, int] = defaultdict(int)
    cat_skipped: dict[int, int] = defaultdict(int)
    cat_invalid: dict[int, int] = defaultdict(int)
    misses: list[dict] = []
    scored: list[dict] = []

    if args.judge:
        # eager-init so we fail fast on missing HF_TOKEN before judging
        _get_judge_client()
        print(f"scoring with LLM-as-judge: {JUDGE_MODEL}")

    n_to_score = sum(1 for r in rows if not r.get("skipped"))
    done = 0

    for r in rows:
        cat = r["category"]
        if r.get("skipped"):
            cat_skipped[cat] += 1
            continue
        qa = QA(
            question=r["question"],
            answer=r.get("gold"),
            evidence=tuple(r.get("evidence", []) or []),
            category=cat,
            adversarial_answer=r.get("adversarial_answer"),
        )
        prediction = r.get("prediction") or ""

        if args.judge:
            verdict, reasoning = judge_one(qa, prediction)
            done += 1
            print(f"  [{done}/{n_to_score}] cat={cat} {'?' if verdict is None else ('✓' if verdict else '✗')}")
            if verdict is None:
                cat_invalid[cat] += 1
                scored.append({**r, "match": False, "judge_reasoning": reasoning, "judge_invalid": True})
                continue
            cat_total[cat] += 1
            if verdict:
                cat_hits[cat] += 1
            else:
                misses.append({"category": cat, "judge_reasoning": reasoning, **r})
            scored.append({**r, "match": verdict, "judge_reasoning": reasoning})
        else:
            match = is_rough_match(qa, prediction)
            cat_total[cat] += 1
            if match:
                cat_hits[cat] += 1
            else:
                misses.append({"category": cat, **r})
            scored.append({**r, "match": match})

    total_hits = sum(cat_hits.values())
    total_count = sum(cat_total.values())
    total_skipped = sum(cat_skipped.values())
    total_invalid = sum(cat_invalid.values())

    print(f"file: {args.path}")
    print(f"rows: {len(rows)} ({total_skipped} skipped)")
    if rows:
        sample_ids = sorted({r["sample_id"] for r in rows})
        models = sorted({r.get("model", "?") for r in rows})
        backends = sorted({r.get("backend", "?") for r in rows})
        print(f"samples: {sample_ids}")
        print(f"model:   {models}  | backend: {backends}")

    scorer_label = (
        f"LLM-as-judge ({JUDGE_MODEL}, Hindsight-style prompt)"
        if args.judge
        else "rough substring match — first-cut, not final scoring"
    )
    print("\n" + "=" * 60)
    print(f"SUMMARY ({scorer_label})")
    print("=" * 60)
    invalid_col = " invalid" if args.judge else ""
    print(f"{'category':<22} {'hits':>5} / {'total':<5}  {'≈acc':>5}  {'skip':>4}{invalid_col}")
    cats = sorted(set(list(cat_total.keys()) + list(cat_skipped.keys()) + list(cat_invalid.keys())))
    for cat in cats:
        label = f"cat {cat} ({CATEGORY_LABELS.get(cat, '?')})"
        hits = cat_hits.get(cat, 0)
        total = cat_total.get(cat, 0)
        skipped = cat_skipped.get(cat, 0)
        invalid = cat_invalid.get(cat, 0)
        acc = f"{hits/total*100:>4.0f}%" if total else "  n/a"
        invalid_str = f" {invalid:>7}" if args.judge else ""
        print(f"{label:<22} {hits:>5} / {total:<5}  {acc:>5}  {skipped:>4}{invalid_str}")
    print("-" * 60)
    overall_acc = f"{total_hits/total_count*100:>4.0f}%" if total_count else "  n/a"
    invalid_str = f" {total_invalid:>7}" if args.judge else ""
    print(f"{'overall':<22} {total_hits:>5} / {total_count:<5}  {overall_acc:>5}  {total_skipped:>4}{invalid_str}")

    if args.misses:
        print("\n" + "=" * 60)
        print(f"MISSES ({len(misses)})")
        print("=" * 60)
        for m in misses:
            cat = m["category"]
            label = CATEGORY_LABELS.get(cat, "?")
            gold = m.get("gold")
            adv = m.get("adversarial_answer")
            gold_display = gold if cat != 5 else f'"Not mentioned" (decoy: {adv})'
            print(f"\ncat {cat} ({label}) — {m['sample_id']}")
            print(f"  Q: {m['question']}")
            print(f"  gold: {gold_display}")
            print(f"  pred: {(m.get('prediction') or '').strip()}")

    if not args.no_md:
        md_path = args.path.with_suffix(".md")
        write_markdown_report(md_path, rows, scored, cat_hits, cat_total, cat_skipped)
        print(f"\nmarkdown report: {md_path}")


if __name__ == "__main__":
    main()
