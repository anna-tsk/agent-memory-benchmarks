"""Re-score a baseline_qa_*.jsonl with the current rough-match heuristic.

Reads an existing JSONL run and prints a per-category summary using
`is_rough_match` from baseline_qa. Also writes a sibling .md file with
human-readable per-question rows (renders nicely in IDE preview).

Usage:
    python summarize_run.py runs/baseline_qa_<timestamp>.jsonl
    python summarize_run.py runs/baseline_qa_<timestamp>.jsonl --misses
    python summarize_run.py runs/baseline_qa_<timestamp>.jsonl --no-md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from baseline_qa import CATEGORY_LABELS, is_rough_match
from loader import QA


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

    out_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="path to a baseline_qa_*.jsonl")
    parser.add_argument(
        "--misses", action="store_true",
        help="also list every miss (rough_match=False) for inspection",
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
    misses: list[dict] = []
    scored: list[dict] = []  # rows + match flag, used by the .md writer

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
        match = is_rough_match(qa, r.get("prediction") or "")
        cat_total[cat] += 1
        if match:
            cat_hits[cat] += 1
        else:
            misses.append({"category": cat, **r})
        scored.append({**r, "match": match})

    total_hits = sum(cat_hits.values())
    total_count = sum(cat_total.values())
    total_skipped = sum(cat_skipped.values())

    print(f"file: {args.path}")
    print(f"rows: {len(rows)} ({total_skipped} skipped)")
    if rows:
        sample_ids = sorted({r["sample_id"] for r in rows})
        models = sorted({r.get("model", "?") for r in rows})
        backends = sorted({r.get("backend", "?") for r in rows})
        print(f"samples: {sample_ids}")
        print(f"model:   {models}  | backend: {backends}")

    print("\n" + "=" * 60)
    print("SUMMARY (rough substring match — first-cut, not final scoring)")
    print("=" * 60)
    print(f"{'category':<22} {'hits':>5} / {'total':<5}  {'≈acc':>5}  {'skip':>4}")
    cats = sorted(set(list(cat_total.keys()) + list(cat_skipped.keys())))
    for cat in cats:
        label = f"cat {cat} ({CATEGORY_LABELS.get(cat, '?')})"
        hits = cat_hits.get(cat, 0)
        total = cat_total.get(cat, 0)
        skipped = cat_skipped.get(cat, 0)
        acc = f"{hits/total*100:>4.0f}%" if total else "  n/a"
        print(f"{label:<22} {hits:>5} / {total:<5}  {acc:>5}  {skipped:>4}")
    print("-" * 60)
    overall_acc = f"{total_hits/total_count*100:>4.0f}%" if total_count else "  n/a"
    print(f"{'overall':<22} {total_hits:>5} / {total_count:<5}  {overall_acc:>5}  {total_skipped:>4}")

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
