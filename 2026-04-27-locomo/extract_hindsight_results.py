"""Print headline numbers + per-question details from Hindsight benchmark JSONs.

Hindsight's locomo_benchmark.py writes results to
  hindsight/hindsight-dev/benchmarks/locomo/results/<file>.json

That folder is in a parallel repo (sibling to agent-memory-benchmarks);
this script reads from a configurable path so we can keep our research
log self-contained even if the hindsight clone moves.

Usage:
    python extract_hindsight_results.py                   # print all known runs
    python extract_hindsight_results.py PATH.json         # print one file
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


_DEFAULT_DIR = (
    Path(os.environ.get("HINDSIGHT_RESULTS_DIR")
         or Path(__file__).resolve().parent.parent.parent
         / "hindsight"
         / "hindsight-dev"
         / "benchmarks"
         / "locomo"
         / "results")
)

# Each entry: (file in HINDSIGHT_RESULTS_DIR, label, notes)
KNOWN_RUNS = [
    (
        "conv26_cats1to4_default_hindsight.json",
        "Hindsight default (cats 1-4 only, no cat-5)",
        "Reference 'Hindsight + Qwen 72B on conv-26' number; matches their "
        "published 89.61% methodology of skipping cat-5.",
    ),
    (
        "cat5_no_mcq.json",
        "Hindsight cat-5 only, NO MCQ wrapper",
        "Spontaneous abstention test. Cell C of the 2x3 grid.",
    ),
    (
        "cat5_with_mcq.json",
        "Hindsight cat-5 only, WITH MCQ wrapper",
        "Original locomo cat-5 methodology applied to Hindsight. Cell D.",
    ),
]


def summarize(path: Path) -> None:
    with path.open() as f:
        data = json.load(f)

    print(f"\n=== {path.name} ===")
    overall = data.get("overall_accuracy")
    correct = data.get("total_correct")
    total = data.get("total_questions")
    invalid = data.get("total_invalid")
    print(f"  overall: {correct}/{total} = {overall:.1f}% (invalid: {invalid})")

    items = data.get("item_results", [])
    for item in items:
        m = item.get("metrics", {})
        cat_stats = m.get("category_stats", {})
        print(f"  {item.get('item_id')}: {m.get('correct')}/{m.get('total')} (invalid: {m.get('invalid')})")
        for c in sorted(cat_stats):
            s = cat_stats[c]
            n_correct = s.get("correct", 0)
            n_total = s.get("total", 0)
            pct = (n_correct / n_total * 100) if n_total else 0.0
            print(f"    cat {c}: {n_correct}/{n_total} = {pct:.0f}%")

    # Sanity-check: confirm whether MCQ wrapper is present
    detailed = items[0].get("metrics", {}).get("detailed_results", []) if items else []
    if detailed:
        with_mcq = sum(1 for r in detailed if "Select the correct answer" in r.get("question", ""))
        print(f"  MCQ wrapper present: {with_mcq}/{len(detailed)} questions")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, nargs="?", default=None)
    parser.add_argument(
        "--results-dir", type=Path, default=_DEFAULT_DIR,
        help="path to hindsight's results directory (or set HINDSIGHT_RESULTS_DIR)",
    )
    args = parser.parse_args()

    if args.path:
        if not args.path.exists():
            sys.exit(f"file not found: {args.path}")
        summarize(args.path)
        return

    if not args.results_dir.exists():
        sys.exit(
            f"hindsight results dir not found: {args.results_dir}\n"
            f"Set HINDSIGHT_RESULTS_DIR or pass --results-dir."
        )

    print(f"reading hindsight results from {args.results_dir}")
    for fname, label, notes in KNOWN_RUNS:
        path = args.results_dir / fname
        if not path.exists():
            print(f"\n=== {fname}: NOT FOUND ===")
            continue
        print(f"\n--- {label} ---")
        print(f"  notes: {notes}")
        summarize(path)


if __name__ == "__main__":
    main()
