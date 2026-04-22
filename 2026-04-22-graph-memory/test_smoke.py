"""
Smoke test: end-to-end pipeline through extractor.py → graph.py.
Makes real LLM calls — requires ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import sys
from pathlib import Path

from graph import ClaimLinkType, MemoryGraph
from extractor import ingest_claim

CLAIMS = [
    ("Anna's favorite drink is tea.", "user"),
    ("Anna's favorite drink is coffee.", "user"),
    ("Anna lives in Los Angeles.", "user"),
]


def check(condition: bool, label: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


def main() -> int:
    graph = MemoryGraph()
    nodes = []

    print("Ingesting claims...")
    for text, speaker in CLAIMS:
        print(f"  → {text}")
        node = ingest_claim(graph, text, speaker)
        nodes.append(node)
    print()

    failures = 0

    print("Structural checks:")
    anna = graph.get_entity_by_name("Anna")
    if not check(anna is not None, "entity 'Anna' exists"):
        failures += 1

    if not check(len(graph.claims) == len(CLAIMS), f"{len(CLAIMS)} claims in graph"):
        failures += 1

    if not check(len(graph.relation_links) > 0, "at least one relation link created"):
        failures += 1

    conflicts = [
        cl for cl in graph.claim_links.values()
        if cl.relation_type == ClaimLinkType.CONFLICTS
    ]
    if not check(len(conflicts) >= 1, "at least one CONFLICTS link between drink claims"):
        failures += 1

    print()
    print("retrieval_structure for Anna:")
    if anna:
        rs = graph.retrieval_structure([anna.id])
        for k, v in rs.items():
            print(f"  {k}: {v}")
        if not check(rs["has_conflicts"], "has_conflicts is True"):
            failures += 1
    print()

    log_path = Path(__file__).parent / "logs" / "extractor_calls.jsonl"
    if not check(log_path.exists() and log_path.stat().st_size > 0, "extractor log written"):
        failures += 1

    print()
    if failures == 0:
        print("All checks passed.")
    else:
        print(f"{failures} check(s) FAILED.")
    return failures


if __name__ == "__main__":
    sys.exit(main())
