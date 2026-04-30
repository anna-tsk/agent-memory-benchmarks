"""Graph-memory QA condition for the LoCoMo benchmark.

Condition 3 of the experiment:
  retrieve claims from a typed-relation graph with their
  CONFLICTS / COEXISTS / SAME_AS / NEEDS_CLARIFICATION links
  explicitly present in the prompt.

Pipeline per conversation:
  1. build_graph(sample)       → ingest every turn → MemoryGraph
  2. retrieve(graph, question) → entity extraction → graph lookup + link expansion
  3. format_context(results)   → render as text for the answer LLM

The graph-memory module lives in 2026-04-22-graph-memory/ and is added
to sys.path below so this file can import from it without package setup.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

# add graph-memory module to path
_GRAPH_MEMORY_DIR = Path(__file__).resolve().parent.parent / "2026-04-22-graph-memory"
if str(_GRAPH_MEMORY_DIR) not in sys.path:
    sys.path.insert(0, str(_GRAPH_MEMORY_DIR))

from extractor import extract_entities_and_relations, ingest_claim  # noqa: E402
from graph import ClaimLink, ClaimNode, ClaimLinkType, MemoryGraph  # noqa: E402

if TYPE_CHECKING:
    from loader import Sample

# How many claims to include in the answer prompt (token budget guard).
MAX_CLAIMS_IN_CONTEXT = 30


def build_graph(sample: "Sample") -> MemoryGraph:
    """Ingest every turn of a LoCoMo sample into a MemoryGraph.

    Each turn becomes one ClaimNode. The extractor runs entity extraction
    and claim-link classification for every turn as it arrives, exactly
    as it would in a streaming memory system.

    Cost: O(n²) LLM calls where n = number of turns (each new claim is
    compared against all existing claims that share entities). For a
    typical LoCoMo conversation (~500 turns), this is expensive — run
    once and cache the result to disk if you want to re-score.
    """
    graph = MemoryGraph()
    total_turns = sum(len(s.turns) for s in sample.sessions)
    print(f"  ingesting {total_turns} turns into graph ({len(sample.sessions)} sessions)...")
    done = 0
    for session in sample.sessions:
        session_dt = _parse_session_dt(session.date_time)
        for turn in session.turns:
            ingest_claim(
                graph=graph,
                raw_text=turn.text,
                speaker=turn.speaker,
                timestamp=session_dt,
            )
            done += 1
            if done % 50 == 0:
                print(f"    {done}/{total_turns} turns ingested, {len(graph.claims)} claims, "
                      f"{len(graph.claim_links)} links")
    print(f"  graph complete: {len(graph.claims)} claims, "
          f"{len(graph.entities)} entities, {len(graph.claim_links)} claim links")
    return graph


def _parse_session_dt(date_time_str: str) -> datetime:
    """Best-effort parse of LoCoMo's free-form date strings like
    '4:04 pm on 20 January, 2023'. Falls back to utcnow() on failure."""
    try:
        from dateutil import parser as dtparser
        return dtparser.parse(date_time_str, fuzzy=True).replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def retrieve(
    graph: MemoryGraph,
    question: str,
    max_claims: int = MAX_CLAIMS_IN_CONTEXT,
) -> list[tuple[ClaimNode, list[ClaimLink]]]:
    """Return relevant (claim, links) pairs for a question.

    Steps:
    1. Extract entities from the question (one LLM call).
    2. Look up all graph claims that mention those entities (graph index
       lookup — no LLM, no embeddings).
    3. For each found claim, attach its outgoing ClaimLinks so the
       typed relations are available for formatting.
    4. Truncate to max_claims (most recent first — heuristic that
       keeps the answer focused on recent conversation context).
    """
    extraction = extract_entities_and_relations(question)
    q_entities = [e["name"].lower() for e in extraction.get("entities", [])]

    # collect claim ids for matched entities
    matched_claim_ids: set[str] = set()
    for entity in graph.entities.values():
        if entity.name.lower() in q_entities:
            for cid in graph._entity_claims_index.get(entity.id, []):
                matched_claim_ids.add(cid)

    if not matched_claim_ids:
        # fallback: return most recent claims (no match, give the LLM something)
        all_claims = sorted(graph.claims.values(), key=lambda c: c.timestamp, reverse=True)
        matched_claim_ids = {c.id for c in all_claims[:max_claims]}

    # sort by timestamp descending (most recent first)
    matched_claims = sorted(
        (graph.claims[cid] for cid in matched_claim_ids),
        key=lambda c: c.timestamp,
        reverse=True,
    )[:max_claims]

    # attach claim links for each
    result = []
    for claim in matched_claims:
        links = graph.get_claim_links_for_claim(claim.id)
        result.append((claim, links))

    return result


_LINK_LABELS = {
    ClaimLinkType.CONFLICTS: "CONFLICTS WITH",
    ClaimLinkType.COEXISTS: "COEXISTS WITH",
    ClaimLinkType.SAME_AS: "SAME AS",
    ClaimLinkType.NEEDS_CLARIFICATION: "NEEDS CLARIFICATION WITH",
}


def format_context(
    graph: MemoryGraph,
    results: list[tuple[ClaimNode, list[ClaimLink]]],
) -> str:
    """Render retrieved claims + their typed relations as a prompt context block.

    Format:
        [2023-01-20] Caroline: I went to a LGBTQ support group yesterday.
          → COEXISTS WITH: Caroline is a counselor.
          → NEEDS CLARIFICATION WITH: Marie is my best friend. / Tina is my best friend.

    The typed relation labels are explicitly in the prompt so the answer
    LLM can reason about them rather than silently picking a winner.
    """
    if not results:
        return "(no relevant claims retrieved)"

    lines: list[str] = []
    seen_claim_ids = {c.id for c, _ in results}

    for claim, links in results:
        date_str = claim.timestamp.strftime("%Y-%m-%d") if claim.timestamp else "?"
        lines.append(f"[{date_str}] {claim.speaker}: {claim.raw_text}")

        for link in links:
            other_id = link.claim_id_b if link.claim_id_a == claim.id else link.claim_id_a
            other = graph.claims.get(other_id)
            if other is None:
                continue
            label = _LINK_LABELS.get(link.relation_type, link.relation_type.value.upper())
            lines.append(f"  → {label}: {other.raw_text}")
            # if the linked claim isn't already in results, suppress it to avoid
            # duplication — it will appear when its own row is formatted
        lines.append("")

    return "\n".join(lines).rstrip()
