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

import pickle
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
MAX_CLAIMS_IN_CONTEXT = 20
# Cap links rendered per claim. Conv-26's 14,069 links over 419 claims
# averages ~67 links/claim — without a cap, 20 claims × 67 links blow
# the context budget. 5 keeps the typed-relation signal visible.
MAX_LINKS_PER_CLAIM = 5

# Ingestion is the expensive step (O(n²) LLM calls). Cache the built
# graph per sample so subsequent runs reuse it. Delete a cache file by
# hand if the extractor changes and you want to re-ingest from scratch.
_CACHE_DIR = Path(__file__).resolve().parent / "graph_cache"


def _cache_path(sample_id: str) -> Path:
    return _CACHE_DIR / f"{sample_id}.pkl"


def save_graph(graph: MemoryGraph, sample_id: str) -> Path:
    _CACHE_DIR.mkdir(exist_ok=True)
    path = _cache_path(sample_id)
    with path.open("wb") as f:
        pickle.dump(graph, f)
    return path


def load_graph(sample_id: str) -> MemoryGraph | None:
    path = _cache_path(sample_id)
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def build_graph(sample: "Sample") -> MemoryGraph:
    """Ingest every turn of a LoCoMo sample into a MemoryGraph.

    Cache-aware: if `graph_cache/<sample_id>.pkl` exists, load it instead
    of re-ingesting. Delete the cache file by hand to force a rebuild.

    Each turn becomes one ClaimNode. The extractor runs entity extraction
    and claim-link classification for every turn as it arrives, exactly
    as it would in a streaming memory system.

    Cost on cache miss: O(n²) LLM calls where n = number of turns (each
    new claim is compared against all existing claims that share
    entities). For a typical LoCoMo conversation (~500 turns), this is
    expensive — typically ~$5–10 in API calls.
    """
    cached = load_graph(sample.sample_id)
    if cached is not None:
        print(
            f"  loaded cached graph for {sample.sample_id}: "
            f"{len(cached.claims)} claims, {len(cached.entities)} entities, "
            f"{len(cached.claim_links)} claim links "
            f"(delete {_cache_path(sample.sample_id)} to rebuild)"
        )
        return cached

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
    cache_path = save_graph(graph, sample.sample_id)
    print(f"  saved graph cache to {cache_path}")
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
    2. Look up all graph claims that mention those entities.
    3. If no entity match, return EMPTY context — for cat-5 (adversarial)
       questions, this is the correct signal: "the memory has nothing
       relevant to surface, please abstain." A fallback to "30 most
       recent claims" would actively mislead the model into confabulating
       from irrelevant context, the exact failure mode this experiment
       is testing other systems for.
    4. Truncate to max_claims (most recent first).
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
        # Intentionally return empty. format_context() will render a
        # "(no relevant claims retrieved)" marker that signals absence.
        return []

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
    top_level_ids = {c.id for c, _ in results}

    for claim, links in results:
        date_str = claim.timestamp.strftime("%Y-%m-%d") if claim.timestamp else "?"
        lines.append(f"[{date_str}] {claim.speaker}: {claim.raw_text}")

        # Cap links rendered per claim and dedupe against claims already
        # shown as their own top-level row. Without this, conv-26's
        # 14k+ link graph blows the 30k context budget per question.
        shown = 0
        for link in links:
            if shown >= MAX_LINKS_PER_CLAIM:
                break
            other_id = link.claim_id_b if link.claim_id_a == claim.id else link.claim_id_a
            if other_id in top_level_ids:
                continue  # already shown as its own row
            other = graph.claims.get(other_id)
            if other is None:
                continue
            label = _LINK_LABELS.get(link.relation_type, link.relation_type.value.upper())
            lines.append(f"  → {label}: {other.raw_text}")
            shown += 1
        lines.append("")

    return "\n".join(lines).rstrip()
