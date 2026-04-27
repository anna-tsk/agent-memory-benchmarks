from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from graph import ClaimLinkType, ClaimNode, MemoryGraph

# --- backend config ---
# "hf" uses a local HuggingFace model (no API key needed)
# "anthropic" uses the Anthropic API (requires ANTHROPIC_API_KEY)
BACKEND = "hf"
HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
HF_MAX_NEW_TOKENS = 1024

# every LLM call is logged — failures become the training spec for the learned classifier
_log_path = Path(__file__).parent / "logs" / "extractor_calls.jsonl"
_log_path.parent.mkdir(exist_ok=True)

# lazy-loaded HF model/tokenizer
_hf_model = None
_hf_tokenizer = None


def _load_hf():
    global _hf_model, _hf_tokenizer
    if _hf_model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading {HF_MODEL} (first call only)...")
        _hf_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
        _hf_model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL,
            torch_dtype="auto",
            device_map="auto",
        )
    return _hf_model, _hf_tokenizer


def _log(call_type: str, input_data: dict, raw: str, parsed: dict | None, error: str | None) -> None:
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "call_type": call_type,
        "input": input_data,
        "raw_output": raw,
        "parsed": parsed,
        "error": error,
    }
    with _log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _llm(prompt: str) -> str:
    if BACKEND == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    else:
        import torch
        model, tokenizer = _load_hf()
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=HF_MAX_NEW_TOKENS, do_sample=False)
        answer_ids = generated_ids[:, inputs.input_ids.shape[1]:]
        return tokenizer.batch_decode(answer_ids, skip_special_tokens=True)[0].strip()


def _parse_json(raw: str) -> tuple[dict | None, str | None]:
    """Strip code fences and parse. Returns (parsed, error)."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, str(e)


def extract_entities_and_relations(raw_text: str) -> dict:
    """
    Returns:
      {"entities": [{"name": str, "type_description": str}],
       "relations": [{"source": str, "predicate": str, "target": str}]}

    predicate uses the exact words from the claim, not a normalized form.
    type_description is free text — no fixed ontology.
    """
    prompt = f"""Extract entities and relations from this claim.

Claim: "{raw_text}"

Return JSON only:
{{
  "entities": [{{"name": "...", "type_description": "..."}}],
  "relations": [{{"source": "entity name", "predicate": "exact words from claim", "target": "entity name"}}]
}}

Rules:
- type_description is free text describing the entity naturally (e.g. "a person", "a city", "an emotion")
- predicate must use the exact words from the claim, not a normalized form
- only include relations between extracted entities
- entities with no relations to other entities are still valid to include"""

    raw = _llm(prompt)
    parsed, error = _parse_json(raw)
    if parsed is None:
        parsed = {"entities": [], "relations": []}

    _log("extraction", {"raw_text": raw_text}, raw, parsed, error)
    return parsed


def classify_claim_link(
    claim_a: str,
    claim_b: str,
) -> tuple[Optional[ClaimLinkType], str]:
    """
    Decides whether two claims that share an entity need a ClaimLink, and what type.
    Returns (ClaimLinkType or None, reasoning).

    None means no link is warranted — the claims are unrelated despite sharing an entity.
    NEEDS_CLARIFICATION means a link IS warranted but the type cannot be confidently
    chosen from the text alone — surfaced to the agent as an open question rather
    than silently collapsed.
    The reasoning string is logged as the primary signal for diagnosing classifier failures.
    """
    prompt = f"""Two claims share at least one entity. Decide if they need a link.

Claim A: "{claim_a}"
Claim B: "{claim_b}"

Choose one:
- "conflicts":            they assert incompatible things about the same slot
                          (e.g. "Anna's favorite drink is tea" vs "...is coffee")
- "coexists":             both can be true simultaneously about the same entity
                          (e.g. "Anna likes tea" and "Anna likes coffee")
- "same_as":              they say the same thing in different words
- "needs_clarification":  the claims clearly relate but you cannot confidently
                          distinguish between two of the above from the text alone,
                          and the difference would matter (e.g. "Marie is my best
                          friend" / "Tina is my best friend" — could be conflicts
                          if "best friend" is exclusive, coexists if not)
- "none":                 they share an entity but are otherwise unrelated

Prefer "needs_clarification" over guessing. Only use "none" when the claims are
genuinely unrelated despite the shared entity.

Return JSON only:
{{"relation_type": "conflicts" | "coexists" | "same_as" | "needs_clarification" | "none", "reasoning": "one sentence"}}"""

    raw = _llm(prompt)
    parsed, error = _parse_json(raw)

    result: Optional[ClaimLinkType] = None
    reasoning = ""

    if parsed:
        reasoning = parsed.get("reasoning", "")
        rt = parsed.get("relation_type", "none")
        result = {
            "conflicts": ClaimLinkType.CONFLICTS,
            "coexists": ClaimLinkType.COEXISTS,
            "same_as": ClaimLinkType.SAME_AS,
            "needs_clarification": ClaimLinkType.NEEDS_CLARIFICATION,
        }.get(rt)

    _log("claim_link_classification", {"claim_a": claim_a, "claim_b": claim_b}, raw, parsed, error)
    return result, reasoning


def ingest_claim(
    graph: MemoryGraph,
    raw_text: str,
    speaker: str,
    timestamp: Optional[datetime] = None,
) -> ClaimNode:
    """
    Full pipeline:
    1. Extract entities and relation links from raw text
    2. Add entities and claim to graph
    3. Classify claim links against all existing claims that share entities
    """
    if timestamp is None:
        timestamp = datetime.utcnow()

    extraction = extract_entities_and_relations(raw_text)

    entity_map: dict[str, str] = {}  # name -> entity_id
    for e in extraction.get("entities", []):
        node = graph.add_entity(e["name"], e.get("type_description", ""))
        entity_map[e["name"]] = node.id

    # claim node is created before relation links so its id is available for them
    claim = graph.add_claim(
        raw_text=raw_text,
        speaker=speaker,
        timestamp=timestamp,
        entity_ids=list(entity_map.values()),
        relation_link_ids=[],
    )

    relation_link_ids = []
    for r in extraction.get("relations", []):
        src_id = entity_map.get(r["source"])
        tgt_id = entity_map.get(r["target"])
        if src_id and tgt_id:
            link = graph.add_relation_link(
                source_entity_id=src_id,
                target_entity_id=tgt_id,
                predicate=r["predicate"],
                claim_id=claim.id,
            )
            relation_link_ids.append(link.id)

    claim.relation_link_ids = relation_link_ids

    # check all existing claims that share entities and classify links
    for other in graph.claims_sharing_entities(claim.id):
        link_type, _ = classify_claim_link(claim.raw_text, other.raw_text)
        if link_type is not None:
            graph.add_claim_link(claim.id, other.id, link_type)

    return claim
