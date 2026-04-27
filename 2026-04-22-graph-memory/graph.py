from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class ClaimLinkType(Enum):
    COEXISTS = "coexists"
    CONFLICTS = "conflicts"
    SAME_AS = "same_as"
    NEEDS_CLARIFICATION = "needs_clarification"


@dataclass
class EntityNode:
    id: str
    name: str
    type_description: str  # free-text, no fixed enum
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RelationLink:
    id: str
    source_entity_id: str
    target_entity_id: str
    predicate: str          # taken directly from claim ("lives in", "is", "was", etc.)
    claim_id: str           # which claim created this link
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ClaimNode:
    id: str
    raw_text: str
    speaker: str
    timestamp: datetime
    entity_ids: list[str]
    relation_link_ids: list[str]
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ClaimLink:
    id: str
    claim_id_a: str
    claim_id_b: str
    relation_type: ClaimLinkType
    created_at: datetime = field(default_factory=datetime.utcnow)


class MemoryGraph:
    def __init__(self):
        self.entities: dict[str, EntityNode] = {}
        self.relation_links: dict[str, RelationLink] = {}
        self.claims: dict[str, ClaimNode] = {}
        self.claim_links: dict[str, ClaimLink] = {}

        self._entity_name_index: dict[str, str] = {}           # name -> entity_id
        self._entity_claims_index: dict[str, list[str]] = {}   # entity_id -> [claim_id]
        self._claim_links_index: dict[str, list[str]] = {}     # claim_id -> [claim_link_id]

    # --- entity graph ---

    def add_entity(self, name: str, type_description: str = "") -> EntityNode:
        if name in self._entity_name_index:
            return self.entities[self._entity_name_index[name]]
        node = EntityNode(id=str(uuid.uuid4()), name=name, type_description=type_description)
        self.entities[node.id] = node
        self._entity_name_index[name] = node.id
        self._entity_claims_index[node.id] = []
        return node

    def add_relation_link(
        self,
        source_entity_id: str,
        target_entity_id: str,
        predicate: str,
        claim_id: str,
    ) -> RelationLink:
        link = RelationLink(
            id=str(uuid.uuid4()),
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            predicate=predicate,
            claim_id=claim_id,
        )
        self.relation_links[link.id] = link
        return link

    def get_entity_by_name(self, name: str) -> Optional[EntityNode]:
        eid = self._entity_name_index.get(name)
        return self.entities[eid] if eid else None

    def get_relation_links_for_entities(
        self, entity_id_a: str, entity_id_b: str
    ) -> list[RelationLink]:
        return [
            link for link in self.relation_links.values()
            if {link.source_entity_id, link.target_entity_id} == {entity_id_a, entity_id_b}
        ]

    # --- claims graph ---

    def add_claim(
        self,
        raw_text: str,
        speaker: str,
        timestamp: datetime,
        entity_ids: list[str],
        relation_link_ids: list[str],
    ) -> ClaimNode:
        claim = ClaimNode(
            id=str(uuid.uuid4()),
            raw_text=raw_text,
            speaker=speaker,
            timestamp=timestamp,
            entity_ids=entity_ids,
            relation_link_ids=relation_link_ids,
        )
        self.claims[claim.id] = claim
        for eid in entity_ids:
            self._entity_claims_index.setdefault(eid, []).append(claim.id)
        self._claim_links_index[claim.id] = []
        return claim

    def add_claim_link(
        self,
        claim_id_a: str,
        claim_id_b: str,
        relation_type: ClaimLinkType,
    ) -> ClaimLink:
        link = ClaimLink(
            id=str(uuid.uuid4()),
            claim_id_a=claim_id_a,
            claim_id_b=claim_id_b,
            relation_type=relation_type,
        )
        self.claim_links[link.id] = link
        self._claim_links_index.setdefault(claim_id_a, []).append(link.id)
        self._claim_links_index.setdefault(claim_id_b, []).append(link.id)
        return link

    def get_claims_for_entity(self, entity_id: str) -> list[ClaimNode]:
        return [self.claims[cid] for cid in self._entity_claims_index.get(entity_id, [])]

    def get_claim_links_for_claim(self, claim_id: str) -> list[ClaimLink]:
        return [self.claim_links[lid] for lid in self._claim_links_index.get(claim_id, [])]

    def claims_sharing_entities(self, claim_id: str) -> list[ClaimNode]:
        """All other claims that share at least one entity with this claim."""
        claim = self.claims[claim_id]
        seen: set[str] = set()
        result = []
        for eid in claim.entity_ids:
            for cid in self._entity_claims_index.get(eid, []):
                if cid != claim_id and cid not in seen:
                    seen.add(cid)
                    result.append(self.claims[cid])
        return result

    # --- retrieval ---

    def retrieval_structure(self, entity_ids: list[str]) -> dict:
        """
        Structural metadata about the claims touching a set of entities.
        Passed to the reader LM alongside raw claim text.
        """
        claim_ids: set[str] = set()
        for eid in entity_ids:
            for cid in self._entity_claims_index.get(eid, []):
                claim_ids.add(cid)

        links: list[ClaimLink] = []
        seen_link_ids: set[str] = set()
        for cid in claim_ids:
            for link in self.get_claim_links_for_claim(cid):
                if link.id not in seen_link_ids:
                    seen_link_ids.add(link.id)
                    links.append(link)

        link_type_counts: dict[str, int] = {}
        for link in links:
            t = link.relation_type.value
            link_type_counts[t] = link_type_counts.get(t, 0) + 1

        return {
            "claim_count": len(claim_ids),
            "claim_link_count": len(links),
            "link_type_counts": link_type_counts,
            "has_conflicts": link_type_counts.get("conflicts", 0) > 0,
            "has_needs_clarification": link_type_counts.get("needs_clarification", 0) > 0,
        }
