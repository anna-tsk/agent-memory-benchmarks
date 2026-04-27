"""
Render a MemoryGraph as an interactive HTML file using pyvis.

Two visual layers:
  - Entity nodes (blue) connected via RelationLink nodes (gray diamonds, labeled with predicate)
  - Claim nodes (green) connected by ClaimLinks (colored by type)
  - Dashed lines: claim → its entity nodes AND claim → its relation link nodes

Usage:
    from visualize import render
    render(graph, "graph.html")   # opens in browser by default
"""
from __future__ import annotations

import webbrowser
from pathlib import Path

from pyvis.network import Network

from graph import ClaimLinkType, MemoryGraph

# --- visual config ---
ENTITY_COLOR = "#4A90D9"
CLAIM_COLOR = "#5CB85C"
RELATION_LINK_COLOR = "#AAAAAA"

CLAIM_LINK_COLORS = {
    ClaimLinkType.CONFLICTS:           "#E74C3C",
    ClaimLinkType.COEXISTS:            "#F0AD4E",
    ClaimLinkType.SAME_AS:             "#9B59B6",
    ClaimLinkType.NEEDS_CLARIFICATION: "#48C9B0",
}


def render(graph: MemoryGraph, output_path: str | Path = "graph.html", open_browser: bool = True) -> Path:
    output_path = Path(output_path)

    net = Network(height="750px", width="100%", bgcolor="#1a1a2e", font_color="white", directed=True)
    net.barnes_hut(gravity=-10000, central_gravity=0.2, spring_length=220, spring_strength=0.04, damping=0.15, overlap=0.5)

    for entity in graph.entities.values():
        net.add_node(
            f"e:{entity.id}",
            label=entity.name,
            title=f"{entity.name}\n{entity.type_description}",
            color=ENTITY_COLOR,
            shape="ellipse",
            size=20,
            font={"size": 14, "color": "white"},
        )

    # RelationLinks become small diamond nodes so claims can point at them
    for link in graph.relation_links.values():
        net.add_node(
            f"r:{link.id}",
            label=link.predicate,
            title=f"predicate: {link.predicate}",
            color=RELATION_LINK_COLOR,
            shape="diamond",
            size=10,
            font={"size": 9, "color": "#CCCCCC"},
        )
        net.add_edge(f"e:{link.source_entity_id}", f"r:{link.id}",
                     color=RELATION_LINK_COLOR, arrows="to", width=1.5)
        net.add_edge(f"r:{link.id}", f"e:{link.target_entity_id}",
                     color=RELATION_LINK_COLOR, arrows="to", width=1.5)

    for claim in graph.claims.values():
        short = claim.raw_text if len(claim.raw_text) <= 60 else claim.raw_text[:57] + "..."
        net.add_node(
            f"c:{claim.id}",
            label=short,
            title=f'"{claim.raw_text}"\nspeaker: {claim.speaker}\n{claim.timestamp.strftime("%Y-%m-%d %H:%M")}',
            color=CLAIM_COLOR,
            shape="box",
            size=15,
            font={"size": 11, "color": "white"},
        )
        dashed_edge = {"color": {"color": ENTITY_COLOR, "opacity": 0.3}, "dashes": True, "arrows": "to", "width": 1}
        for eid in claim.entity_ids:
            net.add_edge(f"c:{claim.id}", f"e:{eid}", **dashed_edge)
        for rlid in claim.relation_link_ids:
            net.add_edge(f"c:{claim.id}", f"r:{rlid}", **dashed_edge)

    for cl in graph.claim_links.values():
        color = CLAIM_LINK_COLORS.get(cl.relation_type, "#FFFFFF")
        net.add_edge(
            f"c:{cl.claim_id_a}",
            f"c:{cl.claim_id_b}",
            label=cl.relation_type.value,
            color=color,
            title=cl.relation_type.value,
            arrows="to",
            width=2.5,
            font={"size": 11, "color": color},
        )

    legend_html = """
    <div style="position:fixed;bottom:20px;left:20px;background:#2a2a4e;padding:12px 16px;
                border-radius:8px;font-family:sans-serif;font-size:12px;color:white;z-index:999">
      <b>Legend</b><br><br>
      <span style="color:#4A90D9">&#9679;</span> Entity &nbsp;
      <span style="color:#5CB85C">&#9632;</span> Claim &nbsp;
      <span style="color:#AAAAAA">&#9670;</span> Relation<br><br>
      <span style="color:#AAAAAA">&#8212;</span> entity–relation–entity<br>
      <span style="color:#E74C3C">&#8212;</span> conflicts<br>
      <span style="color:#F0AD4E">&#8212;</span> coexists<br>
      <span style="color:#9B59B6">&#8212;</span> same_as<br>
      <span style="color:#48C9B0">&#8212;</span> needs_clarification<br>
      <span style="color:#4A90D9;opacity:0.5">- -</span> claim→entity/relation
    </div>
    """

    net.set_options("""
    {
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true
      },
      "physics": {
        "enabled": true,
        "barnesHut": {
          "avoidOverlap": 1
        },
        "stabilization": { "iterations": 300 }
      },
      "nodes": {
        "margin": 12
      }
    }
    """)

    net.save_graph(str(output_path))

    settle_js = """
    <script>
      window.addEventListener("load", function() {
        network.on("stabilizationIterationsDone", function() {
          network.setOptions({ physics: { enabled: false } });
        });
      });
    </script>
    """

    # inject legend + settle script into the saved HTML
    html = output_path.read_text()
    html = html.replace("</body>", legend_html + settle_js + "\n</body>")
    output_path.write_text(html)

    if open_browser:
        webbrowser.open(output_path.resolve().as_uri())

    print(f"Saved: {output_path.resolve()}")
    return output_path
