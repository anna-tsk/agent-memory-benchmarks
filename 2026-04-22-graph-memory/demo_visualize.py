"""Quick demo: build a graph by hand and open the visualizer."""
from datetime import datetime
from graph import ClaimLinkType, MemoryGraph
from visualize import render

g = MemoryGraph()

anna = g.add_entity("Anna", "a person")
tea  = g.add_entity("tea", "a drink")
coffee = g.add_entity("coffee", "a drink")
la   = g.add_entity("Los Angeles", "a city")

t = datetime(2026, 4, 22, 12, 0)

c1 = g.add_claim("Anna's favorite drink is tea.", "user", t, [anna.id, tea.id], [])
c2 = g.add_claim("Anna's favorite drink is coffee.", "user", t, [anna.id, coffee.id], [])
c3 = g.add_claim("Anna lives in Los Angeles.", "user", t, [anna.id, la.id], [])

rl1 = g.add_relation_link(anna.id, tea.id, "favorite drink is", c1.id)
rl2 = g.add_relation_link(anna.id, coffee.id, "favorite drink is", c2.id)
rl3 = g.add_relation_link(anna.id, la.id, "lives in", c3.id)

c1.relation_link_ids = [rl1.id]
c2.relation_link_ids = [rl2.id]
c3.relation_link_ids = [rl3.id]

g.add_claim_link(c1.id, c2.id, ClaimLinkType.CONFLICTS)

render(g, "demo_graph.html")
