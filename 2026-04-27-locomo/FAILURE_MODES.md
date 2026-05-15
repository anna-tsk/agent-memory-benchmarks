# Cat-5 failure-mode analysis

Categorizing every failure in the cat-5 no-MCQ runs (Hindsight and the
typed-relation graph backend) by the mechanism that caused the model to
produce an answer instead of abstaining. The taxonomy in the writeup
collapses what the data shows are three distinguishable modes.

## Taxonomy

Working from the 15 Hindsight failures + 2 graph failures, the failure
modes that actually appear:

- **Misattribution.** Retrieval surfaces a real claim about entity X.
  The question asks about entity Y. The model attributes the claim
  about X to Y. The asked-about entity and the answered-about entity
  don't match.
  *Hindsight examples:* HF-4 (Caroline's adoption plans → Melanie),
  HF-5 (Melanie's running → Caroline), HF-13 (Caroline's transgender
  poetry event → Melanie).

- **Bridge confabulation.** Retrieval surfaces two separately-true
  facts. The model merges them into a connection that was never made
  in the conversation. Both pieces are real; the bridge is invented.
  *Hindsight examples:* HF-1 (Melanie + dad + horseback riding all
  separately retrieved → "Melanie went horseback riding with her dad"),
  HF-15 (Melanie + adoption + LGBTQ+ separately retrieved → "Melanie's
  agency supports LGBTQ+ individuals").

- **Pure fabrication.** The model produces a specific detail (a song
  title, an instrument, a sidewalk color) that has no obvious anchor
  in any retrieved fact. Retrieval-induced confidence appears to be
  enough for the model to commit to invented specifics.
  *Hindsight examples:* HF-9 ("Caroline plays acoustic guitar"),
  HF-12 ("Sara Bareilles, 'Brave'"), HF-11 (specific religious-
  conservative encounter on a hike).

## Summary

| failure mode | Hindsight (15 fails / 47q) | Graph (2 fails / 47q) |
|---|---:|---:|
| Misattribution         | 7 | 1 |
| Bridge confabulation   | 2 | 0 |
| Pure fabrication       | 6 | 0 |
| (Likely LoCoMo labeling error) | 0 | 1 |
| **Total failures**     | **15** | **2** |
| **Correct abstentions** | 32 / 47 (68%) | 45 / 47 (96%) |

The single graph-backend misattribution is the same mechanism as
Hindsight's seven misattributions — just *much* rarer because entity
matching gates retrieval. The other failure mode that produces the
remaining graph "failure" appears to be a LoCoMo labeling discrepancy,
not an architectural problem: the model gave an answer grounded in
material the conversation does seem to contain (Melanie's painting
description, D9:16).

## Full per-failure table

### Hindsight cat-5 no-MCQ (15 failures of 47)

| # | Question | Model's prediction | Mode |
|---|---|---|---|
| HF-1 | What activity did Melanie used to do with her dad? | "Melanie used to go horseback riding with her dad." | Bridge confabulation |
| HF-2 | What does Caroline do to keep herself busy during her pottery break? | "Caroline uses painting and drawing to keep herself busy during her pottery break." | Pure fabrication |
| HF-3 | Did Caroline make the black and white bowl in the photo? | "No, Melanie made the black and white bowl in the photo." | Pure fabrication |
| HF-4 | What are Melanie's plans for the summer with respect to adoption? | "Melanie is considering adoption after hearing about her friend's success and may pursue it herself." | Misattribution (Caroline's plans → Melanie) |
| HF-5 | What does Caroline say running has been great for? | "Running has been great for Melanie's mental health." | Misattribution (Melanie's statement → Caroline) |
| HF-6 | What was grandpa's gift to Caroline? | "Caroline's grandmother, not her grandfather, gave her a special necklace that symbolizes love, faith, and strength." | Misattribution (grandmother → grandpa) |
| HF-7 | What did Melanie find in her neighborhood during her walk? | "During her walk, Melanie found a rainbow-painted sidewalk in her neighborhood." | Misattribution (Caroline's finding → Melanie) |
| HF-8 | What inspired Melanie's painting for the art show? | "Melanie's painting for the art show was inspired by the vibrant purple sunset..." | Pure fabrication |
| HF-9 | What type of instrument does Caroline play? | "Caroline plays the acoustic guitar." | Pure fabrication |
| HF-10 | How did Caroline's son handle the accident? | "Caroline's son was scared during the accident, but he was reassured and handled it well." | Misattribution (Melanie's son → Caroline's son) |
| HF-11 | What setback did Caroline face recently? | "Caroline faced a setback during a recent hike when she encountered a group of religious conservatives who said something that really upset her." | Pure fabrication |
| HF-12 | Who is Caroline a fan of in terms of modern music? | "Caroline is a fan of Sara Bareilles, specifically her song 'Brave'." | Pure fabrication |
| HF-13 | What was the poetry reading that Melanie attended about? | "The poetry reading Melanie attended was a transgender poetry reading where transgender people shared their stories through poetry..." | Misattribution (Caroline's event → Melanie) |
| HF-14 | What kind of place does Melanie want to create for people? | "Melanie wants to create a place of acceptance and love, where people can feel safe and supported, especially for the LGBTQ+ community." | Misattribution (Caroline's aspiration → Melanie) |
| HF-15 | What type of individuals does the adoption agency Melanie is considering support? | "The adoption agency Melanie is considering supports LGBTQ+ individuals." | Bridge confabulation |

### Graph backend cat-5 no-MCQ (2 failures of 47)

| # | Question | Model's prediction | Mode |
|---|---|---|---|
| GR-1 | What inspired Melanie's painting for the art show? | "Melanie's painting for the art show was inspired by her desire to convey a sense of tranquility and serenity..." (cites verbatim quote attributed to Melanie about peaceful blue streaks) | Likely LoCoMo labeling error — the answer is grounded in retrieved content the model can quote verbatim |
| GR-2 | What happened to Caroline's son on their road trip? | "During the road trip, Melanie's son got into an accident. It was a scary experience, but he was okay." | Misattribution (Melanie's son → Caroline's son) |

## Discussion

**The graph backend doesn't introduce new failure modes** — it just
makes the same modes much rarer. Of the three Hindsight modes,
misattribution still appears once in the graph runs (GR-2: Melanie's
son's accident attributed to Caroline's son). Bridge confabulation and
pure fabrication, the two modes most clearly produced by Hindsight
returning related-but-unconnected retrieval results, **do not appear at
all** in the graph backend's failures.

This is the architectural mechanism the writeup hypothesized, visible
in the failure distribution:

- *Pure fabrication* in Hindsight is what happens when similarity
  retrieval surfaces a few topically-adjacent facts and the model
  pads with plausible specifics. The graph backend returns
  `(no relevant claims retrieved)` when entity match fails, removing
  the seed material that the pure-fabrication mode pulls from.
- *Bridge confabulation* in Hindsight is what happens when two real
  facts about different entities co-occur in the retrieved set. The
  graph backend's entity-gated retrieval rarely surfaces facts about
  multiple unrelated entities together unless the question genuinely
  involves both.
- *Misattribution* survives in the graph backend (GR-2) because
  entity sets can overlap legitimately — Melanie has a son, Caroline
  has a son, and the question about Caroline's son retrieved facts
  about Melanie's son via shared "son" / "road trip" entities. This
  is the irreducible mode that better entity disambiguation could
  reduce further.

The 7 → 1 reduction in misattribution and the elimination of the other
two modes are why the graph backend's spontaneous abstention rate
(96%) is so much higher than Hindsight's (68%) on the same questions.
