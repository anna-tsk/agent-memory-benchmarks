# Session Notes: 2026-04-17

These notes summarize the work from the session that began with the probe-design discussion around 9:59am.

## Starting Point

The day started with a proposed set of probes built around ordinary underspecified language:

- normal human language is genuinely underspecified
- exclusivity is unclear
- the model must choose between collapse, ambiguity preservation, coexistence, temporal revision, or clarification

We agreed this was a better direction than artificial contradiction probes. The goal shifted from asking whether the model can apply explicit updates to asking what ontology the model imposes on sequential natural-language claims.

## Probe Set V2

We created and refined:

```text
data/examples/raw_observation_probes_v2.json
```

The v2 probe set currently contains 14 probes:

- Best Friend
- Hometown
- Home
- Favorite Person
- Boss
- Relationship Status
- Age
- Name Preference
- Diet
- Sister
- Mother
- First Language
- Favorite City
- Doctor

We removed the Christian/religion probe because it was too obvious and over-signaled temporal revision.

We also changed the Home probe from:

```text
LA is home.
New York is home now.
```

to:

```text
LA is home.
New York is home.
```

The word `now` was removed because it strongly cued temporal revision. The probe is now categorized as expressive looseness.

## Bucket Cleanup

The initial bucket labels were too specific and noisy. We simplified them to four analysis-only categories:

```text
false_singularity
expressive_looseness
temporal_change
category_boundary
```

These bucket labels are printed in logs but are not shown to the model.

## Prompt Conditions

We found that output format changes substantially alter model behavior. In particular:

- `Clarifying question` pushes the model toward conversational repair and sometimes weird narrative assumptions.
- `Missing information or useful follow-up` pushes the model toward a more analytic inventory of missing evidence.
- Minimal prompting can reveal a different default policy than either structured prompt.

We therefore added explicit prompt framing conditions to:

```text
scripts/probe_raw_context_qwen.py
```

The supported conditions are:

```text
minimal
analytic
conversational
all
```

The conditions mean:

```text
minimal:
  ask only the question

analytic:
  Direct answer
  Possible interpretations
  Assumption(s) made
  Missing information or useful follow-up
  Memory handling

conversational:
  Direct answer
  Possible interpretations
  Assumption(s) made
  Clarifying question
  Memory handling

all:
  run minimal, analytic, and conversational sequentially
```

## Prompt Visibility

We discovered a hidden prompt-wrapper issue: the script had inserted language like:

```text
If the observations are underspecified, preserve that uncertainty.
```

This was not in the JSON file, so it was easy to miss. It was removed.

To prevent this kind of confusion going forward, we added:

```text
--show-prompt
```

This prints the exact chat messages passed to the tokenizer.

## Logging

We added:

```text
--log-name
```

This writes a terminal-style log to:

```text
runs/<name>_<timestamp>.txt
```

The logs include settings, context, optional prompt messages, questions, conditions, and model outputs.

Example:

```bash
python scripts/probe_raw_context_qwen.py \
  --probe-id probe_sister \
  --condition all \
  --show-context \
  --show-prompt \
  --max-new-tokens 768 \
  --log-name sister_same_speaker_all_conditions
```

## Probe Filtering

We added:

```text
--probe-id
```

This lets us rerun one or more specific probes without relying on file order.

Example:

```bash
python scripts/probe_raw_context_qwen.py \
  --probe-id probe_sister \
  --condition all \
  --log-name sister_check
```

Multiple IDs are supported:

```bash
--probe-id probe_sister --probe-id probe_doctor
```

## Same-Speaker Rendering Fix

The first full run revealed a major probe-format artifact: the model sometimes interpreted the observations as coming from different speakers.

For example, in the sister probe it produced nonsensical interpretations like:

```text
"Maya is my sister" was made by a speaker who refers to themselves as Maya.
```

To control this, we changed context rendering from:

```text
Speaker: Maya is my sister.
Speaker: Leila is my sister.
```

to:

```text
All observations are statements from the same speaker.

Observations:
- 2026-01-01T09:00:00Z | Maya is my sister.
- 2026-02-15T12:00:00Z | Leila is my sister.
```

This makes source identity explicit while still leaving the relation between claims open.

## Key Empirical Observations

The strongest finding so far is that the model does not have a stable relation policy. The same observations yield different ontologies depending on the prompt framing.

In the Best Friend probe:

```text
Marie is my best friend.
Tina is my best friend.
```

the conditions produced different interpretations:

- minimal: treats the issue as uncertainty over a singular slot
- analytic: allows plural coexistence more readily
- conversational: tends toward temporalization and follow-up behavior

This led to a sharper interpretation:

```text
The model is not merely deciding whether to collapse or preserve ambiguity.
It is choosing an ontology for the claim relation.
```

For example:

```text
singular slot:
  there is one best_friend value, but we do not know which

plural coexistence:
  both Marie and Tina can be best friends

temporalized singular slot:
  Marie may have been best friend before; Tina may be best friend now
```

The Sister probe after the same-speaker fix was especially revealing:

```text
Maya is my sister.
Leila is my sister.
```

The minimal condition still said the statements contradict each other. That is a clear false singularity failure: a person can have more than one sister.

The analytic and conversational conditions recovered plural coexistence more often, but still included unnecessary uncertainty or unnecessary clarification.

## Interpretation

Qwen2.5-3B-Instruct appears weak for this task. It often:

- contradicts itself within a structured answer
- overuses recency
- invents speaker/source ambiguity
- treats plural-compatible relations as singular slots
- turns ordinary looseness into contradiction
- asks clarification questions about information already present
- corrects the speaker's category labels, e.g. vegetarian -> pescatarian

However, this does not make the model useless. It is useful as a small-model failure microscope. It exaggerates the exact problems we are trying to characterize.

The important emerging claim is:

```text
Persistent memory systems need not only storage and retrieval.
They need stable claim-to-belief interpretation.
```

Or more specifically:

```text
The bottleneck is not only remembering facts.
It is choosing the ontology under which sequential claims become memory.
```

## Next Model Step

We decided to try a stronger Qwen model next while keeping the family fixed:

```text
Qwen/Qwen2.5-7B-Instruct
```

Example sister run:

```bash
python scripts/probe_raw_context_qwen.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --probe-id probe_sister \
  --condition all \
  --show-context \
  --show-prompt \
  --max-new-tokens 768 \
  --log-name sister_qwen25_7b_all_conditions
```

The point is to distinguish small-model incompetence from a more general prompt-sensitive ontology-selection problem.
