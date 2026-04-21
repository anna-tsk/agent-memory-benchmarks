# Initial Probe Notes

Date: 2026-04-16

These notes summarize the first exploratory pass on belief revision behavior in a raw instruct model. This was not yet a formal benchmark. The goal was to observe how the model resolves sequential claims when no explicit memory mechanism is implemented.

## Goal

We wanted to test what Qwen2.5-3B-Instruct does when given small timestamped contexts such as:

```text
2026-01-01 | Anna: I live in LA.
2026-02-15 | Anna: I live in SF.
```

The key question was whether the model would:

- treat the later statement as replacing the earlier one
- preserve both statements
- recognize ambiguity
- ask for clarification
- over-collapse the situation into one current belief

This was motivated by a ChatGPT interaction where the model initially answered that Anna lived in San Francisco, then later acknowledged that this answer may have collapsed ambiguity if Anna could live in both LA and SF.

## Repository Setup

We created two related but distinct tracks.

The first track is a benchmark scaffold:

- `data/schema/benchmark.schema.json`
- `data/examples/synthetic_trajectories.json`
- `scripts/run_context_only_qwen.py`

This track still has structured fields such as entities, initial facts, updates, metadata, evaluation queries, and expected answers. It is useful for eventual scoring.

The second track is exploratory:

- `data/examples/raw_observation_probes.json`
- `scripts/probe_raw_context_qwen.py`

This track deliberately avoids expected answers and explicit update labels. It stores only raw observations and questions. This is the track used for the more interesting qualitative probes.

## Model

Model:

```text
Qwen/Qwen2.5-3B-Instruct
```

The model was run locally through Hugging Face `transformers`, using Qwen's chat template. Generation was deterministic:

```python
do_sample=False
```

The system prompt was intentionally minimal:

```text
You answer questions.
```

This was important because we did not want to strongly instruct the model to prefer recency, ambiguity, or clarification.

## Context-Only Benchmark Probe

The first runner, `scripts/run_context_only_qwen.py`, serializes structured benchmark trajectories into text. Initially the rendered context included clues such as:

```text
revise
Supersedes: fact_anna_lives_la
```

We removed the `Supersedes` line from model-facing context because it directly told the model which fact replaced which earlier fact. The JSON still keeps `supersedes` for future evaluator use, but the model no longer sees it in the prompt.

Current rendered update example:

```text
Updates:
- At 2026-02-15T12:00:00Z [personal], revise: Anna lives_in San Francisco.
```

This is still scaffolded because it includes `revise`, but it is less explicit than before.

## Raw Observation Probe

The second runner, `scripts/probe_raw_context_qwen.py`, was added to observe model behavior without scoring.

Raw probe example:

```json
{
  "id": "probe_anna_la_sf",
  "title": "Anna Says LA Then SF",
  "observations": [
    {
      "timestamp": "2026-01-01T09:00:00Z",
      "speaker": "Anna",
      "content": "I live in LA."
    },
    {
      "timestamp": "2026-02-15T12:00:00Z",
      "speaker": "Anna",
      "content": "I live in SF."
    }
  ],
  "questions": [
    "Where does Anna live?",
    "Why do you say that?",
    "Is it possible Anna lives in both places?"
  ]
}
```

The important design choice here is that the context does not say:

- moved
- revised
- corrected
- superseded
- expected answer
- expected behavior

It only provides timestamped statements.

## Independent vs Dialogue Mode

At first, each question was asked independently. That meant a question like:

```text
Why do you say that?
```

was ambiguous, because the model did not see its own previous answer.

We added `--dialogue-mode`, where each model answer is appended to the chat history before the next question is asked.

Independent mode:

```text
Context + Q1
Context + Q2
Context + Q3
```

Dialogue mode:

```text
Context + Q1 + A1 + Q2 + A2 + Q3
```

This lets later questions refer to previous answers.

## Observed Behavior

For the raw LA/SF probe in independent mode, the model gave a nuanced first answer. It said that the current residence was not definitive, but that the most recent observation suggested San Francisco.

Qualitative pattern:

```text
latest statement is treated as strongest evidence
but earlier conflicting statement still creates uncertainty
```

However, when asked whether Anna could live in both places, the model initially over-collapsed the situation and said that living in both was not possible or that the statements were contradictory.

That is an important failure mode:

```text
The model treats two residence claims as mutually exclusive without direct evidence that they are mutually exclusive.
```

In dialogue mode, the model became more explicit about a transition. It used language like:

```text
change over time
relocation
shift in stated residence
clear transition
most recent location
```

This suggests that once the model has produced an interpretation, subsequent answers may reinforce that interpretation rather than reopen the ambiguity.

## Preliminary Finding

The raw instruct model appears to use a recency heuristic for current-belief questions:

```text
If Anna first says LA and later says SF, the later SF claim is treated as the best current answer.
```

But the model is inconsistent about uncertainty:

- It sometimes acknowledges that the current answer is not definitive.
- It sometimes treats the later statement as evidence of relocation.
- It sometimes says simultaneous residence is not indicated.
- It may overstate contradiction or mutual exclusivity.

This is exactly the behavior we want to study. The model is not merely storing facts; it is resolving relations among facts, often using unstated conversational assumptions.

## Research Implication

The initial structured benchmark over-defined the problem by labeling later information as an `update` or `revise`. That is useful for a later scoring pipeline, but it may hide the core research question:

```text
How should a memory substrate decide whether a new claim replaces, adds to, contradicts, or leaves ambiguous an older claim?
```

The more interesting representation may separate:

- raw observations: what was actually said
- extracted propositions: optional structured facts
- inferred relations: add, revise, correct, conflict, ambiguous
- clarification behavior: whether the system should ask a follow-up

For now, the raw observation probes are better suited to studying the model's natural assumptions.

## Next Steps

1. Add more ambiguous probe pairs:
   - Anna likes coffee / Anna likes tea
   - Anna works at Nova / Anna works at Orbit
   - Anna's office is Room 204 / Anna's office is Room 318
   - Anna has a cat named Mochi / Actually, Miso

2. Compare independent mode and dialogue mode.

3. Increase `--max-new-tokens` when answers are cut off:

```bash
python scripts/probe_raw_context_qwen.py \
  --limit 1 \
  --dialogue-mode \
  --max-new-tokens 256
```

4. Add a logging mode that saves outputs to JSONL for later qualitative coding.

5. Eventually define labels for analysis, not necessarily for model prompting:
   - recency collapse
   - ambiguity preserved
   - asks clarification
   - additive interpretation
   - replacement interpretation
   - contradiction interpretation

These labels can help analyze behavior without forcing the model to see the labels during the probe.
