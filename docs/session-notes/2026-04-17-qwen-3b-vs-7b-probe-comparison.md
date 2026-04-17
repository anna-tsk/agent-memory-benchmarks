# Qwen2.5 3B vs 7B Probe Comparison

Date: 2026-04-17

This note compares the same-speaker 3B full run with the 7B full run:

- `runs/all_probes_qwen25_3b_same_speaker_all_conditions_20260417_153142.txt`
- `runs/all_probes_qwen25_7b_all_conditions_with_prompts_20260417_150608.txt`

Both runs use the current context rendering:

```text
All observations are statements from the same speaker.

Observations:
- timestamp | claim
- timestamp | claim
```

This removes the earlier `Speaker:` rendering artifact where Qwen sometimes interpreted names inside the statements as different speakers.

## High-Level Result

Qwen2.5-7B-Instruct is meaningfully better than Qwen2.5-3B-Instruct, but not simply correct.

The main difference is:

```text
3B often fails at basic relation parsing.
7B usually parses the observations correctly, but still chooses questionable relation policies.
```

So the failure shifts from:

```text
low-level confusion / contradiction / wrong entity tracking
```

to:

```text
higher-level ontology choice: replacement vs coexistence vs ambiguity
```

That shift is useful. The 7B model is closer to the actual research question.

## Probe-by-Probe Comparison

| Probe | 3B Pattern | 7B Pattern | Takeaway |
|---|---|---|---|
| Best Friend | Minimal says contradiction. Analytic allows two best friends. Conversational hedges. | Stronger recency: often says Tina is current best friend, while acknowledging possible contexts. | 7B is more temporally decisive, but maybe too eager to overwrite social looseness. |
| Hometown | Allows both or says unclear. More cautious. | Minimal says typically one hometown, but analytic/conversational allow two hometowns. | 7B shows better category flexibility under structured prompts, but default still has singular bias. |
| Home | 3B often chooses New York and invents move/abandonment. | Minimal allows multiple homes. Analytic/conversational choose New York as current. | 7B is better in minimal but still recency-biased when scaffolded. |
| Favorite Person | 3B treats it as likely change over time. | 7B also treats it as change over time. | Both models impose temporal revision on expressive looseness. |
| Boss | 3B minimal says contradiction; structured run can be confused. | 7B says Priya current, but notes possible overlap. | 7B handles timestamps better but still assumes replacement. |
| Relationship Status | 3B says “alternates” / cannot determine in minimal. | 7B cleanly says married as current. | 7B clearly better on strong temporal-change cases. |
| Age | 3B says 30 cleanly. | 7B minimal says 29 or 30 depending on birthday; structured says 30. | 7B is more cautious, arguably overcomplicating minimal. |
| Name Preference | 3B minimal says use both / ask. Some structured answers are internally wrong. | 7B consistently says Anya based on most recent preference. | 7B much better for preference updates. |
| Diet | Both say not strictly vegetarian / pescatarian-ish. | 7B says not strictly vegetarian anymore; adds possible diet change. | Both correct the label rather than preserving self-description tension. |
| Sister | 3B minimal falsely says contradiction. | 7B minimal says both Maya and Leila are sisters. | Major 7B improvement; clean false-singularity fix. |
| Mother | 3B says contradiction / recency. | 7B also overwrites to Lina. | Both are too aggressive; this is a harder social/family-role case. |
| First Language | 3B says contradiction and unclear. | 7B minimal says contradiction/unclear, but structured picks Russian. | Both treat first language as strongly singular; 7B more willing to pick latest. |
| Favorite City | 3B says change to Tbilisi. | 7B says change/alternation depending condition. | Both temporalize expressive preference. |
| Doctor | 3B says change but oddly says only Chen is definitive in minimal. | 7B says Dr. Alvarez current and overwrites Chen. | 7B is more coherent, but still assumes replacement where coexistence is possible. |

## Condition Effects

The three prompt conditions still matter for both models.

### Minimal

For 3B, minimal often exposes brittle failures:

```text
best friend -> contradiction
sister -> contradiction
relationship status -> alternates / unclear
```

For 7B, minimal is much stronger:

```text
sister -> both sisters
home -> multiple homes
relationship status -> married
name preference -> Anya
```

But 7B minimal still imposes singularity in some categories:

```text
hometown -> typically one hometown
first language -> contradiction
mother -> Lina overrides Sara
doctor -> Dr. Alvarez overrides Dr. Chen
```

### Analytic

Analytic scaffolding often improves multiplicity awareness, but can also introduce recency/update language.

For 7B:

```text
hometown -> two hometowns
sister -> at least two sisters
home -> New York current
best friend -> Tina current
```

So analytic does not simply mean “more nuanced.” It changes the ontology differently by domain.

### Conversational

Conversational remains the least stable. It often pushes toward:

```text
clarify why it changed
assume latest is current
tell a change story
```

7B’s conversational answers are less nonsensical than 3B’s, but still often assume revision.

## Model-Size Findings

### 1. 7B Fixes Many Basic Parsing Failures

The sister probe is the cleanest example:

```text
3B: Maya and Leila contradict each other.
7B: both Maya and Leila are sisters.
```

This is a real capability difference.

### 2. 7B Is More Coherent With Timestamps

For relationship status, name preference, boss, and doctor, 7B uses the later timestamp more consistently. 3B sometimes mishandles chronology or contradicts itself.

### 3. 7B Is More Decisive, Sometimes Too Decisive

7B often makes a stronger current-state judgment:

```text
Tina is best friend
New York is home
Priya is boss
Lina is mother
Russian is first language
Dr. Alvarez is doctor
```

That improves update-like cases but hurts coexistence/ambiguity cases.

### 4. The Hard Problem Remains

Even 7B does not robustly distinguish:

```text
single-slot attributes: age, relationship status
plural-compatible roles: sister, doctor, boss
expressive looseness: home, favorite person, favorite city
identity/category ambiguity: hometown, first language, vegetarian
```

It handles some correctly but not by a stable general rule.

## Interpretation

The 3B model was partly too weak for the main claims. It produced low-level errors that are not the real target.

The 7B model moves the experiment closer to the actual research problem:

```text
The model can parse the observations, but still has unstable claim-to-belief interpretation.
```

The emerging story:

```text
Scaling from 3B to 7B improves basic discourse coherence and reduces nonsense.
But it does not eliminate prompt-sensitive ontology selection.
```

More directly:

```text
7B knows that people can have two sisters.
But it still does not reliably know when two sequential claims should coexist, revise, or remain underdetermined.
```

## Useful Focus Probes

For a compact follow-up analysis, focus on:

1. Sister: clean size effect. 3B fails, 7B fixes.
2. Best Friend: 7B remains recency-biased despite social looseness.
3. Home: minimal vs structured flips the model’s ontology.
4. Mother: both models overwrite too aggressively despite possible plural family structures.
5. Doctor: both models assume replacement, but plural doctors are normal.
6. Diet: both models correct the speaker’s label instead of modeling self-description tension.
7. First Language: strong singular-category assumption.

This subset is enough to show:

```text
some failures are model-size limitations
some failures persist as ontology-selection problems
prompt framing continues to matter even at 7B
```
