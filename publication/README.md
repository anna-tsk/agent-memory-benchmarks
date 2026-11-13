# Selected evidence and reproducibility scope

This is a small publication package drawn from the frozen May audit and November experiment. It contains no new answers, judgments or retrieval run. Original artifacts remain unchanged in the local research workspace. Historical paths inside JSON are provenance identifiers, not promises that the corresponding files are included here.

## Included records

| File | What it provides |
|---|---|
| [may_results.json](may_results.json) | Saved no-MCQ per-conversation counts, macro averages and pooled scores; original source identifiers retained. |
| [november_scores.json](november_scores.json) | 48 projected score/flag records, question/pair identifiers and recorded coverage classes; most answers and source passages omitted. |
| [november_results.json](november_results.json) | Unchanged saved aggregate results, including per-conversation and pair statistics. |
| [scoring_sensitivity.json](scoring_sensitivity.json) | Saved single-flag flips and extreme scenarios; reviewer concern prose omitted. These are mechanical scenarios, not adjudicated alternatives. |
| [retrieval_summary.json](retrieval_summary.json) | Unchanged frozen diagnostic summary, including the 16-turn accounting and sent-prompt checks. |
| [caroline_trace.json](caroline_trace.json) | Selected source passage, cached claim and entity fields, full candidate/source-ID lists, recorded context checks and exact R011 answer. Full prompt text omitted. |
| [execution_settings.json](execution_settings.json) | Selected unchanged model settings, common prompts and retrieval rules from the execution protocol; operational authorization and pricing fields omitted. |
| [SCORING_RUBRIC.md](SCORING_RUBRIC.md) | Byte-identical historical rubric. Its original human-review plan is superseded by the amendment below. |
| [SCORING_AMENDMENT.md](SCORING_AMENDMENT.md) | Byte-identical record of the change to provisional AI-only scoring and preserved scoring tensions. Referenced original submissions remain local. |
| [provenance.json](provenance.json) | Original file identifiers and hashes, including excluded sources. A hash alone does not provide or independently validate a source. |
| [MANIFEST.json](MANIFEST.json) | Integrity hashes for the proposed presentation package. |

## Checks that run without model services

Use Python 3.9 or later; the checker uses only the standard library:

```sh
python3 publication/check_records.py
```

The [checker](check_records.py) reads only included files. It checks their hashes, recomputes saved count totals from the score extract, checks the declared pair counts, confirms the recorded 16-turn accounting, and checks the Caroline entity-name mismatch and omission from recorded candidate/rendered IDs. It neither imports the graph/extractor nor changes files, accesses the network or executes a model.

Passing these checks establishes consistency of this package, not correctness of the AI judgments, validity of the source labels, authenticity independent of the original logs, or full retrieval reproducibility. Equal hashes can establish identity with a supplied file, not the truth of that file's claims.

## Original dataset

Obtain LoCoMo from its [official repository](https://github.com/snap-research/locomo), subject to its [CC BY-NC 4.0 license at the inspected revision](https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/LICENSE.txt). The experiment used `data/locomo10.json` from commit `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`:

```sh
git clone https://github.com/snap-research/locomo /path/to/locomo
git -C /path/to/locomo checkout 3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376
python3 publication/check_records.py --locomo /path/to/locomo/data/locomo10.json
```

The optional argument verifies SHA-256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4` and compares the Caroline question and source turn with the excerpt. The acquisition commands are instructions, not an experiment run. The dataset file, images and complete conversations are not redistributed in this package. The question and brief D2:12 quotation are attributed, unchanged excerpts; the local entity/trace annotations are additions.

## What remains unavailable here

Complete source-validation packets, all answers, raw API requests/responses, graph caches, full retrieval traces, original ingestion logs and the locally patched Hindsight runner/results are excluded. Reconstructing November retrieval would require those frozen caches, query outputs, rendering code and source mapping; rerunning answer generation would additionally require model services and would create a new run. Exact remote serving revisions were not recorded. The original May PDF is missing from the local workspace.

Historical graph code remains in the dated directories. The November experiment repaired chronology and ordering through its frozen preparation/rendering helper; the May code is not a drop-in reproduction of that experiment. The complete historical audit is therefore not reproducible from this package alone.

## Attribution and code provenance

LoCoMo is by Maharana, Lee, Tulyakov, Bansal, Barbieri and Fang ([ACL 2024](https://aclanthology.org/2024.acl-long.747/)). The graph representation, extractor, local QA harness and publication checker are project code. The locally adapted Hindsight benchmark is separately identified; upstream Hindsight is [Vectorize's implementation](https://github.com/vectorize-io/hindsight/blob/89c34c31357ecd80e2ab2ed3244d9e37515cb5f8/README.md), under the [MIT license](https://github.com/vectorize-io/hindsight/blob/89c34c31357ecd80e2ab2ed3244d9e37515cb5f8/LICENSE). Its code is linked, not copied into this package. Recorded prompts preserve the experiment's wording; this package makes no claim of prompt originality.

No new blanket license is assigned to this repository's own material. Upstream terms and attribution remain applicable to the cited excerpts and software; publication does not imply unrestricted reuse.
