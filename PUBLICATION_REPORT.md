# Publication-readiness report

*12 November 2026 · Prepublication checks; the scoped package is authorized for commit and push.*

**The README, canonical research note and Caroline walkthrough are ready to commit together with the selected `publication/` package.** Their links close over that package and three already tracked implementation files. This is a presentation and evidence-extract release, not a complete reproducibility release. The listed files must be committed and pushed together for their links to resolve on GitHub.

## Files prepared

| File | Change |
|---|---|
| [README.md](README.md) | 745-word landing page; provisional November results, readable case example, scope and artifact links. |
| [Research note](DISTINGUISHING_ABSTENTION_FROM_EVIDENCE_LOSS.md) | Canonical version 3; verified related work and prior LoCoMo reports; causal/novelty claims narrowed; original numerical tables retained. |
| [Caroline case study](case_studies/caroline_adoption_agency.md) | Source-to-answer chain with exact question, claim ID, matching failure, recorded context omission and exact response. |
| [Selected artifacts](publication/README.md) | Saved May/November results, score/flag extract, sensitivity calculations, retrieval summary, case trace, rubric/amendment, settings and original hashes. |
| [Offline checker](publication/check_records.py) and [manifest](publication/MANIFEST.json) | Integrity, arithmetic and recorded-identifier checks; no model calls or retrieval replay. The manifest lists the exact proposed commit files and existing code dependencies. |
| [.gitignore](.gitignore) | Excludes credentials, tooling state, previews and the unreleased research workspace from accidental additions. It does not remove already tracked files. |

The original canonical draft was first copied byte-for-byte to `DISTINGUISHING_ABSTENTION_FROM_EVIDENCE_LOSS_v1.md`. Versions 2 and the prior version-3 review copy remain unchanged. Historical audits, source logs, graph implementation and all frozen experiment records are untouched.

## Verification

- All three research-note tables and original source-hash literals are unchanged. All 484 run, analysis and diagnostic manifest entries still match.
- Markdown parses with the expected headings and tables; no unintended escaping. The proposed files have no unresolved relative links. Citation titles, author lists and dates were checked against primary records; all 13 external link destinations returned HTTP 200.
- The Caroline response matches the original provider-response body exactly. Its final-context omission is explicitly attributed to the frozen diagnostic, not claimed as newly reproduced from the selected extract.
- The offline checker passes in an isolated directory containing only the proposed package and listed tracked code dependencies. Its optional dataset check also passes against the existing pinned data. No fresh dataset clone/download or model experiment was run; full historical replay remains untested from a public checkout.
- HTML previews were generated locally. Browser visual inspection remains unavailable because the browser-access host failed to start; no visual-browser pass is claimed.

## Exclusions and remaining gaps

Raw API logs/prompts, complete datasets/conversations, graph pickles, full source-validation packets, the complete AI-scoring submission and locally patched Hindsight files are intentionally excluded. Their identifiers/hashes and availability limits are recorded, with no broken public links to them. The original May PDF remains missing. These omissions prevent full historical reproduction but do not prevent publishing the accurately scoped case study and selected records.

Credential-pattern, email and private-path scans found no matches in the proposed package. This is a heuristic content check, not a security certification. The existing tracked tree contains machine-specific paths in **21 grid-run log files** under `2026-04-27-locomo/logs/grid_run_20260517_235151/` and in `2026-04-27-locomo/run_conv_full_grid.sh`. Those 22 historical files were not modified or removed. They are outside the new documentation package; this pass does not certify the entire existing repository/history as sanitized. Review their exposure before any broader public release, without silently changing frozen provenance.

LoCoMo's inspected repository is CC BY-NC 4.0; Hindsight's is MIT. The case includes attributed brief LoCoMo excerpts; no whole dataset or upstream Hindsight code is newly bundled. The repository has no general license for its own code/writing, so unrestricted reuse is not claimed. License links and dataset acquisition instructions are in the artifact guide.

No new experiment, model call, human adjudication, ablation, graph repair or history rewrite was performed. Commit the explicitly listed files together rather than using a blanket workspace addition. The commit and push of this scoped package are authorized; any separate cleanup of pre-existing historical paths remains outside this publication step.
