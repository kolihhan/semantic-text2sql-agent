# Lessons — Text2SQL

1. [Verified: `runs/qwen9b-frozen100/summary.json`] Executable SQL is not semantically correct. Conservative guarded execution succeeds on 90/100, yet set-EX matches 39/100.
2. [Verified: `docs/pro-research-gates.md`] The *tested* 4-candidate consensus had no oracle headroom (16/20), so resampling N alone has not shown a rescue mechanism; this is not a rejection of all search.
3. [Verified: `docs/pro-research-gates.md`] Tested cross-DB 2-shot lexical retrieval 8/20→7/20; tested Spider value grounding 9/12→9/12; tested Qwen answer-grain planning 9/12→9/12 with higher cost. Small n and source shift preclude blanket rejection of their larger hypothesis classes.
4. [Verified: `docs/pro-research-gates.md`] Some source reference labels conflict with physical SQLite data (collation/case) and repeated question families are dependent. Preserve official labels, flag anomalies, do not post-hoc rewrite scores.
5. [Verified: `docs/pro-research-gates.md`] Old frozen100 covers every BIRD DEV physical DB, preventing **new-database** generalization claims within BIRD DEV.
6. [Inference] Blindly repeating semantic prompts on heavily seen DEV questions is more likely to optimize exposure than transfer.
7. [Unresolved] Independent semantic score has no agreed rubric, scoring script, human agreement or repeat variance. It is **not interchangeable** with the existing set-EX or lexical grounding metrics.

Research decision audit: earlier “REJECT” labels in PRO notes apply to the **specific tested variants**; if their paired sample is beneath the owner-approved statistical gate, translate to “not promoted / insufficient evidence,” not universal elimination. Audit every five experiment rounds.

8. [Verified: Actions `43833618248` + recovered historical cohort manifests] The first prospective 500/300 ID-only split had **59 known previously used cases in DEV and 23 in HOLDOUT**. Excluding only the original frozen100 did **not** ensure independent questions: cross-branch GitHub Actions artifacts and hard-coded prior pilot cohorts matter. V1 is quarantined, not silently deleted.
9. [Verified: Actions `43834589773`, `manifests/bird-dev500-holdout300-20261116-v2.json`] Audit of seven recovered extra experimental cohorts plus prior BIRD16/24 recovered **126 additional unique exposed case IDs** beyond the original100. Re-freezing V2 after excluding all 126 plus two duplicated question rows produced DEV500/holdout300 with **0 known overlap**. Completeness of historical exposures cannot be guaranteed, and all 11 BIRD DEV DBs remain previously exposed. V2 is not a source-independent generalization test.
