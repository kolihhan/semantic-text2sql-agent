# Lessons — Text2SQL

1. [Verified: `runs/qwen9b-frozen100/summary.json`] Executable SQL is not semantically correct. Conservative guarded execution succeeds on 90/100, yet set-EX matches 39/100.
2. [Verified: `docs/pro-research-gates.md`] The *tested* 4-candidate consensus had no oracle headroom (16/20), so resampling N alone has not shown a rescue mechanism; this is not a rejection of all search.
3. [Verified: `docs/pro-research-gates.md`] Tested cross-DB 2-shot lexical retrieval 8/20→7/20; tested Spider value grounding 9/12→9/12; tested Qwen answer-grain planning 9/12→9/12 with higher cost. Small n and source shift preclude blanket rejection of their larger hypothesis classes.
4. [Verified: `docs/pro-research-gates.md`] Some source reference labels conflict with physical SQLite data (collation/case) and repeated question families are dependent. Preserve official labels, flag anomalies, do not post-hoc rewrite scores.
5. [Verified: `docs/pro-research-gates.md`] Old frozen100 covers every BIRD DEV physical DB, preventing **new-database** generalization claims within BIRD DEV.
6. [Inference] Blindly repeating semantic prompts on heavily seen DEV questions is more likely to optimize exposure than transfer.
7. [Unresolved] Independent semantic score has no agreed rubric, scoring script, human agreement or repeat variance. It is **not interchangeable** with the existing set-EX or lexical grounding metrics.

Research decision audit: earlier “REJECT” labels in PRO notes apply to the **specific tested variants**; if their paired sample is beneath the owner-approved statistical gate, translate to “not promoted / insufficient evidence,” not universal elimination. Audit every five experiment rounds.
