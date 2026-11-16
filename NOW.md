# NOW — Text2SQL — 2026-11-16

Status: **METRIC_GATE / DATA_GATE**. Target semantic correctness; do not optimize undefined independent semantic metric. Base fixed to **Qwen3.5-9B**.

- [Verified: `runs/qwen9b-frozen100/summary.json`] Historical BIRD DEV100 set-EX-compatible **33→39/100**, guarded conservative execution success **75→90/100**, rescues 6, regressions 0. Original Actions run `37307108955` did not complete cleanly.
- [Verified: `docs/pro-research-gates.md`] 4-candidate consensus, cross-DB lexical two-shot, planning/review, value-grounding pilots were previously tested in specific settings; do not recycle as untested ideas.
- [Verified: Actions `43733784127`] Exact frozen grain-first Qwen prompt on BIRD DEV20: strict set EX **4/20→5/20**, 1 rescue on 1 DB, 0 strict set regressions, bag EX 4/20→4/20; failed preregistered cross-DB promotion gate; PR #61 closed unmerged.
- [Verified: Actions `43738683554`] Corrected native batched SQLite tool loop on fresh BIRD DEV12: strict set **3/12→3/12**, bag **3/12→3/12**, execution **11/12→12/12**, model calls **16→31**, 0 rescues. PR #63 closed unmerged. **Latest tested mechanism had no semantic EX gain.**
- [Verified: `docs/pro-research-gates.md`] Historical frozen100 reached all **11** physical BIRD DEV DBs. New same-source splits can only be question-disjoint.
- [Verified: repo tree] No independent user-defined semantic score, label agreement, or scoring-repeat variance documented.
- [Blocked] BIRD 1,534 dataset, official difficulty metadata and historical100 ID list not present as a validated local input. Dev500/holdout300 are specified, **not yet frozen**.
- [Owner override, 2026-11-16] Qwen-only; SQL QLoRA adapter on BIRD TRAIN permitted after decontamination and metric gate; stronger teacher only with separate consent.

**One next experiment:** CPU-only verification of the semantic rubric and source integrity (no Qwen calls) before accepting any new semantic score or holdout. See `IDEAS.md`. Do not reinterpret diagnostic rescoring as new model accuracy.
