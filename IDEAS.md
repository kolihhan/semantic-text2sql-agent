# Ranked ideas — Text2SQL — 2026-11-16

## Selected: M0 — metric reliability and frozen population (CPU, no model calls)

- **Hypothesis:** current semantic/EX confusion and repeated DEV exposure can make apparent gains unreliable.
- **Predeclared outcome:** accept only an explicit semantic rubric + evaluator implementation + blinded double annotation agreement + repeated-score variance, and an immutable input/SHA-backed question-disjoint BIRD500/300 manifest with original 100 excluded. If inputs missing: **BLOCKED / insufficient evidence**, never pass by skipping.
- **Cost estimate:** 0 GPU-hours, 0 model calls; human annotation time needed for any subjective semantic rubric (estimate only after rubric agreed).
- **Result:** not run; no metric promotion.

## M1 — failure taxonomy on *already exposed* frozen100

Classify 52 executed-but-wrong Guarded SQL cases into schema linking / value / evidence misuse / join-or-aggregation / shape / execution or cap, with multi-label + primary root-cause, two human annotators on a sample. Diagnostic only, no new score or tuning. No holdout consumption.

## M2 — fresh Qwen-only paired experimental leverage

Only after M0, select one evidence mechanism with a documented *potential right candidate* and test as a single-factor ablation: e.g. non-gold retrieved BIRD TRAIN examples **vs** original scaffold. Note: previous lexical 2-shot retrieval was already tested and negative on n=20; any new mechanism must be demonstrably different. Then paired dev and source-independent validation, with approved sample gate.

## M3 — QLoRA adapter factorial

BIRD TRAIN-only, contamination-audited rejection-sampled trajectories; four arms (base / prompt / adapter / combined). Compare EX and **approved** semantic metric, per-case rescues/regressions, 95% paired CIs, and regression suite. No teacher model without prior owner consent.
