# CPU-only original Qwen9B TRAIN96 conservative candidate selector — negative, 2026-11-19

**Immutable actual execution:** [GitHub Actions #44488125995](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44488125995) succeeded in both `source-and-seal` and `original-physical-score`. Reused only [original run #44476207751](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44476207751) frozen source, 12 sealed original SQL-generation artifacts, and 12 physically scored candidate artifacts. No new Qwen calls, no train or protected BIRD DEV500/HOLDOUT300.

**Gold-blind separation:** first job receives only question, original publisher evidence (not gold), SQL text of `baseline_first_SQL` and `treatment_final_SQL`. It freezes choices and uploads artifact `qwen9b-role96-selector-GOLD-FREE-decisions` **before** second job downloads any physical EX labels. Second job publishes `qwen9b-role96-conservative-selector-CPU-FINAL`.

| Fixed 88 scorable / 96 frozen | Original first direct SQL | Constraint-based conservative selector |
|---|---:|---:|
| Physical SET EX correct | **35/88** | **36/88** |
| Candidate choice changed | — | 5/88 |
| Paired rescue / regression | — | 1 / 0 |
| Net | — | +1 |
| Total model calls in this CPU audit | 0 | 0 |

The heuristic uses only (a) explicit `column = 'literal'` original question evidence, (b) conservative singular-answer projection width, and (c) **illustrative** obvious invalid-syntax signals. It abstains/tie-breaks to original direct SQL rather than claiming an answer. It is not a general SQL parser or verified source-of-truth schema/FK validator.

**Decision: fail screen; no promotion, no expensive follow-up.** Predeclared `min_net_correct=3` and `require_scorable=96` fail (+1 and only 88 scorable). All eight reference-SQL failures in `retail_world` remain censored. These exact questions were already exposed during prior research, so even a larger retrospective improvement would not prove generalization. Candidate generation remains the dominant bottleneck: 2+2 original Qwen candidate union Oracle@4 is at most 50/88, not deployable performance.

References: [full pairwise mechanism audit](./qwen9b-train96-paired-candidate-coverage-audit-20261119.md), [prior prompt-only negative result](./qwen9b-train96-role-grain-k2-negative-20261119.md).
