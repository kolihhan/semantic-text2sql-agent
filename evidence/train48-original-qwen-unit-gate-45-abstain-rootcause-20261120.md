# Original Qwen TRAIN48: 45 abstentions are mostly out-of-scope SQL structures, NOT demonstrated Entity–Relation failure (2026-11-20)

**COMPLETE historical-source CPU-only audit. No SQL model calls, execution, training, publisher gold SQL, DEV500 or HOLDOUT300.**

## Immutable provenance & replay

- [Initial source-only physical unit gate #44714355170](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44714355170) completed its **first job** and preserved artifact **14618986454**, 48 original Qwen K4 selected SQLs vs source questions, 8 pinned BIRD TRAIN DBs, with one three-state decision and `unit_reason` per task. The old-score job on that run failed due to import, not source evaluation.
- [Separate post-seal original scorer join #44714997078](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44714997078) SUCCESS, old Boolean score aggregate artifact **14618602456** (26 correct, 20 wrong, 2 publisher reference execution censored). These score labels were never passed to the original source-only gate.
- **New full SHA-reconciled raw-source abstention breakdown [Actions #44742925833](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44742925833)** SUCCESS, artifact **14631093641**, `ORIGINAL-QWEN-TRAIN48-UNIT-GATE-EXACT-ABSTAIN-CAUSES-AND-K4-OUTCOMES`. Source-only original 48 questions, all 8 original fixed-Qwen K4 generated shards, frozen unit decision artifact and original already-scored Boolean outcome shards rejoined by exact source ID/question SHA/selected original SQL SHA. No reference SQL fetched, no new SQLite execution, no Qwen calls.
- All 48 have the same fixed original Qwen3.5-9B digest `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90` and source SHA. TRAIN48 was **exposed in a previous experiment**; this is descriptive posthoc diagnosis, not an untouched evaluation nor an independent mechanism prevalence estimate.

## Precisely 45 first abstentions, full non-overlapping counts

| First `ABSTAIN` reason | n | Original selected K4 SQL correct | Wrong | Reference censored |
|---|---:|---:|---:|---:|
| `NOT_SINGLE_SIMPLE_COUNT` | **32** | 15 | 15 | 2 |
| `COMPLEX_OR_UNSAFE_SQL` | **7** | 4 | 3 | 0 |
| `ANSWER_UNIT_NOT_GROUNDED` | **4** | 3 | 1 | 0 |
| `TARGET_UNIT_NOT_SINGLE_SOURCE_TABLE` | **1** | 0 | 1 | 0 |
| `TARGET_TABLE_HAS_NO_SIMPLE_PK` | **1** | 1 | 0 | 0 |
| **TOTAL ABSTAIN** | **45** | **23** | **20** | **2** |

Other three original selected K4 cases were `CONSISTENT` (2 child rows joined to parent PK, 1 direct from target) and all **3 were correct**. Zero `CONFLICT` and no measurable detection or repair efficacy.

### Why "structure unsupported" is not evidence of semantic grounding error

- **39/45 (86.7%)** abstain immediately because the original selected SQL is not supported by the deliberately narrow single simple count regex, or is complex/grouped/multi-SELECT. The coarse reason name `COMPLEX_OR_UNSAFE_SQL` **does NOT imply the SQL is actually unsafe**; this also includes perfectly legitimate `WITH`, `GROUP BY` etc.
- Within the **32** `NOT_SINGLE_SIMPLE_COUNT` cases, **31 original selected SQLs do not even contain any `COUNT(...)` operator**. Among those 31, **27 question texts also lack a simple lexical count marker** defined as `how many | number of | count of | number | count | total number`. **This is a regex observation, not human-verified true task semantics**; count-like tasks can express "average", "ratio", "how many" synonyms, or use other SQL operators.
- The 32 cases contain **15 already-correct, 15 already-wrong, 2 unscorable** original selected SQLs. Hence extending COUNT-only parser coverage into non-count selected SQLs may create *false interventions* on right cases, not just extra true-error coverage.
- The four `ANSWER_UNIT_NOT_GROUNDED` abstentions contain **3 already correct**, 1 wrong. This is a tiny subset; an entity recognizer would not automatically solve most of this cohort.
- **None** of the 45 abstentions reached `RELATION_UNIT_DIRECTION_UNPROVEN` or `ONE_TO_MANY_FK_BUT_NO_PHYSICAL_DIFFERENCE`. This is a consequence of *sequential early exit*, **not evidence** that relation direction is semantically correct when SQL structure is unsupported.

## Independent research correction

**False conclusion to reject:** 45/48 abstentions demonstrate that most errors arise because the model fails entity linking/FK reasoning. The data instead shows a mismatch between broad BIRD TRAIN48 question types and a Count-only research instrument.

**Better next CPU-only experimental design:**

1. Before looking at original SQL correctness labels, predeclare a **source-question eligibility gate** for actual population/count/ratio/grain questions, including common paraphrases but explicitly excluding unrelated ranking, names, listing, sorting, average without denominator semantics unless defined. The requirement should use question + schema, not the existing selected Qwen SQL or labels, to avoid selecting only the problems we already know how to fix.
2. Stratify eligible questions by answer unit (entity count, join-row count, ratio numerator/denominator, relationship set inclusion, temporal scope), with a separate `UNKNOWN` bucket; obtain physical FK/table role validation and independent human/frozen source adjudication for a representative sample. Preserve already-correct original SQL controls.
3. Measure both **mechanism exposure (% eligible out of all)** and **conflict detector specificity/sensitivity within the eligible subset**, distinguish wrong result from proven unit-error ground truth, then only if a genuinely independent source-/DB-disjoint gate is predictive authorize original Qwen3.5-9B candidate A/B at equal compute/turn budget. Keep separate relation binding research; do not merge all errors under COUNT.
4. **Do not loosen current CONFLICT threshold or add rules on known-correctness TRAIN48/DEV500 until the eligibility definition is frozen and an independent cohort is chosen.** No K4 vote tinkering (PR #100 negative), no generic structural counterhypothesis prompt (PR #116 negative), and no blind COUNT DISTINCT (PR #124 false-positive controls).

**Stop-loss:** The current three-state Count-only gate is rejected for generation A/B on this broad cohort; an Entity–Relation research hypothesis remains plausible, but the historical abstention causes do not establish it as the limiting failure mechanism. All PRs remain research-only/Draft, no default architecture changes.
