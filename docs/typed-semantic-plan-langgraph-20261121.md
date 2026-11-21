# Typed Semantic Plan Verifier — Explicit LangGraph workflow (research only)

## Mechanism, model, and non-goals

- **Model**: original fixed Qwen3.5-9B; **no SQL training** or model replacement.
- **Goal**: improve *real strict Semantic Execution Match*, not SQLite compilation success.
- Historical PR #111 generated SQL *from* a semantic plan. This PR #130 does **not**. SQL generation and bounded syntax repair remain the same. The semantic contract is a **separate postgeneration hypothesis**, derived only from question, evidence and schema, never from candidate SQL or gold labels.
- Historical PR #118's lexical conflict detector found 3/61 wrong DEV100 and zero TRAIN48 signals. Its low recall does not justify claiming correct SQL when a check abstains.

## Actual graph, enabled only by `semantic_plan_verifier=True`

```
START -> generate -> verify (preflight)
                        | fail       -> repair -> verify (max 2 total)
                        | pass       -> semantic_router (CPU, one visit)
                                       | selective, no risk
                                       | -> execute [semantic_unverified]
                                       |
                                       | always or selective risk
                                       v
                                  semantic_plan_contract
                                         | Qwen +1 call
                                         | cache in Graph state
                                         v
                                  semantic_plan_verify
                                    | CONFLICT -> semantic_plan_repair
                                    |                   | Qwen +1 call MAX
                                    |                   v
                                    |                verify -> semantic_plan_verify (cached plan)
                                    |                   | still wrong/invalid
                                    |                   v
                                    |             semantic_plan_refuse
                                    | CONSISTENT -> execute [ok]
                                    | UNVERIFIED -> execute [semantic_unverified]
```

The `semantic_plan_policy` is `always` by default for this opt-in experimental mode; `selective` uses an intentionally conservative **unvalidated** CPU question/SQL risk heuristic. Skipped coverage is measured as `semantic_unverified`, **not** as verified. A model-generated high-confidence contract is still not a proof of correctness, and checking an AST is not the same as proving a SQL answer.

### Node responsibilities / state

- `verify`: single statement/read-only/SQLite compile only; **zero semantic model calls**.
- `semantic_router`: deterministic risk/cost decision, not a correctness verdict.
- `semantic_plan_contract`: extracts a typed JSON semantic contract using question+schema/evidence. This cached `semantic_plan_extraction` is never regenerated for the semantic repair.
- `semantic_plan_verify`: SQLGlot AST + actual physical catalog comparison, with `consistent/conflict/unverified` outcome and typed violations. **No Qwen calls**. Unknown relations, filters, projections, joins without declared FK, and unsupported CTE are not falsely certified.
- `semantic_plan_repair`: one explicit violation-guided SQL edit, no generic repeat critique.
- `semantic_plan_refuse`: blocks SQL if the same contradiction remains or repaired SQL violates preflight; no infinite loop.
- `execute`: may return physical rows but keeps status `semantic_unverified` unless the measured structural contract agreed.

Original guarded `main` graph is unchanged when experimental flag is disabled; other existing research switches are mutually exclusive.

## Cost and acceptance

| Arm | Calls beyond baseline generation for preflight-passing SQL | Risk |
|---|---:|---|
| Default Guarded | 0 (only syntax repair as before) | executable-but-wrong can pass |
| Always-on Plan Verifier | +1 plan call; +1 if explicit conflict | extra latency even on correct SQL |
| Selective Plan Verifier | +1 on CPU-risk-triggered SQL; +1 if explicit conflict | under-detection when CPU router skips |

Historical fake/synthetic improvements are **not** BIRD EX. The graph/cost controls passed CI [#44877085748](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44877085748), **126 tests**, including actual graph node/edge shape, one-time contract reuse after repair, retry budget, low-risk skip, wrong SQL conflict detection and default unchanged. No real Qwen model predictions were run for the new workflow. Do not merge or claim EX gain without:

1. Source-locked same-model Qwen baseline/always/selective paired SQL and independently scored physical strict SET/BAG EX;
2. Verified original-correct controls, semantic CONFLICT vs false-positive counts, undetected wrong SQL, rescues and regressions;
3. Real model call count and p50/p95 latency, constrained by compute budget;
4. Database-disjoint/source-shift replication. Previously exposed DEV100, DEV500 and protected TRAIN64 are not a pristine holdout for tuning.
