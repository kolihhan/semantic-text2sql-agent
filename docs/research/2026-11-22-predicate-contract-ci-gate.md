# Predicate-contract CPU gate — verified 2026-11-22

Status: **CPU TEST GATE PASS; SEMANTIC EFFICACY UNTESTED**.

## Independent evidence
- Branch: `research/physical-signal-blind-audit-20261122`.
- Prior commit `448f81872283d17f2982ddc462eeb6b639afe24d`: CI run [44949358140](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44949358140) failed (105 passed, 3 failed); missing sqlglot dependency.
- Fix commit `3a4fdc70fdc6d5ec6c51e62f9d0006eb8672cc92`: CI run [44954212113](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44954212113) **success**, job `136998622787`, `108 passed in 1.85s`, compileall succeeded. The dependency was installed as sqlglot 28.10.1.
- This establishes only a green test suite, not any Qwen inference, semantic detection accuracy, or EX improvement. No result artifact for a new paired model evaluation was reviewed in this gate.

## Frozen next gate (no gold or protected DB)
1. Use the existing detector and existing frozen SQL predictions in read-only diagnostic mode. Record eligible / abstained / flagged counts by physical database, without correctness labels or gold SQL.
2. Predefine negative controls: valid literal absent from database, ambiguous alias, nested scope, multiple candidate columns, case/collation mismatch, and intentionally conflicting question/predicate binding. The detector must abstain on ambiguity and never infer incorrectness solely from an empty probe.
3. Record SQLGlot version, branch SHA, physical DB hashes, exact case IDs and per-case evidence before any label inspection. Exclude HOLDOUT300 and protected `simpson_episodes` / `works_cycles`.
4. Only if a source-grounded, independently falsifiable conflict survives negative controls: implement optional LangGraph conditional routing with original-SQL fallback and at most one selective repair. Keep baseline unchanged.
5. Gate real Qwen3.5-9B Q4_K_M digest `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90` on complete paired DEV100 with strict SET EX, BAG EX, rescues, regressions, database spread, calls/tokens/latency and independent source/DB transfer. Historical Guarded = 39/100. Do not promote from replay-only results.

## Stop rule
Reject this signal if it has zero eligible conflicts or unacceptable false positives; do not tune on observed EX labels. Do not repeat the prior literal-normalization intervention (31/64 to 32/64 with BAG regression). No promotion PR until both paired DEV100 and independent transfer gates pass.
