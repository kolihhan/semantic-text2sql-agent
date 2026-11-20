# Question–Schema–SQL Counted Unit Consistency Gate: frozen prior Qwen TRAIN48 retrospective screen (2026-11-20)

**FINAL: strict three-state rule survives synthetic tests but FAILS real candidate-generation eligibility. ZERO CONFLICT hits. No new Qwen, no SQL generation/training.**

## Method, isolation and exact archived artifacts

- Freeze original same Qwen3.5-9B model digest `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`, original pre-gold TRAIN48 source+4 original K4 SQL candidates per task from previous structured counterhypothesis [Actions #44508116418](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44508116418). Original K4 gold-blind selected candidate used for this comparison, **NOT the original single-sample K1**.
- Synthetic `tests/test_question_schema_sql_unit_gate.py`: true 1:N Patient→Laboratory bad count -> **CONFLICT**; correct `COUNT(DISTINCT Patient.ID)` -> **CONSISTENT**; child transaction count joined to customer PK -> **CONSISTENT**, grouping/subquery, missing FK, implicit target PK, ratio and insufficient physical multiplicity -> **ABSTAIN**. Also fixed real-world failures uncovered by tests: `src/semantic_sql` is actual package path; NEVER execute SQL with literals replaced for parsing; SQLite `PRAGMA foreign_key_list` can have missing target column, fail-closed.
- First physical source-only run [Actions **#44714355170**](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44714355170) executed all 48 original selected Qwen K4 SQLs against eight SHA-verified real public TRAIN SQLite DBs **without publisher SQL or historical accuracy labels**, emitted immutable original source judgment artifact **14618986454** `BIRD-ORIGINAL-QWEN-TRAIN48-UNIT-GATE-SOURCE-ONLY-FROZEN`. Its *downstream* old-score job failed on a Python import packaging issue; source-gate job and sealed uploaded artifact **SUCCESS**. Do not pretend the whole two-job run succeeded.
- Instead of rerunning expensive database downloads/old model inference, the independent **score-only** replay [Actions **#44714997078**](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44714997078) consumed that exact sealed source artifact and eight historical original K4 **Boolean** scorer shards, then SUCCESS. Final score artifact **14618602456**. No new Qwen/DB execution/references; `TRAIN48` was used earlier for *another* experiment and thus is **not an untouched promotion holdout**; it was not used to tune this rule after scored outcome, but some TRAIN examples were manually inspected before implementation.
- 48 original TRAIN questions across 8 original physical databases; 46 public reference SQL outcomes scorable (26 original K4 gold-blind-selected SET-correct, 20 wrong), 2 publisher reference executions censored. This is **not 48 independent physical databases** or generalization.

## Actual original-Qwen selected SQL correctness versus UNIT-ONLY gate

| Unit-only gate | Original selected SQL correct | Original selected SQL wrong | Censored |
|---|---:|---:|---:|
| **CONSISTENT** | **3** | **0** | 0 |
| **CONFLICT** | **0** | **0** | 0 |
| **ABSTAIN** | **23** | **20** | 2 |
| Total | 26 | 20 | 2 |

Exactly 3 CONSISTENT arose from **shipping** DB (original source indices 4400, 4430, 4459); all original selected candidate SQLs are K4 already-correct. **45/48 ABSTAIN (93.75%); zero actual conflict detections on 20 original wrong results.** There are no affected candidates for a new generation intervention.

This is *not* evidence that the rule is precise across wrong/correct populations: with zero conflict flags, false-alarm risk cannot be estimated or bounded usefully; "0 regressions" would be a tautology of no action. Nor does a unit-consistent flag establish whole-query correctness.

## Exact 3-state intervention specification, unchanged after this screen

**CONSISTENT / KEEP ORIGINAL:** only when one explicit count question directly names a grounded schema table, original SQL has exactly one simple count, and it counts directly from the target table, `COUNT(DISTINCT target_PK)`, or target child rows joined only to declared PK parents. No claim beyond counted unit.

**CONFLICT / PROPOSE ONE NEW CANDIDATE (NOT automatically execute):** require explicit target noun grounded to one table with a simple PK, one simple `COUNT(*)` in original SQL, exactly one documented 1:N joined child referencing that PK, and *physical read-only execution of exact original WHERE/ON* showing `COUNT(*) != COUNT(DISTINCT target_pk)`. No publisher SQL, target IDs, task-specific SQL rewrite or gold output used to decide. The rule produces a **diagnostic candidate indication**, not SQL generation.

**ABSTAIN / DO NOT MODIFY:** every other case: no unambiguous answer entity or FK mapping, missing FK target, complex/nested/grouped/ratio query, multiple COUNTs, SQL parse uncertainty, unscorable physical query, or equal row-vs-entity population despite potential fanout. Actual BIRD TRAIN48 mainly falls here.

## Predeclared decision and next research direction

- Required screen for *considering* a later official paired fixed-Qwen A/B was **≥3 wrong original selected SQLs flagged CONFLICT, across ≥2 different TRAIN DBs, with zero original-right conflict flags**; actually **0**, **STOP**. Even a passing retrospective screen would not by itself authorize promotion.
- Do **not** weaken this gate against its visible TRAIN48 wrong cases or DEV500 to manufacture positive examples; that would be adaptive overfit. Its main weakness is narrow entity grounding and absence of annotated target/schema role for nontrivial questions. A future independent mechanism would need a source-first typed target entity/answer grain and true FK/semantic population binding, then a newly frozen source/DB-disjoint evaluation, including original correct controls and abstention precision.
- PR #116 prior generic K4 structural prompt already selected 26/46→25/46 at 2.16× generation time. PR #124 confirmed 3/4 handpicked already-K1-correct queries would regress under arbitrary COUNT DISTINCT replacement. Neither negative result is contradicted.
- HOLDOUT300 **untouched**; model unchanged; no training; no new prospective BIRD EX. **DRAFT / DO NOT MERGE.**

**Conclusion: explicit, bounded decision logic exists, but its real screening exposure is zero. This specific conservative mechanism is rejected for a new costly generator A/B.**
