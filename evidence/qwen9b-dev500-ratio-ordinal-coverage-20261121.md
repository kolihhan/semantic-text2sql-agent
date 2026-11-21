# Qwen3.5-9B original DEV500: source-only semantic mechanism coverage (2026-11-21)

## Research verdict: BOTH PREDECLARED MECHANISMS FAIL SOURCE COVERAGE GATE

Verified physical CPU [Actions #44911589476](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44911589476): **SUCCESS**. Eight synthetic positive/negative-control tests passed. Original model digest `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`, fixed raw source SHA `ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb`.

- SHA-pinned 1534 BIRD source rows were projected into exactly `case_id, db_id, question` before the model candidate screening process; raw source file deleted. The detector never opened publisher SQL/labels.
- Restored both original immutable generation runs, **32 SHA-pinned original cache shards**, and verified source/question/DB identity. Analyzed **500 unique questions x 8 original Qwen9B generated SQL candidates = 4,000 SQLs**. This is historical exposed DEV, **NOT independent**, and does **not** touch HOLDOUT300.
- Mechanisms were defined before checking the originals, with the preregistered CPU-only gate of >=3 K1 source-flagged questions spanning >=2 physical DB families, followed by negative control and evidence requirements before any Qwen calls.

| Mechanism | K1 flagged unique questions | Distinct physical DBs | Any K1–K8 flagged questions | Gate |
|---|---:|---:|---:|---|
| `NTH_RANK_LIMIT1_WITHOUT_OFFSET` | **1** | 1 (california_schools) | 1 | **FAIL** |
| `SCALAR_GROWTH_RATE_SPLIT_OUTPUT` | **0** | 0 | 0 | **FAIL** |

**One observed ordinal source contradiction candidate:** #50 (California Schools), question requests 7th-ranked value; all eight original SQL candidates return rank-1 because of `ORDER BY ... LIMIT 1` with no offset/window. This is source-observable and was already documented in PR #123. It is not proof of a new repair or an EX rescue.

**Key limitations:** The detector is deliberately high-precision and narrow. 0/500 says only that these *exact lexical+SQL patterns* were not present in frozen K1, not that no ratio/ordinal semantic errors exist. One flagged question is not a reliable precision estimate; the historical cohort is exposed and source selection is not independent. A generic ratio detector or broad fuzzy ordinal heuristics would constitute a **new predeclared mechanism**, not a retrospective relaxation of this failed gate.

## Research decision

1. **Stop both mechanisms before any new Qwen9B inference or SQL modification.** Repeating Qwen calls on one exposed case cannot demonstrate generalization.
2. Historical DEV500 K1 203/500, selected K8 233/500, oracle-any K8 285/500 ([PR #122](https://github.com/kolihhan/semantic-text2sql-agent/pull/122)) leaves 215/267 selected-wrong cases with zero correct generated candidates. New candidate generation is the larger frontier, but requires a real cross-DB grounded mechanism.
3. Next proposed source-first research: evaluate an explicit *answer entity versus joined row population* witness gate as one hypothesis only if question-level counted entity, FK direction and physical cardinalities are all unambiguous. PR #124 gave four cross-DB examples **and counterexamples**; PR #125 demonstrated an irrelevant TRAIN48 source cohort with zero actionable triggers. Include originally correct controls to avoid spurious COUNT DISTINCT, ratio and group regressions. Do not modify baseline Guarded, train SQL, open HOLDOUT300, or promote performance from the present evidence.

Code: [source-only CPU probe](https://github.com/kolihhan/semantic-text2sql-agent/blob/research/source-first-ratio-ordinal-probe-20261121/scripts/probe_qwen9b_dev500_source_mechanisms.py), [synthetic tests](https://github.com/kolihhan/semantic-text2sql-agent/blob/research/source-first-ratio-ordinal-probe-20261121/tests/test_probe_qwen9b_dev500_source_mechanisms.py). Original Actions artifact: `original-qwen9b-dev500-source-only-ratio-ordinal-research-screen`.

**No new accuracy, repairs, training, or score claims.**
