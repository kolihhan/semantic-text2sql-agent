# BIRD DEV500 — all 215 Qwen3.5-9B K8 zero-correct-candidate failures

**Frozen original data only. No inference, training, tuning, hidden holdout, or production-code changes.**

- Original Qwen9B model SHA: \`56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90\`
- Source SHA256: \`ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb\`
- Physical scoring: 8-second read-only SQLite time limit, not a new official unbounded BIRD result
- 32 immutable cache shards, 500 unique questions, 4,000 original model-generated SQLs
- 8 original physical result-scoring worker artifacts; 499 reference SQLs scored; #1126 certified all-eight invalid and reference timeout

## Observed operational failure classes (mutually exclusive)

| Physical situation | Questions | What this proves |
|---|---:|---|
| All 8 candidates executed, **none matched reference** | 97 | Pure compiler repair cannot directly solve these observed mismatches |
| Some executed, but none matched | 90 | Mixed reliability + result mismatch; semantic cause unknown |
| 0/8 executed | 28 | Execution/reliability bottleneck; includes one all-invalid reference-timeout-certified question |
| **All K8 no correct candidate** | **215** | Correct SQL not reachable in the eight original samples |

## Mechanical candidate diversity, not meaning

- 88/215 have eight distinct normalized SQL texts; distinct strings do not imply correct or different semantic strategies.
- 183/215 have at least four distinct SQL texts.
- 5/215 generated the same normalized SQL eight times.
- 28/215 had at least one model response ending at the 384-token generation cap; this is not necessarily the cause of the wrong answer.

## Retrospective SQL AST comparison — evidence gaps, **not root causes**

| Non-exclusive AST disagreement indicator | Number of cases |
|---|---:|
| \`no_parsed_candidate_has_all_reference_tables\` | 60 |
| \`no_parsed_candidate_matches_reference_output_width\` | 45 |
| \`all_8_candidates_same_coarse_ast_shape\` | 40 |
| \`no_parsed_candidate_has_reference_distinct\` | 40 |
| \`at_least_one_model_generation_hit_token_limit\` | 28 |
| \`no_parsed_candidate_has_reference_division\` | 8 |
| \`no_parsed_candidate_has_reference_join\` | 8 |
| \`no_parsed_candidate_has_reference_sum\` | 6 |
| \`all_8_candidates_same_normalized_sql\` | 5 |
| \`no_parsed_candidate_has_reference_group\` | 5 |
| \`no_parsed_candidate_has_reference_order\` | 5 |
| \`no_parsed_candidate_has_reference_limit\` | 4 |
| \`no_parsed_candidate_has_reference_where\` | 4 |
| \`no_parsed_candidate_has_reference_avg\` | 3 |
| \`no_parsed_candidate_has_reference_count\` | 3 |
| \`no_parsed_candidate_has_reference_exists\` | 3 |

These labels compare the *existing* gold SQL's AST with all eight already-generated candidate SQL ASTs. Alternate equivalent SQL can use different tables, projection widths, operators or values. A parser rejection is not a database execution failure. Counts overlap and must never be summed to 215.

## Where to put effort, using evidence rather than candidate-count sweeps

1. **97 fully executable zero-oracle cases:** priority for human blind review of relation roles, output grain, predicates and aggregation. Compare question + schema to one candidate **before** consulting gold SQL; then check the publisher reference and adjudicate ambiguity.
2. **90 mixed cases:** split compiling/timeout problems from successful-but-wrong candidates, then investigate incorrect semantics on the executed subset.
3. **28 zero-executable cases:** review identifier quoting, invented physical columns, unsupported dialect, pathological queries and generated token limits using actual compiler errors (the K8 cache itself does not contain individual compile-error reasons).
4. This census alone does **not** justify a new QLoRA run, an optimizer, a new plan module, or a claim of more EX. Freeze a causal error mechanism and independent evaluation first.

## Auditable source and restrictions

- Original generation Actions #44086713331 and #44093103960.
- Original complete physical frontier run #44222560767; independent CPU aggregate #44223938142.
- CSV/JSON-style per-case source-ID ledger is in \`evidence/dev500-zero-oracle-215-ledger.json\`. It includes 215 case IDs, DB/difficulty, question excerpt, real execution counts, SQL-diversity stats and non-causal structural flags; no original 4,000 SQLs or reference gold SQL text copied into git.
- Frozen BIRD DEV500 and HOLDOUT300 manifest: \`manifests/bird-dev500-holdout300-20261116-v2.json\`. The HOLDOUT300 case IDs were validated as disjoint but never scored/selected.
- Dataset attribution: BIRD (Li et al., NeurIPS 2023); source mirrored by \`birdsql/bird_sql_dev_20251106\`, pinned revision \`3c11fb193e5439b338e23677fa0aae11e8b85db9\`.
- The 500 DEV question identities and 11 physical databases are already exposed; neither this report nor subsequent tuning on these 215 proves independent generalization.
