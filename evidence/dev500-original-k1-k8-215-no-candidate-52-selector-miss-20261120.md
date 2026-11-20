# BIRD DEV500 original Qwen3.5-9B K1↔K8: physical candidate generator versus selector frontier (2026-11-20)

**Complete CPU-only retrospective decomposition. No new model inference, SQL execution, training, Guarded edits or HOLDOUT300 contact.**

## Pinned evidence

- Immutable source-only original Qwen K1 risk/control corpus + score join [Actions #44660486791](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44660486791), score artifact `14595466511`.
- Eight original 8-second-per-query SQLite prefix-frontier worker artifacts from [Actions #44222560767](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44222560767), pinned fixed Qwen9B model digest `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`, pinned public BIRD DEV source digest `ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb`.
- [**New zero-inference joined and strictly hash-validated run #44667878387**](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44667878387) SUCCESS, artifact **14599155860** `ORIGINAL-QWEN-DEV500-K1-K8-ALLWRONG215-SELECTOR-MISS52`. No publisher SQL used in this new join.
- Self-correction: first two workflow attempts failed closed on an incorrectly transcribed source SHA (missing final digits); after correcting the exact 64-character digest, original worker validation and full 500-case aggregate passed. Neither failed workflow is a model experiment or negative benchmark finding.

## Exclusive outcome taxonomy — all original frozen 500

| Original K1 | Original K8 selected | K8 oracle: any of original 8 SQLs correct? | Count | Interpretation |
|---|---|---|---:|---|
| correct | correct | yes | **197** | Stable correct |
| wrong | correct | yes | **36** | K8 rescues K1 |
| correct | wrong | yes | **6** | K8 selection harms K1 |
| wrong | wrong | yes | **46** | Correct alternative candidate generated, but missed |
| wrong | wrong | no | **215** | No matching candidate generated in eight |
| **Total** | | | **500** | |

K1 **203/500 (40.6%)**, K8 **233/500 (46.6%)**, oracle-any correct among 8 **285/500 (57.0%)**. K8 generated-a-correct-candidate selector misses: **6+46=52/500**, corresponding to **19.5% of final 267 errors**. All-eight-no-match cases: **215/500**, corresponding to **80.5% of final 267 errors**.

K1→K8 paired gains **36 rescues, 6 regressions, net +30**. The K8 selection lost six cases where first original candidate was already correct. **Do not simply increase K** without monitoring this regression.

### Database heterogeneity

| DB | n | All-eight no-match | Oracle-positive but selector missed |
|---|---:|---:|---:|
| financial | 34 | 27 | 3 |
| california_schools | 28 | 20 | 3 |
| card_games | 62 | 29 | 8 |
| codebase_community | 62 | 17 | 4 |
| debit_card_specializing | 19 | 12 | 2 |
| european_football_2 | 41 | 14 | 2 |
| formula_1 | 56 | 25 | 7 |
| student_club | 53 | 12 | 4 |
| superhero | 43 | 6 | 7 |
| thrombosis_prediction | 53 | 28 | 3 |
| toxicology | 49 | 25 | 9 |

### Source-only question-family proxy breakdown (overlapping tags; NOT causal diagnoses)

| Keyword family | n | K1 right | K8 right | No correct K8 candidate | Selector miss |
|---|---:|---:|---:|---:|---:|
| Ranking/top | 85 | 23 | 25 | 51 | 9 |
| Ratio/percent | 43 | 12 | 13 | 21 | 9 |
| Count | 118 | 52 | 59 | 50 | 9 |
| Temporal | 32 | 10 | 11 | 19 | 2 |
| Relation | 32 | 12 | 12 | 18 | 2 |

Do not sum rows because tags overlap. Question taxonomy is a simple source-English keyword proxy: difficulty, DB schema, question length and example distributions confound the apparent failure rate.

## Independent reasoning and next experiment gate

1. **Generation bottleneck exists within original K8**, but **not every all-eight-wrong case was semantically adjudicated**. One of 500 publisher reference SQLs was not scorable at 8 seconds; do not turn `pass8_oracle_set=False` into a universal gold-label semantic proof.
2. Improving the final answer by writing a better post-generation verifier cannot rescue 215 all-eight-wrong cases unless it **generates or repairs a new candidate**; rescoring existing candidates has an upper bound of 285/500 under this frozen pool.
3. A selector-alone improvement has headroom of 52/500 but might regress previously correct K8 selections. All selection tests must score paired rescues AND regressions, same original candidate pool, 8-second cap, real costs.
4. **Repeated vanilla Semantic Planning is not an evidence-based first move.** Previously fixed-Qwen prospective BIRD DEV32 planning comparison [#43851807351](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43851807351) reported **15/32 baseline → 13/32 planning, 1 rescue and 3 regressions**; do not relabel it a win or overfit DEV500.
5. For novel candidate generation: select a pre-registered source-only semantic mechanism such as output entity/grain, relational bridge direction, conjunction scope, comparison denominator and time scope. On a frozen predeclared mixed original-K1-right/K1-wrong set, generate alternatives under the exact same Qwen3.5-9B, model/API/latency budget; retain primary real paired result correctness, rescue/regression, K1 and K8 baselines, per-DB cluster and generation costs. HOLDOUT300 stays sealed until pre-registered promotion.
6. **Decision:** priority candidate generation/semantic grounding; secondary candidate selection using existing K8 pool. No blanket COUNT DISTINCT, no SQL QLoRA, no default Guarded modification.

Full case-ID labels and 11-DB/keyword groupings are in run artifact **14599155860**. The evidence reuses old public DEV outcomes; zero new benchmark score.
