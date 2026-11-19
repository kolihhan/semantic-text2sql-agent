# Fixed original Qwen3.5-9B TRAIN96 semantic role/grain K2 A/B — complete **negative** (2026-11-19)

## Verified original run and CPU recovery
- [Original physical-model Actions #44476207751](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44476207751): all **12 physical DB** inference + publisher-score shards succeeded; original final aggregate failed due to a whole unscorable reference DB.
- [CPU-only score recovery Actions #44483833946](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44483833946): completed **SUCCESS**, artifact `final-fixed-Qwen9B-roles96-CPU-salvaged-original-scores`, id **14509273891**. This reuses EXACT original frozen source + ALL 12 previously generated immutable scored files, **no additional model call or SQL regenerations**.
- Original Qwen3.5-9B digest `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`, same two model calls per arm, seeds 42/43, same physical SQLite/schema. Control = two independent normal SQLs; treatment = two independent generator-prompt role/grain/ratio-constrained SQLs. Deployed first candidate selected **before** reference SQL scoring.

## Actual strict physical SET/BAG execution results (88 scorable out of 96)

| Metric | Original control | Semantic-role treatment | Change |
|---|---:|---:|---:|
| **SET execution match** | **35/88 (39.77%)** | **29/88 (32.95%)** | **-6 correct** |
| BAG execution match | 34/88 | 27/88 | -7 |
| Second candidate SET match | 36/88 | 35/88 | -1 |
| Hindsight oracle@2 (NOT deployable) | 42/88 | 42/88 | 0 |
| Selected SQL executable | — | 68/96 | see record |
| Model calls | 192 | 192 | 0 |
| Original generation wall-seconds across shards | 4,647.20 | 5,264.05 | +616.85 (+13.27%) |

Paired main SET EX **7 rescues**, **13 regressions**, **net -6**; rescues occur across 5 DBs. **FAIL** preregistered promotion: requires >= +8 net, >=4 rescue DBs, all 96 publisher references executable, and DB-clustered CI lower >0.

### Source censoring / why original Actions final aggregate failed
All **8 publisher SQL executions from `retail_world`** returned SQLite **`OperationalError`**, while the other 11 DBs' 88 questions were scorable. This is a **reference-side scoring limitation**, not evidence that 8 control or treatment answers were correct/incorrect.
- `scorable=88`, `censored=8`, `unscorable_database_ids=['retail_world']`.
- Original 12-DB cluster bootstrap is **unestimable**; corrected CPU aggregator explicitly emits `paired_db_cluster_CI95=null` and `bootstrap_CI_status=UNESTIMABLE_SOME_PHYSICAL_DATABASES_UNSCORABLE`, instead of failing or claiming a confidence bound.
- Frozen original physical TRAIN96 question IDs exclude all 615 historically exposed question IDs; no BIRD DEV500, no reserved HOLDOUT300, no last two protected original physical TRAIN DBs. New questions share some previously seen database *schemas*; do not claim new-database generalization.

**Research decision:** REJECT semantic role/grain **prompt-only** generator variant. It makes more regressions than rescues even at equal Qwen call count and raises overall latency. Avoid ungrounded claims that prompt constraints fix semantic generation. Keep default original generator and architecture unchanged; inspect paired regressions and the reference-side `retail_world` SQLite failure before designing a different mechanism or new frozen evaluation. Original final run remains red because historical workflow is immutable; CPU recovered artifact is the valid complete report for the scorable subset.
