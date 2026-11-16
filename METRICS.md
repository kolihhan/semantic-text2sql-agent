# Text-to-SQL metric contract — 2026-11-16

## Objective and immutable controls

- **Primary real objective:** semantic correctness of the SQL answer without replacing **Qwen3.5-9B**. Pinned historical model: `qwen3.5:9b-q4_K_M`, digest `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`. Verify digest before each new run.
- **Semantic score (separate user-defined metric): NOT YET DEFINED.** Historical conversations and current repo establish only semantic *goal* (requested projection, filter, literal, relation, group/grain, aggregation, answer shape), not a numerical metric with rubric/weights. Do not invent a metric or silently relabel EX as the requested independent semantic score.
- **Scorer script:** NOT SPECIFIED; **LLM-as-judge:** not specified/not enabled; **manual agreement:** not measured; **same-output score variance:** not measured for an independent semantic scorer. **STOP all optimization of this semantic score until owner explicitly supplies/approves a definition and a blinded annotation/repeatability protocol.**
- The *existing* SQL execution-match score below is the defensible operational correctness proxy, NOT an independently validated semantic judgment.

## Existing metrics and source of truth

| Metric | Operational definition | Implementation | Known limitations |
| --- | --- | --- | --- |
| Strict set-EX-compatible | For same SQLite DB, predicted and gold read-only execution row tuples compared as sets, equality scored 1/0; capped/errored results are not valid matches | `evaluation/run_bird.py::_official_ex_results`, pinned BIRD reference `483554eae102996f5ec1f4feab4e78ef29c2a394` | Project-local reproduction of BIRD contract; not direct execution of upstream scorer; ignores row order and multiplicity |
| Execution success | Predicted read-only SQL runs without error or unsafe/truncation issue | `evaluation/run_bird.py`, `src/semantic_sql/execution.py` | NOT correctness |
| Bag-EX diagnostic | Compare row multiplicities, not only sets; when used specify script and cap protocol | Separate PRO diagnostics in `docs/pro-research-gates.md` | Not interchangeably scored with strict set-EX; audit provenance first |
| Grounding precision/recall | Table/column/anchor/FK/value lexical coverage inferred from gold SQL | `evaluation/grounding_metrics.py` | Proxies, gold-SQL parser approximations; **not semantic accuracy** |

The frozen-100 historical artifact `runs/qwen9b-frozen100/summary.json`: Direct **33/100** vs Guarded **39/100** set-EX-compatible; wrong→right **6**, right→wrong **0**; execution success 75/100→**90/100 conservative**. Semantic revision 39/100→38/100 (0 rescues, 1 regression). **Evidence: checked-in summary; provenance Actions `37307108955` has a failed treatment shard/skipped aggregate.** An offline replay is not an independent model rerun. This is a *seen development slice*, not generalization.

## Noise, pairing, statistics

- Identical stored SQL + same immutable SQLite snapshot + same deterministic scorer should give identical set-EX labels; this does **not** establish stochastic model-run variance or semantic-human-label reliability. Audit capped/unscorable separately.
- Match identical task IDs, data SHA, model digest, inference configuration, schema and repair budget. Report per-case both arms, rescue (b), regression (c), delta ((b-c)/n), exact two-sided McNemar p, and paired bootstrap percentile 95% CI (resample **task IDs**, 10,000 draws with fixed seed). Include DB-cluster bootstrap as sensitivity: BIRD DEV has only **11 physical DBs**, so source independence is impossible within its sample.
- A statistical effect must have 95% CI excluding 0, nonnegative regression/cost controls and independent source replication before **promotion**. **Sample-size threshold is PROPOSED, NOT OWNER-APPROVED**: use ≥200 paired dev cases and ≥30 discordant pairs before any decisive directional label, then recompute required n from pilot discordance; if below, label **INSUFFICIENT EVIDENCE**, not rejection.
- If a semantic LLM judge is ever approved, evaluate identical outputs ≥3 times under recorded seed/settings, report within-output variance and blinded double-human-label agreement (raw agreement and Cohen's kappa). No stronger model may answer inference questions; stronger-model teacher data require prior explicit consent and disclosure.

## Frozen population and leakage

- BIRD DEV population: **1,534** tasks. **ACTUALLY FROZEN 2026-11-16**, new DEV500 + untouched HOLDOUT300 under `manifests/bird-dev500-holdout300-20261116.json`, Actions `43833618248`, source SHA `ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb`. Selection used physical database × official native difficulty, fixed seed `bird-metric-gate-20261116-v1`, excluded historical100 IDs and one exact duplicate question group (two extra question IDs); 1,432 eligible after those exclusions. Original 100 kept historical only. **Zero model runs** on these newly frozen populations.
- **CRITICAL:** old frozen100 already touches all 11 BIRD DEV physical DBs (see `docs/pro-research-gates.md`). The new 300 can be **question-disjoint only**, not database/source independent. Never claim it proves out-of-source generalization. Additional external database corpus is required.
- Do not invent difficulty labels: if difficulty metadata is absent, block requested difficulty stratification and record that blocker; do not silently approximate using gold SQL.
- Training data: BIRD TRAIN only for SQL QLoRA adapter; exclude all BIRD DEV question/SQL and near-duplicates. Preserve hashed manifests and fail closed on unavailable exclusion data. Never use BIRD DEV labels for prompt retrieval.
- Default guarded path remains schema-grounded; `max_repairs=2`; value-grounding separate; all arms same Qwen9B digest.

## Exposure ledger

| Population | Known views/uses as of 2026-11-16 | Status |
| --- | --- | --- |
| Original BIRD DEV frozen100 | Historical matched generation, semantic revision, offline replay and multiple subsequent forensic analyses; **true count not reconstructable from available repo artifacts** | Contaminated development evidence; no new holdout claims |
| BIRD DEV various 16/20 pilots | Multiple published prompt/retrieval/consensus comparisons; count incomplete | Seen; no score-based re-splitting |
| New BIRD DEV500-v2 | 0 committed inference runs; Actions `43834589773`; excludes 100+126 known exposures | FROZEN, unused dev; question-disjoint |
| New BIRD DEV300-v2 | 0 committed inference runs; Actions `43834589773`; excludes 100+126 known exposures | FROZEN, untouched by new inference; not physical-DB independent |
| BIRD DEV v1 split | Exposure audit found 59 old-pilot overlaps in dev and 23 in holdout | QUARANTINED; never evaluate as an untouched holdout |
| Spider DEV pilot12 | One reported intervention and repeated post-hoc audits | Seen; exploratory only |

Every future run must append population, manifest SHA, run id, **cumulative exposures**, intervention, hashes, seeds, and whether labels/traces were inspected.

## Four-arm adapter plan (not authorized to execute yet)

A: existing prompt+base; B: revised prompt+base; C: existing prompt+adapter; D: revised prompt+adapter. Fix model/base, task IDs, tool configuration, budget, scoring and seed schedule. Log adapter/data/prompt SHA, decoder, token/time/GPU estimate, failure taxonomy and regression suite. Adapter training permission is updated by owner instruction 2026-11-16; previous `docs/pro-research-gates.md` wording “No SQL training” is superseded for **QLoRA adapters only**; no base swap, no stronger inference model or unauthorized teacher.
