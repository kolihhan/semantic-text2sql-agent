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

- BIRD DEV population: **1,534** tasks. **V2 ACTUALLY FROZEN**: DEV500 + HOLDOUT300 in `manifests/bird-dev500-holdout300-20261116-v2.json` (Actions `43834589773`), original pinned source SHA `ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb`. Hash strata = physical database × official difficulty; seed `bird-metric-gate-20261116-v2`; exclude historical100, **126** additional recovered previously used BIRD DEV IDs and **2** exact duplicate-question rows. Eligible novel rows **1,306**, with 500 dev + 300 holdout, zero ID overlaps. **V1 `...20261116.json` is QUARANTINED** because 59 prior exposed extra IDs were in its dev and 23 in its holdout. Neither partition uses a new physical DB; previously undocumented exposures cannot be ruled out. Only 32 V2 DEV cases were used in prospective Qwen9B screening run `43851807351`; remaining 468 DEV and reserved 300 HOLDOUT not evaluated in that run.
- **CRITICAL:** old frozen100 already touches all 11 BIRD DEV physical DBs (see `docs/pro-research-gates.md`). The new 300 can be **question-disjoint only**, not database/source independent. Never claim it proves out-of-source generalization. Additional external database corpus is required.
- Do not invent difficulty labels: if difficulty metadata is absent, block requested difficulty stratification and record that blocker; do not silently approximate using gold SQL.
- Training data: BIRD TRAIN only for SQL QLoRA adapter; exclude all BIRD DEV question/SQL and near-duplicates. Preserve hashed manifests and fail closed on unavailable exclusion data. Never use BIRD DEV labels for prompt retrieval.
- Default guarded path remains schema-grounded; `max_repairs=2`; value-grounding separate; all arms same Qwen9B digest.

## Exposure ledger

| Population | Known views/uses as of 2026-11-16 | Status |
| --- | --- | --- |
| Original BIRD DEV frozen100 | Historical matched generation, semantic revision, offline replay and multiple subsequent forensic analyses; **true count not reconstructable from available repo artifacts** | Contaminated development evidence; no new holdout claims |
| BIRD DEV various 16/20 pilots | Multiple published prompt/retrieval/consensus comparisons; count incomplete | Seen; no score-based re-splitting |
| New BIRD DEV500-v2 | 32 prospective EX-rated comparisons in Actions `43851807351`; 468 unused DEV | FROZEN; 32 now exposed, question-disjoint only |
| New BIRD DEV300-v2 | 0 committed inference runs; Actions `43834589773`; excludes 100+126 known exposures | FROZEN, untouched by new inference; not physical-DB independent |
| BIRD DEV v1 split | Exposure audit found 59 old-pilot overlaps in dev and 23 in holdout | QUARANTINED; never evaluate as an untouched holdout |
| Spider DEV pilot12 | One reported intervention and repeated post-hoc audits | Seen; exploratory only |

Every future run must append population, manifest SHA, run id, **cumulative exposures**, intervention, hashes, seeds, and whether labels/traces were inspected.

## Four-arm adapter plan (not authorized to execute yet)

A: existing prompt+base; B: revised prompt+base; C: existing prompt+adapter; D: revised prompt+adapter. Fix model/base, task IDs, tool configuration, budget, scoring and seed schedule. Log adapter/data/prompt SHA, decoder, token/time/GPU estimate, failure taxonomy and regression suite. Adapter training permission is updated by owner instruction 2026-11-16; previous `docs/pro-research-gates.md` wording “No SQL training” is superseded for **QLoRA adapters only**; no base swap, no stronger inference model or unauthorized teacher.

## Prospective one-pass structural intent DEV32 (2026-11-16)

[Verified: successful paired 8-shard + strict aggregate Actions `43851807351`](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43851807351): same immutable Qwen3.5-9B digest, two independently generated SQL candidates, identical 2-repair Guarded path on 32 SHA-selected BIRD DEV500-v2 question IDs. Baseline **15/32**, one-pass structural intent prompt **13/32**; paired wrong→right **1**, right→wrong **3**, net **-2 / 32 = -6.25 percentage points**. **REJECT as a champion / insufficient evidence for generic claim**; one case is not a robust proof, and there are real regressions. Original independent semantic score still undefined. Used DEV, not reserved HOLDOUT300; this is not new-DB generalization. All original 32 task IDs exposed now. Baseline rates across distinct populations (frozen100 vs new32) must NOT be compared as a treatment effect.

## Newly completed original-source BIRD TRAIN validation64 (2026-11-16)

Source data CI: https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43888747753 — original publisher BIRD TRAIN SQL checked against pinned physical source SQLite, 242 compiled training candidates from 16 DBs and 64 compiled validation candidates from 4 **different** DBs, 14 prospective training references excluded by preflight. This verifies *compilation*, not the semantic truth of labels. SHA256 TRAIN: b6851fb0f2b5027833aff45a0b5e01326359b245be1c812edc66986d0df1d402; validation: cc112305bc9de33a03cea704f2edecfdfb45e62cc953ae498cea27fbed93b506. No BIRD DEV database overlap, no DEV300 consumption.

Physical model CI: https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43892823922 — all eight Qwen3.5-9B original Direct + Guarded shards and fail-closed strict aggregate SUCCESS, complete 64/64 same source-verified questions across 4 physical databases. Strict set-EX **27/64 Direct→31/64 Guarded**, 4 wrong→right, 0 right→wrong; Guarded bag-EX 28/64; execution success **46/64→54/64**. Not an adapter arm and not a new semantic judge score. Both original paths were matched to pinned digest 56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90. Do not compare absolute 31/64 BIRD TRAIN vs 39/100 DEV as an intervention. Four databases were source-disjoint from 16 TRAIN databases, but not an independently authored external corpus or hidden test.

SQL QLoRA32 new workflow: https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43895721569. Launch does not prove optimizer execution or accuracy. Maintain original fixed base. Require physical trained adapter hash, matched inference on all same 64, paired rescues/regressions and no holdout leakage before any claim.
