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


## Latest SQL research discrimination (2026-11-16)

[Physical Qwen3.5-9B VAL64](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43892823922): original-source DB-disjoint BIRD TRAIN validation64 had **31/64 Guarded strict set-EX**, **54/64 Guarded execution success**. It follows exactly that **23 executable-but-wrong** and **10 nonexecuting** cases remain. **23/33 (~69.7%) of strict-EX failures are not solved by mere SQLite compilation/execution.** This is a measurement partition, **not a human-verified root-cause taxonomy**; further posthoc diagnostics must distinguish projection, joins, filters, literal grounding and grain without treating lexical flags as causal evidence. Existing Guarded +4/64 rescues over Direct are original-scaffold effects, not a new adapter uplift.

[Physical SQL QLoRA32 attempt](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43895721569): source SHA verified 242 original BIRD TRAIN items across 16 DBs, full-context 4096 fit **242/242**, original fixed Qwen3.5-9B model loaded. **FAILED at GradScaler BF16 unscale on NVIDIA T4 before any verified optimizer step**, with no checkpoints/final adapter, archived `partial_failure` proof. Distinguish code/trainer precision failure from model quality. Do not report improvement or schedule a blind repeat. Any rerun requires a corrected precision path verified on T4 and prospective paired strict EX against the unchanged frozen 64 original tasks, independent semantic-score contract still missing.

## Physical SQLite implicit FK correction: positive data integrity, negative paired EX (2026-11-17)

- [CPU source SHA-checked run](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43904292207): BIRD TRAIN original validation64 had 284 malformed `.None` FK target occurrences in 32/64 fixed input prompts; actual parent primary keys resolved all 284, unchanged question/text schema columns and FK-count; 9 tests passed. This is structural correctness only.
- [Immutable prospective cohort freeze](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43904889296): prior-unseen physical BIRD TRAIN source databases shakespeare, movies_4, car_retails, books (8 questions per DB, 32 cases), no overlap with earlier 242 TRAIN/64 VAL databases or 11 DEV databases; source labels preflight validated. Cohort SHA `ee1b5035302d36a1241952bd2e5acd3dfc2518d1fba2581f557e04e179960535`. Public source, not hidden/corpus-independent.
- [Completed real matched Qwen3.5-9B inference](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43905918715): 8/8 physical inference shards SUCCESS and strict 32/32 aggregate. Same fixed digest, system, model, temperature0, seed42, num_predict768, 2 repairs. **Guarded strict set-EX 19/32 original FK context vs 18/32 corrected FK context**, 0 rescues / 1 regression, net **-1/32 = -3.125 pp**. Direct 19→18, Guarded execution success 31/32→31/32, model calls 37→37. By DB, books 5→5; car_retails 4→3; movies_4 6→6; shakespeare 4→4. Regression source index 1097 arose from response changing from ORDER BY/LIMIT 1 to MIN-subquery that returned more dates; corrected FK edge not directly used by either query. This is a paired model response difference, not evidence FK target names mislead semantically in general.
- **Decision before score:** net EX change <=0 means reject as a performance intervention. [Candidate PR #78](https://github.com/kolihhan/semantic-text2sql-agent/pull/78) closed unmerged. Keep confirmed schema integrity bug for separate engineering work, but do not promote it as an EX improvement, nor claim semantic metric improvement or independent BIRD DEV success. Source label semantic quality has not been human adjudicated.


## BIRD TRAIN source-only physical value grounding: paired outcome and reproducibility (2026-11-17)

Original new-DB freeze [#43920461939](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43920461939): SHA `4de8471341b38d0fb00ded1f17f265d33b528ec78b9779eb94937914df196955`, 16 source questions (4 each from soccer_2016, university, cs_semester, hockey), four physical source SQLite DBs never used in the 242 SFT or 64 VAL or 32 FK experiments; BIRD DEV excluded, labels compile-checked. Exact source entity→column values from read-only SQLite, no reference answers consulted for retrieval. [First actual 8/8 physical Qwen3.5-9B shards](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43921074189) original Guarded **9/16** set-EX → entity grounding **11/16**, 2 rescues/0 regressions; execution 15→16. [Independent rerun of same 16/source/model](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43924579052) original **8/16** → grounded **11/16**, 3 rescues/0 regressions, execution 15→16. Same pinned model digest `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`, seed42, temperature0, 2-repair Guarded; source unchanged.

**A/A repeat instability:** original source #5750 (hockey) changed from `SELECT MAX(M.height)...` (original success in first run) to a `SELECT height... ORDER BY height DESC LIMIT 1` expression (original failure second run), while grounded remained correct. Do not treat +3 on repeat as independent holdout gain or as meeting original predeclared promotion gate after peeking. Original +2 did not clear its net>=3, 0 regressions and >=2 rescued DB threshold.

**Mechanism audit:** #590 (cs_semester) question requests percentage of *Professor Ogdon Zywicki's* research assistants. Original erroneous WHERE filters `student.first_name/last_name`; physical evidence identified values in `prof.first_name` and `prof.last_name`; new SQL correctly filters professor and passes. #6032 (university) question requests university name **and university ID** in Turkey. Both original/new WHERE already filter `country.country_name='Turkey'`, but old SELECT returns `country_id` instead of `university.id`; new projects correctly. The latter improvement is **not clearly caused by value grounding**. One credible mechanism-aligned rescue out of two original, no model training. Publisher SQL semantics are not human adjudicated, all public BIRD TRAIN.

**Second independent source cohort** [freeze run #43926576493](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43926576493): 16 questions from four different further DBs chicago_crime, cookbook, shooting, retail_complains, excludes all 28 old BIRD TRAIN test/training DBs and 11 BIRD DEV DBs, SHA `faa4f7ba1de9923cbf8e51a500727dfe4c76a08c3f6366735cf21bff152655b7`, 25 source value→column hits. [Matched fixed Qwen9B replication in progress](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43927133612). No score claim until all 8 shards and complete 16-case aggregate; require net>=3, zero regressions across >=2 DBs and further larger sample before promotion.
