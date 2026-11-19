# Original Qwen3.5-9B equal-K4 structural counterhypotheses — **complete negative**, 2026-11-19

## Immutable evidence and integrity
- Official complete [GitHub Actions #44508116418](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44508116418): **SUCCESS**, 1 source freeze + **all 8** physical model/SQLite score shards + `strict-complete-final` SUCCESS.
- Official final artifact: `final-original-Qwen9B-K4-structural-counterhypothesis-TRAIN48` (artifact ID `14522482810`), JSON filename `final-Qwen9B-K4-counterhypothesis-TRAIN48.json`, JSON SHA256 `5cf413e31085aa347051768e13062ebd9e9a482ad834f7384049d1628274a6b8`.
- Question-only source SHA256 `006773fbd40894ba202c9d31aa54e0af95bcbd7f50e82d65871b36b7e2b89fc0`; original model digest `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`.
- **48 new BIRD TRAIN question identities** across 8 previously exposed physical database families. Original 711 known prior TRAIN indices excluded. Last two unexposed original physical BIRD TRAIN databases protected. No BIRD DEV500 or HOLDOUT300 opened, no training, no teacher/model changes. Database-disjoint generalization is **NOT** established.
- Equal **effective** 4 original Qwen3.5-9B calls per arm; two seed-42/43 candidates are shared; baseline other seeds 44/45 are independent normal generation, treatment other seeds 44/45 receive counterhypothesis-specific instructions informed by the first two raw SQL outputs. Total **288 unique physical Qwen model calls** across all 48 cases (not 384 unique calls). Original model and per-arm call budget are matched, but *latency is not*.
- Both K4 arms' candidate SQL, actual read-only SQLite result signatures, and selected candidate identifiers were uploaded as immutable artifacts **before** publisher reference SQL was fetched and scored. Selector: physically executable result-set plurality, earliest valid result on tie. No gold-informed candidate ranking.

## Strict physical outcome — 46 scorable of 48
Two publisher reference SQL queries (`shooting` original source IDs **1804, 1807**) failed with `OperationalError`. They are *unscorable* reference outcomes, NOT model regressions. The other 46 questions are scored on exact physical SQLite SET/BAG result matches.

| Metric | Control: 4 direct independent SQLs | Treatment: shared 2 + 2 structural counterhypotheses | Difference |
|---|---:|---:|---:|
| **Candidate hindsight Oracle@4 SET EX** (NOT deployable) | **32/46 (69.57%)** | **33/46 (71.74%)** | +1 |
| **Gold-blind selected strict SET EX** | **26/46 (56.52%)** | **25/46 (54.35%)** | **−1** |
| **Gold-blind selected strict BAG EX** | **24/46** | **24/46** | 0 |
| Oracle paired rescues/regressions | — | **2 / 1** | net +1 |
| Actually selected paired rescues/regressions | — | **1 / 2** | net **−1** |
| Mean coarse SQL-structure classes across 48 | 1.875 | 1.979 | +0.104 (not SQL semantic validity) |
| Executable candidate slots across 192 effective slots/arm | 161/192 | 163/192 | +2 |
| Total generation wall-seconds measured per arm | 2,971.1 | 6,420.2 | **2.16× treatment latency** |
| Matched effective model calls across 48 | 192 | 192 | 0 |

- Gold-blind selected **database-cluster bootstrap 95% interval** for treatment minus control: **[−0.125, +0.07143]**; mean −0.02174, across 8 original physical DB clusters. Uncertain, no robust positive effect.
- **Predeclared screening gate FAIL:** requires Oracle net ≥4 (actual +1), actual selected net ≥3 (actual −1), selected regressions ≤2 (actual 2), ≥3 rescued physical DBs (actual **1**, `shooting`), ≥42 scorable (actual 46).
- `positive_screen_only=false`, `promotion_allowed=false`. Do **not** merge/change production default SQL generator or promote candidate Oracle as deployed EX.

## Pairwise forensic: why the mechanism did not convert to end-to-end gain

- **New candidate exists but is not selected:** `mental_health_survey` IDs **3588** (users in Canada / 2018) and **3584** (physical health employer survey / 2014). Treatment K4 Oracle rescues 2 otherwise-all-wrong control K4 cases, but votes stay on the wrong familiar result signatures. For #3588, repeated candidates only filter `QuestionID=3` and Canada, while an alternative explores the required 2018 survey and different unique-user grain. For #3584, 3 candidates return the same result via equality / cast against 2014 survey description; an alternative uses `LIKE '%2014%'`. Physical Oracle improvement **does not select** the minority candidate. These are retrospective source-specific observations; not a verified general answer-selection rule.
- **Actual selected regression #4184 (`address`):** requested *cities with Micro CBSA*. Control plurality picked one-column `city`; treatment plurality tied on the first two-column `city, CBSA_name` output, causing a wrong reference result shape. This is output projection/ranking, **not** a new join-path capability gain.
- **Actual selected regression #4194 (`address`):** requested congressman name for Guanica. Control returned `first_name, last_name` as two columns, matching publisher result; treatment majority chose a single concatenated full-name column, which fails strict SET EX. Semantically plausible full-name rendering may differ from publisher representation; **do not assert its natural-language answer is wrong without independent adjudication**. It is a real official strict EX regression.
- **Candidate Oracle regression #721 (`computer_student`):** control's K4 contains a correct candidate not retained by treatment K4. This illustrates that replacing generic samples with structural candidates can lose coverage.
- **Actual selected rescue #1803 (`shooting`)**: treatment selected a correct existing candidate under changed vote balance without improving K4 Oracle for this question; selection effects matter independently from generation.
- All six shards other than these sampled examples contain complete saved scored case lists; the file above records checked causal *hypotheses* and directly observed answer-shape differences, not an independently annotated root-cause taxonomy.

## Decision / stopping rule

**STOP** scaling two additional prompt-conditioned structural samples into this exact K4 scheme. The intervention increases coarse structural diversity by only 0.10/category, produces only +1 K4 hindsight Oracle candidate net, degrades true deployed SET EX net by 1, increases source generation time 2.16× and fails multiple original predeclared gates.

Next structural bottleneck, if any further work is funded: define a **schema-grounded typed relational plan or constrained SQL abstract syntax tree** that rules out unsupported joins/projection shape before generating SQL; test its actual candidate coverage with matched model calls on a *new* source-frozen cohort. Then test a **separate gold-blind selector** when Oracle coverage gains are large enough. Neither is yet an improvement or a validated model change.

Do not use these 48 exposed items for prospective lift claims, tune selector on them and call the result an independent test, open protected holdouts, or start more Qwen calls just to chase these two rescues.
