# Source-first, CPU-only four-BIRD-database typed entity-unit and denominator falsification — 2026-11-20

**REAL PINNED DATABASE COUNTS VERIFIED. This is NOT a new model experiment or generalizable repair rule.**

## Evidence and isolation

- [Physical GitHub Actions #44679698194](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44679698194), successful. Final source-only artifact ID **14602933750**, `BIRD-DEV500-FOUR-DB-ENTITY-GRAIN-DENOMINATOR-SOURCE-PHYSICAL`.
- Immutable original question/SQL and physical schemas: [original 81 source-only #44555827205](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44555827205), artifact `14546155987`. Fixed original Qwen3.5-9B digest `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`. Physical SQLite archive SHA256 `cdd6d19faeb45a23970b98d3ef6c40a87987c95459c2cf12076897a60cf5a630`.
- Read-only, 8-second maximum per predeclared aggregate SELECT and explicit actual PK/FK introspection. Four cases selected from known DEV errors, not random/prospective; script uses question, original first SQL and actual database, but **does not read publisher gold/reference SQL, retrain, regenerate, invoke Qwen, touch HOLDOUT300 or change Guarded**.
- This is a physical distinction between entity/grain units, NOT a scored correction or actual causal task A/B.

## Measured primitive counts: physical keys, relationships, and counterexamples

| Case / database | Physical relationship and target | DB aggregated facts | Interpretation |
|---|---|---|---|
| **#219 toxicology** | `bond.molecule_id → molecule.molecule_id`, bond row vs molecule unique entity | 6 triple-bond rows / 4 unique triple-bond molecules. Of those, 3 carcinogenic triple-bond rows / 2 carcinogenic unique molecules | **JOIN fanout exists, BUT both alternative proportions are 50% (3/6=2/4).** Thus changing entity unit may have *no numerical effect* on this frozen dataset. Original first SQL also conditions `m.label='+'` before computing a reversed bond proportion; ratio-scope error distinct from fanout |
| **#571 codebase_community** | `votes.UserId → users.Id`, `votes.PostId → posts.Id`, posts ownership separate from a user's cast votes | User24 made **3 posts**, cast **8 total votes**, but only **1** of those votes was cast on a post owned by user24; 29 votes by any user on their posts | Source join restricts `votes` to own posts; this loses user-level vote activity. Distinct count alone cannot fix wrong relationship scope |
| **#1256 thrombosis_prediction** | `Laboratory(ID,Date)` composite PK, `Laboratory.ID → Patient.ID` | **208 matching lab rows vs 25 unique matching patients** | Strong physical row-vs-entity grain mismatch; this case had earlier case-specific source repair evidence, **not** independently transferable correction |
| **#1525 debit_card_specializing** | `transactions_1k.TransactionID` vs `customers.CustomerID` | **425 transaction rows vs 259 customers** on date; EUR subgroup 7 rows / 7 customers | Transaction-weighted EUR rate **7/425≈1.65%**, distinct-customer EUR rate **7/259≈2.70%**. Actual denominator choice matters; numerator happens not to fan out for this subgroup |

Three additional pre-existing read-only invariant probes (#109 financial, #1028 european_football_2, #1441 student_club) had `COUNT(*) == COUNT(DISTINCT target key)`. **These were selected K8-WRONG controls, NOT originally correct cases.** They show that blindly applying DISTINCT often changes nothing. No false-positive-regression rate can be derived without K1/K8-correct controls on independent source.

## Independent critique and next research gate

1. Four of four **preselected** source errors exhibit a different *primitive* count/relationship population. This is **not 4/4 correction effectiveness**, and especially #219 demonstrates a real physical fanout but a numerically unchanged 50% ratio under the two relevant alternative sample spaces.
2. These are different mathematical failure mechanisms under a broad shared heading. #571 is relationship/scope restriction, while #1256 is row multiplicity; #219 denominator and cohort reversal; #1525 transaction-versus-customer weighting. A single `COUNT(DISTINCT)` regex would not solve them.
3. **P1 condition partially met:** real public FK/PK and numeric evidence across four distinct physical DBs; enough to propose a distinct **typed answer unit + FK path + population grain** test hypothesis. **Not enough** to declare one generic intervention verified or to launch QLoRA.
4. Do not repeat PR #116 generic structural counterhypothesis prompts (46 scorable TRAIN questions: selected 26→25, 2.16× wall clock), nor PR #100 voting/tie tweaks (45/52 wrong vote plurality).
5. Before new Qwen generation A/B: freeze a question-ID-agnostic mechanism that identifies answer entity/population/denominator from the question and schema **without any gold**, demonstrate physical type-checks with correct original SQL controls and independent TRAIN schemas, and stop if the trigger cannot distinguish correct from incorrect original SQL. Otherwise one more source-selected generator run may be another overfit to DEV.
6. **For any subsequent independent A/B** keep Qwen3.5-9B, candidate count and latency matched, compare new correct candidate oracle and GOLD-BLIND selected strict SET/BAG EX, rescues/regressions, DB clustering, and costs. HOLDOUT300 sealed. No source SQL hand-edits as claim of a reusable algorithm.

**Decision:** source-only typed entity-unit hypothesis survives as a research direction, but **no generic source-independent SQL transform or new benchmark improvement is validated**. Do not merge into Guarded.
