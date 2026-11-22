# BIRD-only four-way LangGraph / K4 real EX and cost — 2026-11-23

**Verdict: current opt-in adaptive LangGraph risk routing DOES NOT IMPROVE real BIRD EX. Do not merge/enable.**

## Exact provenance and scope

- Fixed original **Qwen3.5-9B** model digest \`56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90\`.
- Identical **BIRD TRAIN16**, four **previously exposed physical BIRD DB families**: address, shipping, student_loan, university; original frozen question source SHA \`5b25f9d6972c60bf67d39cfd854c3ec4f30f960972e10705d03ef6201a06171c\`.
- Original sealed Qwen first candidate SQL reused from [K4 predecessor #45039635210](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/45039635210), **never regenerated** in Guarded/Adaptive paired inference. Both graph arms start from the **identical** SQL and use identical fixed model / schema / BIRD physical database.
- Real pinned BIRD graph inference/score [#45047859446](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/45047859446). Sealed Qwen candidates uploaded before publisher gold was downloaded/scored.
- Reconciled cost and four-arm EX in **CPU-only**, zero extra Qwen [#45048336642](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/45048336642). Final artifact \`final-BIRD-TRAIN16-Guarded-adaptive-K4-PhysicalK4-StrictEX-cost\`.
- **Neither BIRD DEV500 nor reserved HOLDOUT300 nor final two protected TRAIN DBs used.** This 16-case result cannot be compared as an official DEV100 improvement; DEV100 historical 33/100→39/100 is a separate cohort.

## Strict EX, exact same sixteen questions

| Pipeline / workflow | SET EX | BAG EX | Logical Qwen calls for 16 (all arms include source generation) | Calls / correct SET |
|---|---:|---:|---:|---:|
| **Old Guarded LangGraph:** initial SQL → preflight → syntax repair (if failed) → execute | **9/16** | 7/16 | **22 = 16 initial + 6 repairs** | **2.44** |
| **New adaptive LangGraph:** Guarded → risk decision → conditional schema/data probes → 0/1 semantic counterproposal → verify/fallback → execute | **9/16** | 7/16 | **24 = 16 initial + 8 extras** | **2.67** |
| **Ordinary K4:** four generations → safe SQLite results → gold-blind plurality | **8/16** | 6/16 | **64 = 4×16** | **8.00** |
| **Physical Evidence K4:** two shared originals + two database-conditioned alternatives → plurality | **10/16** | 8/16 | **64 = 4×16** | **6.40** |

This is a **four-way same-cohort comparison, not equal-budget four-way A/B**. All four Qwen-call totals are logical per-arm budgets, not the number of additional CPU runs for this reconciliation. The Guarded and Adaptive paired A/B alone has an identical source and budget cap except for selective tool/model calls; physical K4 treatment changes both its physical facts and counterproposal prompt, which still requires an ablation to isolate causality.

Paired **new vs old LangGraph** = **zero rescues, zero regressions**. **2/16** questions triggered physical evidence probes and both ended at original SQL through explicit fallback. With this narrow risk trigger the extra routing is cost without measured gain.

Against **old Guarded**, ordinary K4: 0 rescues/1 regression; Physical K4: 1 rescue/0 regression. Against **ordinary K4**, Physical K4 remains +2 on this pilot. The original +2 number **must not** be reported as a +2 improvement over old Guarded.

## Actual measured inference time (not inflated into fake end-to-end totals)

| Cost measurement | Guarded | Adaptive | K4 | Physical K4 |
|---|---:|---:|---:|---:|
| Post-first SQL Qwen calls | 6 | 8 | — | — |
| **Post-first SQL Qwen inference wall**, measured | **200.04 s** | **223.62 s** | — | — |
| **All K4 SQL generation wall**, measured | — | — | **809.61 s** | **1749.42 s** |
| Difference vs relevant control | reference | **+2 calls, +23.58 s (+11.8% post-first model wall)** | reference | **+939.81 s (2.16× K4 model wall)** |

**Missing measurements are explicitly UNKNOWN**, not zero: *individual initial first-SQL generation time*, physical SQLite schema/data probe wall time, entire end-to-end request latency, token usage and monetary GPU/API cost. The prior K4 result only recorded **sum of four generations** per arm, so dividing that sum by four would invent a timing. Actual GitHub runner execution time is distinct from model inference wall time and is not an API price.

## Error mechanisms / physical DB distribution

| BIRD DB | Guarded | Adaptive | K4 | Physical K4 |
|---|---:|---:|---:|---:|
| address (4 questions) | 3 | 3 | 3 | 3 |
| shipping (4 questions) | 2 | 2 | 2 | 2 |
| student_loan (4 questions) | 1 | 1 | 1 | 2 |
| university (4 questions) | 3 | 3 | 2 | 3 |

Physical K4 positive screen is concentrated in **student_loan** and **university** vs K4, across previously exposed physical DBs. The Graph's selective hard-coded structural risk gate reached too few instances to rescue any of the 7 Guarded misses; it is NOT demonstrated to detect semantically wrong but executable SQL on fresh BIRD cases.

## Decisions

1. **Do not merge/default-enable** this adaptive risk router; no measurable EX gain, higher Qwen cost.
2. **Do not blindly rerun K4 / build larger agent stacks**; the physical K4 improvement is exploratory and costly. Use a matched-prompt no-physical-evidence ablation before attributing gains to actual SQLite data.
3. If continuing LangGraph semantic research, identify **independent, higher-coverage source-grounded semantic contradictions** with demonstrated right-case control and prospectively frozen BIRD-only data, then route *only actionable* cases. Otherwise keep the old Guarded baseline.
4. Do not consume protected BIRD held-out groups, change models, train SQL models, import Spider, or claim new DEV100 results.
