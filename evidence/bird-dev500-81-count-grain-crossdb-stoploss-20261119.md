# BIRD DEV500 / 81 fully executable wrong SQL: count-grain diagnostic and STOP gate (2026-11-19)

## Frozen source / denominators

- Fixed Qwen3.5-9B digest: `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`.
- Source-only [81-case full physical schema audit](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44555827205), artifact `14546155987`, SUCCESS: **500 DEV source questions, 215 zero-oracle K8, 97 fully executable and all eight wrong, 16 earlier-pilot IDs excluded, 81 remaining across 11 physical DBs**. All 81 question texts and first original Qwen SQL are source-first with per-table schema and keys. `human_fk_and_entity_grain_review=PENDING` for all 81; do NOT pretend this source census is a complete human annotation.
- 81 cases mean **648 prior real model candidate SQLs**, reused; zero new model calls. HOLDOUT300 untouched. Previous full DEV K8 original plurality remains **233/499 scorable** (case 1126 censored), not a new Guarded score.

## The only eight predeclared, physically checked count-grain substitutions

The first five were sealed source-first before publisher SQL was opened; later three were independently sealed in three other DBs before scoring. Every row compares the first ORIGINAL Qwen candidate SQL to a single predeclared COUNT/grain substitution; not an evaluated Guarded+2-repair path.

| Case | Physical DB | Original→single grain candidate | Rescue | What the evidence actually supports |
|---|---|---|---|---|
| 109 | financial | wrong→wrong | 0 | `COUNT(*)` and `COUNT(DISTINCT client_id)` both returned 25. Problem is not this join-duplication correction |
| 1028 | european_football_2 | wrong→wrong | 0 | Count of away wins by joined Match→Team unchanged. Ranking/season/filter interpretation is a competing cause |
| 1256 | thrombosis_prediction | wrong→right | 1 | `COUNT(*)` returned **208**, `COUNT(DISTINCT P.ID)` returned **25** and matched publisher result. Genuine repeated laboratory-row vs patient entity-grain error on this source case |
| 1289 | thrombosis_prediction | wrong→wrong | 0 | Both counts returned 0; a DISTINCT rewrite cannot repair the selection/filter |
| 1441 | student_club | wrong→wrong | 0 | Both counts returned 8; no observed duplicate amplification |
| 463 | card_games | wrong→wrong | 0 | DISTINCT set-translation IDs changes physical result but still misses publisher answer; identifying set vs translations remains unresolved |
| 593 | codebase_community | wrong→wrong | 0 | DISTINCT users changes result, but requirement **both Teacher AND Supporter** is not proved by `IN ('Supporter','Teacher')` (OR semantics) |
| 1525 | debit_card_specializing | wrong→right | 1 | Transaction-row weighted percentage corrected to distinct-customer unit, matching publisher result. This is a different physical DB from #1256 |

**Matched 8/8 retrospectively scorable**; original selected first SQL 0/8, source-only replacements 2/8, 2 rescues/0 regressions. This is **not an overall DEV500 or prospective BIRD EX improvement**: the eight cases were chosen diagnostically from known K8 failures, so sampling is outcome-conditioned. It does not establish a positive effect on new tasks.

Physical runs and immutable artifacts:
- [Five source-first, originally no gold](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44555827205) — `14546155987`
- [Five postblind scored](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44557368301) — `14545991824`: 0/5→1/5, 1 rescue, 0 regressions
- [Three extra independent DB source-first → postblind](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44559280649) — `14547555579`, `14545428694`: 0/3→1/3, 1 rescue, 0 regressions

## Actual decision

**Do not patch existing Guarded to apply global COUNT(DISTINCT).** Two cases across two DBs show the *possibility* of entity-grain correction; six observed counterexamples reject an unqualified rewrite, and differing join structures mean there is no tested, well-calibrated trigger. Existing correct controls #168 (financial), #1091 (football), #1305 (thrombosis) further warn that COUNT(*) on a JOIN is not inherently an error.

A future change is permitted **only** after:
1. Finish source-first entity/FK annotations for the full 81 and correct same-DB controls, distinguishing person/transaction, event/match, and filter/conjunction failures.
2. Define one falsifiable structural trigger without gold or held-out label access, then freeze a question-disjoint matched inference cohort with the exact original Qwen SQL, identical two-repair budget and official-compatible scorer.
3. Run full paired BIRD SET EX, show per-task rescues/regressions, 11-DB cluster uncertainty, and actual compute. Reject on no measurable EX advantage, even if new SQL compiles or returns rows.

**Current deliverable:** source evidence and falsifying controls. New Guarded code **unchanged**; new Qwen calls **0**; new official BIRD matched Guarded EX **not run**; no adapter/QLoRA.

