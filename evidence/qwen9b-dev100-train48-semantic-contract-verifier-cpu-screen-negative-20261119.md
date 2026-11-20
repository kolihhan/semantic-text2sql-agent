# Text2SQL semantic-contract verifier — deterministic opt-in experiment, complete CPU screen (2026-11-19)

## Owner goal, actual runtime and boundary

The real default project architecture in `main` remains **question + schema → direct Qwen SQL → deterministic SQL preflight → bounded ≤2 model repairs on verifier failures → read-only SQLite execution**. There is **no always-on semantic planning component**. Prior generic LLM semantic-revision experiment historically regressed strict EX **39/100→38/100**. This experiment **does not replay a generic LLM reviewer** or create K2/K4/K8 samples.

This draft adds `semantic_contract_checks=False` to the existing `SemanticSQLService` and `run_guarded`. With its default **False**, existing graph nodes, return values, prompts, model calls and read-only execution behavior are unchanged. With explicit opt-in **True**, only high-confidence deterministic contradictions are appended as `semantic_*` verifier issues that use the **existing** bounded repair node. Uncertain join/fanout and missing-relation signals are **advisory only**; they never force an LLM call, rewrite, or rejection. The check reads **question and proposed SQL only**, never gold SQL, correct answer or publisher reference result.

## What the new code recognizes

**Blocking, only under explicit opt-in:**
- `growth_rate_not_computed` — question asks for a growth *rate*, outer SELECT returns arithmetic components without computing a division, absent an actual stored growth-rate field. Historical #169.
- `age_wrong_1900_reference` — requested age grouping, expression computes `birth_year - 1900` rather than age at applicable reference year. Historical #96.
- `two_year_season_collapsed` — explicit season `2015–2016` collapsed to equality `season=2015` with second year not represented. Historical #1142.

**Advisory only:**
- Possible target entity COUNT grain vs joined child rows. Historical #325 and #298.
- Connectivity question missing explicit relation. Historical #217.
- Explicit entity names plus IDs requested while SQL selects only ID. Historical #430.

The detector parses the **outer** SELECT projection with quote and subquery-depth handling, abstains on unsupported `WITH`/CTE queries, and does not confuse inner denominator `FROM` with outer `FROM`. This is still a conservative lexical detector, NOT formal semantic proof. Contract signals cannot prove the right SQL, even if a candidate passes.

## Original physical replay and immutable artifact

**Actions run [#44527211140](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44527211140) completed SUCCESS; all **101** repository tests PASSED**, including real local read-only SQLite synthetic end-to-end tests proving that the existing repair loop corrects a growth-rate query when explicitly opted in, and baseline with default off executes unchanged. Official artifact: `final-original-qwen9b-semantic-contracts-cpu-DEV100-TRAIN48` (ID `14531197226`).

- Exact historical Qwen3.5-9B model digest: `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`.
- **Historic frozen BIRD DEV100:** 4 original original-prediction shards [run #37307108955](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/37307108955), source SHA `0310b7ca2dfb6247234e2073f93e7a75106ee41ba55231f75d06d2ef68efe2ec`. **3 shards COMPLETE, 1 original shard INVALID**. The original full workflow failed; the 39/100 score is a historical stored result, **NOT cleanly rerun** here. Do not promote it to a new score.
- **Separate, also previously exposed BIRD TRAIN48:** 8 original physical shard SQL/EX records [run #44508116418](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44508116418), source SHA `006773fbd40894ba202c9d31aa54e0af95bcbd7f50e82d65871b36b7e2b89fc0`, **46/48 publisher-scorable**; source-K4 context is an archived prediction cohort, not a new K experiment. Neither source is independent unseen validation.

| Original immutable observation | Historic DEV100 Qwen Guarded | Already exposed TRAIN48 selected SQL (direct) | Already exposed TRAIN48 alternative |
|---|---:|---:|---:|
| Source rows | 100 | 48 | 48 |
| Original correct strict EX | **39/100** | **26/46** | **25/46** |
| Blocking semantic signals found | **3** | **0** | **0** |
| Advisory signals found | **4** | **0** | **0** |
| Correct originals flagged (blocking or advisory) | **0/39** | 0/26 | 0/25 |

In the 100 historical cases, **7/61 wrong** scored cases raised one semantic signal, while **0/39 correct** cases raised one. This association is highly vulnerable to **posthoc rule construction from the very same errors** and is **NOT an independent sensitivity/specificity estimate**. The other **54/61** wrong historical cases were **not detected**. The separate TRAIN48 had **no triggers in either arm**, so it provides **no independent positive evidence** of semantic verification utility. No model-generated corrections or real EX improvement measured.

**Predeclared CPU screening gate FAILED:** requires zero known-correct blocking flags on historic DEV100 (observed zero), at least five detected historic wrong actionable cases (observed **three**), and at least three actionable wrong cases on the distinct TRAIN48 cohort (observed **zero**). `positive_screen_only=false`, `promotion_allowed=false`.

## Decision

**STOP. Do NOT enable semantic contracts by default, merge this experimental code, or spend more Qwen inference just to fix these 3 remembered examples.** Original historical **39/100** baseline unchanged; no new Execution Match gain, no new model weights/training, no new candidate draws, no use of reserved BIRD HOLDOUT300/DEV500. The synthetic 0.5 growth-rate answer confirms bounded-control-loop plumbing, not benchmark accuracy.

For a next *distinct* mechanism, prioritize failure classes the current source-only detector could not inspect (wrong join/role, denominator/grain, implicit entity relation, value grounding). A schema-grounded, source-evidenced semantic plan/representation may be appropriate only if it adds real information rather than restating the question. Freeze genuinely new source identities and verify **physical strict EX**, paired regressions, and nondegenerate DB-level uncertainty before promotion.
