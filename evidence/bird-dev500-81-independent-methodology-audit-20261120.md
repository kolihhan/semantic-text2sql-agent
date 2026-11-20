# Independent adversarial audit — DEV500 entity-grain selected-case claims (2026-11-20)

## Verified experiment
- Original source-first study [Actions 44555827205](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44555827205), 81 fully executable wrong K8 cases across 11 physical databases.
- New source-first then publisher-postblind [Actions 44569182407](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44569182407), immutable artifacts `14553596411` and `14553771540`: #219 toxicology, #571 codebase_community, #1080 european_football_2; original 0/3 set-result match, manual unit-targeted revision 3/3, no added Qwen calls.
- Prior selected count-grain eight cases: 2/8 repairs. Combined 5/11 source-selected manual repairs of K8 zero-oracle tasks. Not a deployable result or matched Guarded pipeline EX.

## Logical gaps that must not survive into any paper or README
1. **Zero observed regressions is tautological:** all 11 diagnostic tasks were drawn from *known first-original-Qwen wrong cases*. A correction cannot regress an already-wrong original under the binary metric. Thus 0/11 regressions gives **zero information** about the false-positive/regression rate on originally-correct SQL. Independent correct controls are essential.
2. **Three databases do not establish one cross-DB causal rule:** the three proposed transformations differ categorically: conditional distinct-molecule ratio, independent post/vote subquery ratio, and distinct Player_Attributes identifier. Code is keyed by public DEV **question ID**, an oracle/manual prototype that cannot be installed as general inference-time guard.
3. **Source-first does not repair selection bias:** the gold result is opened only after each proposal was frozen inside this Actions run, but all task IDs were already outcome-conditioned on K8 failure, and many prior public DEV investigations have visited the same 11 DBs. No prospective accuracy gain can be estimated.
4. **Publisher SET result equality is not full end-to-end BIRD Guarded EX:** direct original SQL and hand-changed SQL were tested with the read-only local 8s capped, order-insensitive set-hash scorer. The actual Guarded model-generation/verifier/max2-repair path and its costs were not rerun.
5. **Answer-equivalent output is not necessarily uniquely correct reasoning:** incorrect semantic logic can coincidentally match one frozen SQLite instance. Validate the role/unit/aggregation independent of the publisher SQL and test adversarial physical instances for proposed rule.
6. **The 81 source-first intent classes remain provisional:** 24 AGGREGATION_GRAIN, 26 RELATION_FILTER, 13 RANK_SCOPE, 9 OUTPUT_SHAPE, 5 VALUE_TIME, 4 AMBIGUOUS in the assistant's question-first local ledger; these are hypotheses from questions/SQL, not blinded expert causal labels or interannotator agreement. Source-case distribution is unequal by DB (e.g., 18 card_games, 13 formula_1, 12 toxicology); don't extrapolate case counts to benchmark prevalence.

## Decision / stop-loss
- Keep the default Qwen3.5-9B and existing Guarded unchanged; no SQL QLoRA.
- **No promotion** of global COUNT(DISTINCT), broad relation rewrites, or manually ID-keyed proposals.
- Before any expensive inference, write one generic, **question-ID-agnostic** question-target+schema-PK/FK+SQL detector and freeze its source-only activation rate on a predeclared, question-independent sample of both original-right and original-wrong SQL, never using publisher reference for the trigger.
- Require nonzero trigger, low correct-control false-positive rate and genuine physical join-multiplicity/denominator evidence; then compare the *same original Qwen SQL and max2 repairs* prospectively on fixed BIRD set EX with both regressions and rescues, 11-DB clustered uncertainty and actual runtime. Reserve HOLDOUT300.
- If those prerequisites are not met: **stop, not another hand-picked 3/3**.

This memo is a methodological critique, not new performance evidence.
