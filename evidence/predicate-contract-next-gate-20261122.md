# Predicate-contract gate (preregistered; not an accuracy claim)

Date: 2026-11-22. Status: DESIGN ONLY / NO MODEL RUN.

## Why
The physical risk audit (PR #135, run 44930911891) found 11/13 flagged original executable queries were historically EX-wrong, but 2 were correct; this is enrichment, not semantic proof. The later normalization treatment (run 44932346821) moved 31/64 to 32/64 with two rescues, one regression and BAG decline. Reject automatic normalization. Avoid another generic semantic reviewer.

## One falsifiable mechanism
A *question-to-SQL predicate contract* records only high-confidence literal-to-column binding conflicts using SQLite schema, SQL AST and read-only parameterized value probes. It never rewrites SQL directly. If a conflict is ambiguous or legitimate zero rows cannot be ruled out, abstain. The gate measures whether a source-grounded conflict can be established *without labels*, not whether the SQL is correct.

## Frozen guardrails before implementation
- Never inspect HOLDOUT300, simpson_episodes, works_cycles, or gold SQL in detection, feature design or model prompts.
- Freeze source, database identity/checksum, question IDs and SQL candidates before scoring. Prior BIRD DEV100 and TRAIN64 are development diagnostics only, not independent transfer.
- Explicit negative controls: correct SQL with empty answer; proper-name apostrophes; valid FK-alternative joins; case-sensitive identifiers; literal absent by design; compound predicates; numeric/date literals; same text value in multiple columns.
- AST parse must resolve aliases and nested scopes. Unknown or ambiguous scope => abstain. Value probe must be read-only, bounded, parameterized, with no exported sensitive values.
- Do not auto-normalize a literal merely because a nearby stored value exists. Do not call a Qwen repair unless a conflict is both supported by independent question grounding and unique schema/value binding.
- Before any full experiment: preregister independent physical database cohort and outcome thresholds, including controls and costs. No post-hoc selection.

## Required staged tests
1. Synthetic deterministic positive/negative controls; assert no SQL mutation, abstention on ambiguity, safe SQLite read-only behavior.
2. Label-blind candidate ledger: case ID, source/db digest, AST location, signal, provenance, probe bounds, abstain reason. Keep labels sealed until predictions are frozen.
3. Independent source/DB pilot: paired unchanged Qwen3.5-9B Q4_K_M digest 56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90, same seed and Guarded policy; Strict SET EX, BAG EX, rescues/regressions, DB spread, calls/tokens/latency.
4. Stop/reject if no independent high-specificity cases, no net rescue, any unacceptable correct-to-wrong regression or BAG decline. Only then consider frozen DEV100 paired confirmation; historical SQL replay cannot substitute for fresh inference.

## Next single executable action
Implement a CPU-only AST/scope-resolved predicate-binding detector with the synthetic negative controls above on this isolated branch, then run its GitHub Actions tests. Do not wire automatic repair until the independent blind screen passes.
