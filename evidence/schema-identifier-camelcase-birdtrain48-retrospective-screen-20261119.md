# Text2SQL Schema Linking — original CamelCase tokenizer defect; **real CPU source screen below gate** (2026-11-19)

## Code-level verified bug

Current default evaluation tokenizer in `evaluation/schema_context.py` does `re.sub(r"([a-z])([A-Z])", ..., value.casefold())`: it **lowercases the identifier before detecting a lower→upper boundary**. Therefore `workerBirthDate` is tokenized as one `workerbirthdate` term, not `worker`, `birth`, `date`; `QuestionID` and `CBSAName` have analogous issues. This is a deterministic bug in lexical schema **recall**, NOT proven to be the main cause of BIRD semantic EX errors.

The experimental `evaluation/schema_identifier_cpu_audit.py` leaves the original `select_schema_context` **bit-identical**, changes **only** identifier boundary tokenization, and leaves all matching weights, fallback behavior, FK-neighbor expansion, model prompts, SQL generation, verification and repair untouched. CI tests prove exact original-context replay and synthetic CamelCase correction. Production/default code is **UNCHANGED**.

## Strict evidence provenance and result

- Source: EXACT pre-model immutable BIRD TRAIN48 question-only cohort from original fixed Qwen K4 [Actions #44508116418](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44508116418), SHA `006773fbd40894ba202c9d31aa54e0af95bcbd7f50e82d65871b36b7e2b89fc0`. The 48 TRAIN questions and 8 physical DB families were already exposed in prior work; this is **retrospective** and not an independent new test.
- First original source-only [Actions #44518009961](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44518009961) fully completed the 8 physical SQLite source checks and **uploaded a sealed, label-blind schema-context A/B artifact** `original-TRAIN48-IDENTIFIER-BOUNDARY-GOLD-FREE-sealed` ID `14525819253` **before** reference-SQL scoring. The first scoring job did compute and print the full actual result, but was marked failed by an overly strict assertion accessing absent optional zero-valued `Counter['parse_failed']` (zero parse failures). No source results changed.
- CPU-only recovery of the original sealed schema candidates and originally pinned publisher SQL: [Actions #44518435873](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44518435873), **SUCCESS**. Official immutable final artifact `original-TRAIN48-IDENTIFIER-BOUNDARY-CPU-salvaged-final`, ID `14526432773`, unzipped JSON SHA-256 `49b72f51fab658ff488e95ba5bbab468768de0470c5c09ffb4c9da7e1a4829a2`. No re-download of eight physical DBs, no re-selection, no Qwen inference.
- All 48 pinned original publisher BIRD TRAIN SQLs were parsed using SQL AST, **0 parse failures**. The scored metric is only whether **all real publisher-referenced physical tables appear in the selected schema context**; this is a necessary condition, NOT SQL correctness / model execution match.

| Measured physical AST table-coverage diagnostic | Old default lexical schema context | Corrected CamelCase-only context |
|---|---:|---:|
| Original publisher SQL's **all referenced tables included** | **45/48** | **47/48** |
| Individual publisher SQL table mentions covered | 94/97 | 96/97 |
| Table-coverage rescues / regressions | — | **2 / 0** |
| Schema contexts that changed (no model calls) | — | **4/48** |
| Total emitted schema context UTF-8 bytes | 32,912 | 32,912 |

**Exact source IDs for two gained coverage cases:** `mental_health_survey` source **#3588 and #3590**. Original context omitted publisher-required `Question` table; corrected context includes it. **Two changed no-gold cases** `computer_student` #720 and #710 removed `course` from selected schema; original publisher SQL did not need that table, so observed reference table coverage did not regress. These are retrospective case observations, NOT separate randomized causal evidence. All original DB schemas, SQL labels and question IDs remained frozen.

## Promotion gate, decision, and what to do next

**Predeclared minimum CPU follow-up gate** required **≥3 new fully covered reference-table cases across all 48**, zero regressions, complete reference parsing and ≤1.20× mean context bytes. Actual: **2 new, 0 lost, all48 parsed, ratio1.00**. **FAIL**, `positive_screen_only=false`, `promotion_allowed=false`.

**Decision:** retain opt-in tokenizer as a tested code-level correction, **do not touch production Schema Linking nor launch more Qwen candidate draws** for this alone. The outcome demonstrates a real input-schema-visibility bug and only modest retrospective table-coverage headroom. It does not show any improvement in Qwen strict SET/BAG EX, semantic-plan quality, true unknown-database generalization, or repair correctness.

Return focus to the actual default pipeline `question+schema → direct SQL → deterministic preflight → ≤2 bounded repair → read-only execution`. Structured semantic planning in research branches is **not currently a production step**. For future mechanism selection, first use real failed cases to distinguish *schema visibility*, *wrong relationship/metric/filter/output*, and *correct SQL lost at selection*, with actual physical SQL outcome and no publisher-target leakage. Keep fixed Qwen3.5-9B research model, all protected BIRD DEV500/HOLDOUT300 and reserved BIRD TRAIN database families unmodified.
