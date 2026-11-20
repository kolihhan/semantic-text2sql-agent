# DEV500 81 remaining fully executable K8 failures — all eight original Qwen SQLs inspected (2026-11-20)

**Research-only assistant source-first triage, not independent human expert blind causal labels. No new Qwen calls, no new SQL execution, no reference/publisher SQL examined for labels, HOLDOUT300 untouched.**

## Physical sources and methodological boundary

- Original fixed Qwen3.5-9B model digest `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`; public DEV source digest `ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb`.
- 32 immutable original K8 generation cache shards from [Actions #44086713331](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44086713331) and [#44093103960](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44093103960), **500 unique questions ×8 SQL each**. Each cache has verified original model/source digest and `no_gold_sql_in_cache=true`.
- Source-only exact 81 question/first SQL/physical schema evidence: [Actions #44555827205](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44555827205), immutable artifact **14546155987**, `original-81-source-first-physical-schema.json`. Reconciled all **81 × 8 = 648 original SQL texts** against every source item's `original_eight_sql_sha256` and first-SQL/question SHA, with **81/81 exact matches**.
- These 81 are the 97 K8-all-executable/all-wrong cases **excluding 16 previously hand-selected pilot cases** ([PR #119](https://github.com/kolihhan/semantic-text2sql-agent/pull/119)). Therefore the evaluated cohort is **outcome-conditioned**, previously exposed DEV, and some individual cases already had source discussion (#593, #1256, #219, #571, #1525). A source-only review is not a new blind independent test even if no gold SQL is opened in this step.

## Source-only triage, NOT true causal error rates

| Judgement type | n/81 | Meaning |
|---|---:|---|
| Explicit question-versus-eight-candidate SQL mismatch | **9** | A directly inspectable omission/reversal/field mismatch repeated in the available texts |
| Provisional mechanism, needs physical schema/cardinality/value verification | **53** | Plausible source discrepancy but not a validated root cause |
| Abstain | **19** | Source question + eight SQLs insufficient; do not force a diagnosis |
| **Total** | **81** | 11 original physical database families, complete candidate-source review |

**Explicit source-observation cases:**

| Case | DB | Directly visible repeated candidate failure, without reading gold |
|---|---|---|
| #50 | california_schools | Question asks **7th** highest Math-average school; all eight candidates return **LIMIT 1**, no OFFSET |
| #219 | toxicology | Molecule-level carcinogenic/triple-bond percentage represented as bond-row conditional proportions; denominator entity not separately modeled |
| #571 | codebase_community | User posts/votes ratio computed using votes attached to user's posts, not all votes cast by that user |
| #593 | codebase_community | User must have Teacher **AND** Supporter badges; all eight use `IN('Teacher','Supporter')` (OR) and count badge rows |
| #883 | formula_1 | Requests number **and names of all races**; all eight mix `COUNT(*)` and scalar name without GROUP BY |
| #1108 | european_football_2 | Question date **2011-02-22**, all eight candidate predicates query **2012-02-22** |
| #1256 | thrombosis_prediction | Counts patients but all eight use `COUNT(*)` over Patient→Laboratory 1:N rows; previously physically confirmed one source repair |
| #1502 | debit_card_specializing | Question asks transactions **in EUR**, all eight use `Price/Amount` predicates rather than currency condition |
| #1525 | debit_card_specializing | Percentage asks for **customers** but all eight use joined transaction-row denominator; previously examined |

Important qualification: source mismatches do not prove which single change would make the publisher set result correct; e.g. both missing filters and incorrect output projection may coexist. Some cases overlap existing interventions and are **not newly discovered**.

## Source evidence ledger — complete original remaining 81

Status definitions: `SOURCE_TEXT_CONTRADICTION` = high-confidence textual conflict, **not** verified causal root; `PROVISIONAL` = requires independent physical or semantic adjudication; `ABSTAIN` = do not assert mechanism.

| Case ID | Physical DB | Status | First-pass source-only mechanism |
|---:|---|---|---|
| 40 | california_schools | PROVISIONAL | RANK_SCOPE |
| 50 | california_schools | SOURCE_TEXT_CONTRADICTION | RANK_ORDINAL |
| 109 | financial | PROVISIONAL | RELATION_PATH |
| 174 | financial | PROVISIONAL | ENTITY_ID_ROLE |
| 176 | financial | PROVISIONAL | OUTPUT_GRAIN |
| 219 | toxicology | SOURCE_TEXT_CONTRADICTION | RATIO_DENOMINATOR_UNIT |
| 225 | toxicology | PROVISIONAL | VALUE_BINDING |
| 234 | toxicology | PROVISIONAL | ENTITY_ID_LINK |
| 237 | toxicology | PROVISIONAL | OUTPUT_CONTRACT |
| 239 | toxicology | ABSTAIN | UNDETERMINED |
| 248 | toxicology | PROVISIONAL | RELATION_PATH |
| 267 | toxicology | ABSTAIN | UNDETERMINED |
| 283 | toxicology | PROVISIONAL | OUTPUT_CONTRACT |
| 304 | toxicology | PROVISIONAL | ANSWER_GRAIN |
| 315 | toxicology | PROVISIONAL | OUTPUT_CONTRACT |
| 328 | toxicology | PROVISIONAL | ENTITY_ID_LINK |
| 329 | toxicology | ABSTAIN | UNDETERMINED |
| 340 | card_games | ABSTAIN | UNDETERMINED |
| 347 | card_games | PROVISIONAL | VALUE_GROUNDING |
| 353 | card_games | PROVISIONAL | ANSWER_GRAIN |
| 367 | card_games | ABSTAIN | UNDETERMINED |
| 392 | card_games | PROVISIONAL | TEMPORAL_RANK |
| 402 | card_games | PROVISIONAL | VALUE_GROUNDING |
| 406 | card_games | PROVISIONAL | ANSWER_GRAIN |
| 412 | card_games | PROVISIONAL | VALUE_GROUNDING |
| 427 | card_games | PROVISIONAL | ENTITY_LINK |
| 428 | card_games | PROVISIONAL | ENTITY_ID_ROLE |
| 437 | card_games | PROVISIONAL | OUTPUT_CONTRACT |
| 462 | card_games | PROVISIONAL | ENTITY_LINK |
| 463 | card_games | PROVISIONAL | ANSWER_GRAIN |
| 473 | card_games | PROVISIONAL | QUANTIFIER_SCOPE |
| 482 | card_games | ABSTAIN | UNDETERMINED |
| 484 | card_games | PROVISIONAL | RANK_TIES |
| 519 | card_games | ABSTAIN | UNDETERMINED |
| 527 | card_games | ABSTAIN | UNDETERMINED |
| 565 | codebase_community | PROVISIONAL | VALUE_GROUNDING |
| 571 | codebase_community | SOURCE_TEXT_CONTRADICTION | RATIO_DENOMINATOR_UNIT |
| 584 | codebase_community | PROVISIONAL | RELATION_PATH |
| 593 | codebase_community | SOURCE_TEXT_CONTRADICTION | BOOLEAN_CONJUNCTION |
| 620 | codebase_community | PROVISIONAL | RELATION_PATH |
| 636 | codebase_community | PROVISIONAL | VALUE_GROUNDING |
| 653 | codebase_community | PROVISIONAL | RANK_METRIC |
| 685 | codebase_community | PROVISIONAL | ENTITY_LINK |
| 694 | codebase_community | PROVISIONAL | TEMPORAL_RANK |
| 715 | codebase_community | PROVISIONAL | ENTITY_LINK |
| 741 | superhero | ABSTAIN | UNDETERMINED |
| 766 | superhero | ABSTAIN | UNDETERMINED |
| 842 | superhero | ABSTAIN | UNDETERMINED |
| 847 | formula_1 | PROVISIONAL | TEMPORAL_VALUE |
| 860 | formula_1 | PROVISIONAL | VALUE_BINDING |
| 861 | formula_1 | PROVISIONAL | VALUE_BINDING |
| 871 | formula_1 | PROVISIONAL | VALUE_BINDING |
| 876 | formula_1 | PROVISIONAL | TEMPORAL_VALUE |
| 877 | formula_1 | PROVISIONAL | VALUE_GROUNDING |
| 883 | formula_1 | SOURCE_TEXT_CONTRADICTION | OUTPUT_CONTRACT |
| 898 | formula_1 | PROVISIONAL | OUTPUT_CONTRACT |
| 899 | formula_1 | PROVISIONAL | RANK_GROUP_SCOPE |
| 904 | formula_1 | PROVISIONAL | RANK_METRIC |
| 922 | formula_1 | PROVISIONAL | ENTITY_LINK |
| 924 | formula_1 | PROVISIONAL | VALUE_BINDING |
| 984 | formula_1 | PROVISIONAL | AGGREGATE_GRAIN |
| 1028 | european_football_2 | PROVISIONAL | TEMPORAL_SCOPE |
| 1069 | european_football_2 | ABSTAIN | UNDETERMINED |
| 1080 | european_football_2 | PROVISIONAL | ANSWER_GRAIN |
| 1092 | european_football_2 | ABSTAIN | UNDETERMINED |
| 1108 | european_football_2 | SOURCE_TEXT_CONTRADICTION | TEMPORAL_VALUE |
| 1143 | european_football_2 | PROVISIONAL | ENTITY_LINK |
| 1145 | european_football_2 | ABSTAIN | UNDETERMINED |
| 1213 | thrombosis_prediction | PROVISIONAL | VALUE_GROUNDING |
| 1256 | thrombosis_prediction | SOURCE_TEXT_CONTRADICTION | ANSWER_GRAIN |
| 1265 | thrombosis_prediction | PROVISIONAL | VALUE_GROUNDING |
| 1269 | thrombosis_prediction | PROVISIONAL | VALUE_GROUNDING |
| 1289 | thrombosis_prediction | PROVISIONAL | ANSWER_GRAIN |
| 1441 | student_club | ABSTAIN | UNDETERMINED |
| 1456 | student_club | ABSTAIN | UNDETERMINED |
| 1460 | student_club | ABSTAIN | UNDETERMINED |
| 1474 | debit_card_specializing | PROVISIONAL | TEMPORAL_SCOPE |
| 1494 | debit_card_specializing | ABSTAIN | UNDETERMINED |
| 1502 | debit_card_specializing | SOURCE_TEXT_CONTRADICTION | VALUE_GROUNDING |
| 1525 | debit_card_specializing | SOURCE_TEXT_CONTRADICTION | RATIO_DENOMINATOR_UNIT |
| 1529 | debit_card_specializing | ABSTAIN | UNDETERMINED |

## Independent research implications / stop-loss

1. Cross-database *similarity* is observable: separate entity/population denominator errors appear in toxicology #219, codebase_community #571, thrombosis_prediction #1256, debit_card_specializing #1525. **They are not yet one generalizable algorithm**. Input/output units must first be specified independently and then grounded to physical keys/cardinality.
2. Clear source-condition failures include ordinal #50, conjunction #593, year substitution #1108, currency #1502, and row/result output mixing #883. These mechanisms are qualitatively **different**; merging them into generic Semantic Planning reintroduces an already failed PR #116 approach.
3. Overlap with [PR #116](https://github.com/kolihhan/semantic-text2sql-agent/pull/116) negative prospective counterhypotheses: same fixed Qwen TRAIN48 `26/46→25/46`, generation time 2.16×; **do not repeat generic structured counter-hypothesis prompt**. [PR #100](https://github.com/kolihhan/semantic-text2sql-agent/pull/100) vote census: 45/52 wrong-plurality versus 7 ties. Ordinary tie-break changes cannot fix most selection errors.
4. This source-first triage is **not a completed independent expert blind audit**. Annotator is assistant; source was preselected on known failure, some cases previously exposed; physical FK cardinality, value vocabularies and counterfactual competing explanations are not verified. Record abstentions rather than invent diagnoses.
5. Any next candidate-generation A/B must freeze **one narrowly grounded mechanism** (e.g. typed entity-count unit vs transaction-row unit) and verify multiple source DBs with physical independent evidence. Keep same Qwen3.5-9B, same actual call/time budget and original correct controls; report new correct-candidate oracle, selected strict SET/BAG EX, rescues, regressions, 8-DB uncertainty and cost. HOLDOUT300 sealed. If physical mechanism gate fails, stop before new model run.

**No new benchmark accuracy and no production/Guarded changes.**
