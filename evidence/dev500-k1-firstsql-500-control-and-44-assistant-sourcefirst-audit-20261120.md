# BIRD DEV500 — frozen original first-SQL K1 source-first error signals and correct controls (2026-11-20)

## Physical source / result identity

- [Source-only 500-case generation cache + source-frozen 44-case review and second-job score, Actions **44660486791**](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44660486791). Both jobs SUCCESS; source artifact **14595795616**, post-score artifact **14595466511**.
- Original 32 immutable Qwen3.5-9B K8 cache shards from Actions 44086713331 (2) + 44093103960 (30); 8 original uniform 8-second physical-score workers from Actions 44222560767. **No new model calls** and no changes to Qwen, verifier, prompt, repair count or database.
- SHA-locked publisher public DEV source was reduced in a separate shell process to *question + DB + difficulty only*. Risk extraction code sees no reference SQL or source outcomes. Only the second Actions job opens previously frozen physical worker K1/K8 success labels; no new publisher SQL or SQLite queries scored.
- BIRD DEV500 first original Qwen SQL index 0 is **203/500 = 40.6%** strict 8-second bounded SET-result EX, wrong **297/500 = 59.4%**; K8 selector **233/500 = 46.6%**. Gold references physically scorable **499/500**. K1 here is **direct original single sample, without Guarded max2 repair**, NOT the final Guarded pipeline EX.
- One model-generated first SQL hit the 384-token cap on **8/500** cases (3 california_schools, 5 financial). This is generation metadata, not proof of root cause of their task errors.
- No HOLDOUT300 source task IDs scored or sampled, no QLoRA/training.

## Source-only question type association with first-K1 result

| Source question keyword family (overlap allowed) | n | first SQL correct | correct rate |
|---|---:|---:|---:|
| Count | 118 | 52 | 44.1% |
| Rank/top/min/max | 85 | 23 | 27.1% |
| Ratio/percent | 43 | 12 | 27.9% |
| Temporal cue | 32 | 10 | 31.3% |
| Relation cue | 32 | 12 | 37.5% |
| **All DEV500** | **500** | **203** | **40.6%** |

Database confounding: financial **2/34=5.9%** first SQL correct; california_schools **4/28=14.3%**; student_club **33/53=62.3%**. Strong apparent DB variation; cannot infer ranking/ratio *causes* low EX without matching DB/difficulty and task complexity.

Source-only SQL risk flags (NOT semantic error labels) identified:
- `P_COUNT_STAR_AFTER_JOIN`: **40** first SQLs, **30 wrong / 10 right**. False-positive association on 10 correct rows; `COUNT(*)` after a JOIN often correctly counts the requested child/transaction unit.
- `P_CONJUNCTION_AS_MEMBERSHIP`: **1**, wrong; deliberately narrow regex missed other `and` conjunction questions such as source #593.
- `P_RANK_WITHOUT_ORDER_OR_EXTREMA`: **3**, wrong; literal `MAX` alternatives and natural-language synonyms limit coverage.
- These flags were designed on previously exposed development data, are incomplete lexical evidence, and have no deployment claim.

## Source-frozen assistant first-pass audit on 44 per-DB-selected full question/first SQL pairs

The source-only process preselected **4 questions per database across all 11 DBs**, 44 total, using source risk flags and hashes **before score labels**. **This is an assistant first-pass analysis, NOT independent expert annotation** and the risk-enriched sample is **not representative** of the 500; some selected cases (#593, #1256, #1091) were already discussed before this review.

[Original source-only full Q/SQL selection](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44660486791) and [assistant-authored source-only labels](https://github.com/kolihhan/semantic-text2sql-agent/blob/research/dev500-k1-firstsql-intent-survey-20261120/evidence/dev500-k1-frozen44-assistant-sourcefirst-mechanism-hypotheses-20261120.json).

| Source-only assistant judgement | Total | K1 wrong after later matching historical worker | K1 right |
|---|---:|---:|---:|
| High-confidence apparent question↔first SQL contradiction | **14** | **12** | **2** |
| Potential contradiction; needs physical schema/state verification | **8** | **6** | **2** |
| Abstain / cannot tell | **22** | **12** | **10** |

Source-first examples:
- #271 toxicology: source asks whether bond endpoints include **both** chlorine and carbon; SQL selects atoms using `element IN ('cl','c')` (OR), without proving both bond endpoints.
- #593 codebase_community: teacher **and** supporter badge required; SQL `IN ('Supporter','Teacher')` and `COUNT(*)` (OR plus badge-row count), previously exposed.
- #1503 debit_card_specializing: source asks transactions **in euro**, but SQL filters `Price < 0`; currency-versus-price semantic grounding mismatch.
- #275 toxicology: source requests **molecule ID** but SQL projects **bond ID** aliased `molecule_id`.
- #190 financial: extra `account` join by district prior to correctly named `client→disp→account` joins, likely incorrect multiplicity/path.
- #1037 european_football_2: SQL uses `YEAR()` unsupported by SQLite; attribute snapshots potentially duplicate players.
- #1256 thrombosis: source asks patients and SQL counts `Laboratory` rows; earlier physical reference confirms COUNT DISTINCT patient rescue; prior exposure, not a newly discovered success.

**Crucial counterexamples:** #1427 (budget category with extra `e.type` projection) and #773 (compares eye/hair/skin **FK integer identifiers from distinct conceptual domains**) were marked source-inconsistent but **still matched the frozen reference SET result**. Either the initial semantic label overstates an ambiguity, or the one-instance result scorer admits coincidental equivalence. **Execution match is not proof of semantically correct generalization**. No performance claim from this observation.

## Independent critique of next model experiment

1. *Do not repeat a generic Semantic Planning prompt yet.* A prior fixed-Qwen prospective original 32 BIRD DEV source run [43851807351](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43851807351) had **15/32→13/32**, 1 rescue / 3 regressions. An older TRAIN40 planning pilot showed 15/40→16/40 (3 rescues / 2 regressions) but did not establish transfer. Repeating it on DEV500 now exposed to this inspection would be further development contamination.
2. These 297 first-K1 errors are **not annotated causal roots**. Rank and ratio families are natural high-priority hypotheses, but the next experiment must identify target entity, temporal/reference scope and denominator with *source-only independent evidence*, not publisher SQL AST diff.
3. Overlapping source tags cannot be summed, and an enriched 44-case sample cannot estimate population error mechanism rates. Human rubric adjudication with independent reviewers would add cost but is needed before claiming real causal coverage.
4. User's requirement is eventual same-Qwen same-original-Guarded max2 paired EX, **rescues, regressions, cost and DB-cluster uncertainty**. K1 versus K8 is not the causal model-prompt A/B and should not be described as such.
5. Next admissible generated-SQL intervention must have *an observable, question-ID-agnostic, non-oracle mechanism* and independent prospective evaluation. Preserve HOLDOUT300.

**Disposition:** This round successfully built the correct K1 contrast population, a source-first 44-case qualitative review queue and falsifiable semantic failure hypotheses. **No new BIRD Guarded score, no QLoRA, no production code promotion.**


## Extra challenge: question type versus database composition

Using only the frozen 500 per-case K1 outcomes and 11 physical DB groups, calculate the **expected correct cases for each question type under the observed within-DB baseline correct rate** (no new model/data). This is a descriptive standardization, *not* a causal analysis, randomized test, or new prospective result:

| Source question tag | Observed correct | Expected correct after reweighting by each DB's overall K1 rate | Difference |
|---|---:|---:|---:|
| Rank/top/min/max | 23/85 | 31.82/85 | **-8.82 questions** |
| Ratio/percent | 12/43 | 15.79/43 | -3.79 questions |
| Temporal | 10/32 | 10.51/32 | -0.51 questions |
| Count | 52/118 | 47.19/118 | +4.81 questions |
| COUNT(*) after JOIN risk proxy | 10/40 | 15.67/40 | -5.67 questions |

Thus **ranking remains a more compelling descriptive weakness than generic percent/ratio after this coarse DB reweighting**. But within-DB difficulty, question length, SQL decoding budget and schema complexity remain uncontrolled. Tags overlap, some were chosen after previous exposure, and there is no correction for multiple comparisons. Do NOT describe the table as statistical proof of semantic root cause or guaranteed prompt lift.

