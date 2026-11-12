# Semantic Text-to-SQL: root-cause review and non-overfit design (2026-11-12)

Status: **research specification only**. No new model, selector, benchmark tuning or deployment is claimed.

## 1. Ground truth: completed paired experiment, not imagined improvements

Evidence: GitHub Actions [43211214560](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43211214560), artifact `xiyan-semantic-independent100` (SHA256 cohort `54d1cdf613d4db736a45cfd57b8659b2fa6b468a709e3ad4ef77be4d57bbc266`). XiYanSQL-QwenCoder-7B-2504 GGUF Q4_K_M, BIRD DEV previously unexamined questions; **same public BIRD release**, not hidden-test or official BIRD score. The 100 labels have now been inspected, so this cohort is **exploratory** for future decisions.

| Observed condition | Count |
|---|---:|
| Baseline execution-match correct | 65 |
| Semantic-revision candidate correct | 65 |
| Final gold-blind selection correct | 66 |
| Both candidates correct | 63 |
| Both candidates wrong | **33** |
| Baseline only correct | 2 |
| Revision only correct | 2 |
| Baseline executable/valid but EX-wrong | **28** |
| Baseline non-executable/invalid | 7 |
| Revised text differs from baseline | 21 |
| Incorrect baseline left unchanged by reviser | 20 |
| Revised candidate selected | 4 |
| Wrong-to-correct accepted | 1 |
| Correct-to-wrong accepted | 0 |
| Possible best-of-two oracle ceiling (with labels) | 67 |

All exact figures recomputed from the archived 100-case JSON. The oracle ceiling is **diagnostic, label-dependent**, not an implementable selector.

Paired changed SQL: 13 both wrong, 4 both correct, 2 revision-only correct, 2 baseline-only correct. Only one of the two possible revision rescues was selected. The conservative selector avoided both candidate regressions, but generated **too few correct alternatives** to raise 100-case EX meaningfully. The selection gate of +3 was **not met** (+1).

### Evidence-bearing counterexamples (case IDs not rules)

* **134 / financial:** executable baseline places `ORDER BY district.A16 DESC LIMIT 1` outside an ungrouped `COUNT(account_id)`; successful revision instead constrains the count to districts satisfying `A16 = (SELECT MAX(A16) ...)`. The current selector kept the wrong baseline because keyword checks found no improvement. Demonstrates **aggregation scope / target population**, not presence of COUNT or MAX tokens.
* **284 / toxicology:** revision changed `element='c' OR element='c'` on two atom aliases to `AND`; baseline EX-correct, revision EX-wrong. Preserving Boolean intent matters.
* **1218 / thrombosis_prediction:** revision changed a numeric predicate `UA > 6.5` to `UA > 8.0` without a demonstrated source; baseline EX-correct, revision EX-wrong. Never accept ungrounded literal mutation.
* **268, 866, 955:** invalid physical identifiers were corrected to runnable variants but all remained EX-wrong. Compiler recovery **is not semantic recovery**.
* **479 / card_games:** correcting an invalid table/column choice produced the one selected EX rescue. This is a useful narrow repair mechanism.

These are examples illustrating failure mechanisms; they do **not** establish a complete root-cause taxonomy of all 35 baseline failures. The aggregate contains SQL, database IDs, validity, and EX flags, **not** the original question/evidence/gold SQL; semantic labeling of every case requires reading those sources and ideally independent human review.

### Existing failed ablations are a warning

* Prior Semantic Contract Verifier experiment on a *different* frozen cohort: XiYan **58 -> 33/100**, with 25 correct-to-wrong regressions; see `docs/dgsl-experiment.md`.
* DGSL v1 on a frozen cohort: **58 -> 58/100**, 7 wins, 7 losses. Its improved observed value-grounding/precision came with lower table/column/FK recall; see `docs/dgsl-experiment.md`.
* Previous Qwen9B optional semantic reviewer **39 -> 38/100** on its own frozen 100, from `README.md`. Distinct models/scorers/cohorts must not be compared as a single leaderboard.

## 2. Code-level faults: exact mechanisms

1. `experiments/xiyan_semantic_shards.py::_required_intents` maps loose phrases to flat tokens: COUNT, AVG, SUM, MAX, MIN, DISTINCT_COUNT. `check_question_constraints` searches anywhere in a SQLGlot syntax tree: a COUNT in an unrelated subquery, or any descending ORDER + LIMIT, can satisfy the entire question even with wrong grouping/scope or ranking target.
2. `choose_semantic_sql` only switches when baseline fails execution or the count of **surface** keyword violations strictly decreases. It therefore misses valid semantic rescues (case 134) and cannot distinguish safe from unsafe changes when both queries execute.
3. `revise_with_semantic_audit` is a free-form second SQL-generation call conditioned on baseline SQL. There is no independent, evidence-grounded target semantic plan. Observed 21 edits included 13 both-wrong and two harmful rewrites.
4. `evaluation/schema_context.py::select_schema_context` uses lexical overlap and one-hop FK expansion; `DatabaseCatalog` primarily exposes names and declared FKs, but not enough usable uniqueness, key cardinality, types or data semantics for enforcing grain.
5. Existing `verify_sql_preflight` proves only safe single-statement SQL and SQLite compilation; a successful SELECT or empty result cannot prove NL-to-SQL semantic correctness.
6. Local `score_query` compares **sets of result tuples**: duplicate multiplicity and row order are not faithfully evaluated. Treat this as a project-local EX proxy with known blind spots.

**Mechanism hypothesis, not proven diagnosis:** the main problem is insufficient candidate *generation/grounding*, compounded by an under-specified intent contract and a selector that can only assess superficial clues. The 33 cases with no correct candidate directly bound how much a reranker can gain.

## 3. Proposed minimal semantic layer: evidence-grounded query contract

```
question + supplied domain evidence
    -> [A] typed, incomplete intent contract (no SQL)
    -> [B] schema/value grounding with real metadata and provenance
    -> XiYan SQL candidate (reuse current M-Schema baseline)
    -> [C] SQLGlot qualified AST + scoped relational feature extraction
    -> [D] compare contract and SQL: proven_mismatch / compatible / unknown
    -> [E] route:
         proven invalid SQL -> narrow compiler repair
         proven semantic mismatch -> constrained regeneration (at most 1)
         compatible or unknown -> leave baseline unchanged / abstain when needed
    -> bounded read-only execution + audit trail
```

### [A] Intent contract (NOT just keywords)

Versioned structured representation (partial fields allowed):

* `answer`: entity/measure and expected output columns/shape;
* `grain`: one row per entity, event, joined event, or aggregated group; entity identity/keys if grounded;
* `operation`: COUNT vs COUNT DISTINCT vs SUM/AVG, ratio denominator, quantified EXISTS, scope/subquery;
* `predicates`: typed operator, subject, values, Boolean AND/OR/NOT, time bounds and inclusivity;
* `ranking`: measure/aggregate **being ranked**, partition, direction, top-k, ties;
* `evidence`: exact span from question, supplied domain knowledge or inspected DB value that justifies every binding;
* `unknown`: ambiguous target, insufficient evidence, unclear metric/term.

Do not force an uncertain field into a fabricated exact match. An LLM may propose the contract, but physical identifiers/values require an evidence-binding stage. Keep the question-span provenance.

### [B] Grounding with recall protection

Use the existing physical SQLite catalog, CSV descriptions, declared FKs, and **add types, PK/unique constraints when verifiable**. Link terms to candidate tables/columns, join paths and typed value hits with explicit provenance. An FK is not by itself proof of 1:1 cardinality; verify unique keys or mark cardinality unknown. Include candidate alternatives and use full-schema fallback where recall is uncertain; avoid aggressive pruning that repeated the DGSL 7-win/7-loss tradeoff. A finite budget of **read-only** value probes can establish literal existence/normalization; do not infer business semantics from a sample value alone.

### [C] Scoped AST inspection, not token presence

Use SQLGlot `parse` + `qualify` (with a complete schema) and inspect per-SELECT/CTE scope. Verify:
* output expressions and group-by keys correspond to requested answer/grain;
* COUNT DISTINCT / GROUP BY / SUM and join duplication risk correspond to the entity being counted;
* ranking applies to the requested metric **at the correct grain**, not an arbitrary value;
* all explicit predicates and AND/OR/NOT structures are preserved, including literals and bound units;
* joins and correlated-subquery references are resolvable through physical catalog evidence;
* null, tie handling, time intervals, and empty results are flagged **unknown** when correctness cannot be proved.

Use three-valued decisions, not a single "semantic_ok" bit: `PROVEN_MISMATCH`, `COMPATIBLE_WITH_CHECKED_CONTRACT`, `UNKNOWN`. `COMPATIBLE` is **not** a guarantee of correct answers.

### [D/E] Bounded selector preserving correctness

Do not rewrite a correct/unknown SQL merely because a second model prefers something different. Automatically repair proven compile failures. For high-confidence **grounded** semantic contradictions, construct a narrowly scoped repair request that states the exact broken constraint and its evidence span. Re-parse/re-check after one repair. Never change hard numeric/text literals, Boolean operators, or aggregate grain without an explicit question/evidence justification. When evidence is insufficient, retain baseline and report `unknown` or ask for disambiguation rather than inventing a correction. Record per-case cost, model calls, and why the selector acted.

The first build should **not** include multi-agent voting, a new semantic LLM, or full NL-to-SQL correctness assertions. First add a passive typed contract and AST observer, then evaluate precision of actionable mismatches.

## 4. Anti-overfit protocol, pre-register before implementation

**Data boundary**

* **Exploratory:** all previously observed frozen100 cohorts and the independent100 artifact above. Its case IDs and their SQL/gold outcome MUST NOT become hard-coded examples, per-database rules, prompt demonstrations, routing thresholds, or supervised labels.
* **Development:** BIRD TRAIN only for implementation fixtures and, if later warranted, SFT. Split by **database**, not random rows, for internal validation; ensure descriptions and template paraphrases do not cross boundaries.
* **Confirmatory:** reserve **previously untouched BIRD DEV database groups** and at least one schema/dataset shift (e.g. Spider holdout) *before* reading their labels. If BIRD DEV DBs have all been explored, choose a genuinely unexamined source; row-disjoint within a previously studied DB is weaker evidence.

**Adversarial / metamorphic suite (generic generated databases, not BIRD-specific)**

1. Multiplying child rows must not change COUNT DISTINCT of parent entities.
2. Swapping OR/AND, >/>=, date boundaries and numeric constants should be detected as **constraint differences**, not silently repaired.
3. Permuting table/column names and irrelevant schema descriptions should not alter a correctly grounded plan, subject to available metadata.
4. Add tied maxima, NULLs, empty results, duplicate rows and dirty value representations; inspect sensitivity to each.
5. Equivalent SQL rewrites (extra parentheses, aliases, safe join rearrangements) must not be flagged as mismatches.
6. Ambiguous natural language must produce an uncertainty signal instead of a confident invented metric.
7. Explicit join paths but missing/ambiguous FK/uniqueness must remain `UNKNOWN`, not false proof.

**Ablations / acceptance**

Compare on the *same* frozen paired cohort and exact model/runtime:

- A0: M-Schema XiYan baseline;
- A1: baseline + passive semantic contract/AST diagnostics (**no SQL changes**);
- A2: A1 + narrow deterministic **compile** repairs;
- A3: A2 + one evidence-grounded conditional **semantic** regeneration.

Report EX alongside candidate validity, **correct->wrong, wrong->correct**, critical literal/predicate regression counts, change rate, abstentions, latency and extra model calls. Optionally include test-suite/multiset/order-sensitive checks on generated data; project-local set-of-tuples EX must not be sold as a semantic proof. Prefer per-database paired summaries and uncertainty intervals, and do not promote on a small one-run +1 gain. Freeze thresholds **before** confirmation. Stop/escalate if high-confidence mismatch precision is poor, if no candidate improvements exist, or if latency doubles without durable correctness benefit.

**Success condition:** meaningful positive net correctness on untouched database/source shifts **without a statistically/materially concerning correct-to-wrong rate**, with demonstrated gain specifically attributable to the semantic layer over A2, and reasonable runtime cost. No bespoke rules keyed to BIRD IDs, DB names, or observed gold SQL.

## 5. External grounding

* BIRD benchmark: https://arxiv.org/abs/2305.03111 (value/knowledge/dirty-schema challenges).
* CHESS: https://arxiv.org/abs/2405.16755 (retrieval, schema selection, candidate generation, unit testing).
* XiYan-SQL: https://arxiv.org/abs/2507.04701 (schema filter, multiple candidates and candidate selection; not proof we should copy its full multi-generator stack).
* DIN-SQL: https://papers.neurips.cc/paper_files/paper/2023/hash/72223cc66f63ca1aa59edaec1b3670e6-Abstract-Conference.html (decomposition of complex semantics).
* SQLGlot qualification API: https://github.com/tobymao/sqlglot/blob/main/sqlglot/optimizer/qualify.py.
* Text2SQL evaluation limitation taxonomy: https://arxiv.org/abs/2501.18197 (ambiguity, noisy labels, biased match metrics).
* Spider 2.0: https://spider2-sql.github.io/ (out-of-distribution schema complexity).
