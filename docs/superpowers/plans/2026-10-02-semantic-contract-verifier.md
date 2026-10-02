# Semantic Contract Verifier (SCV) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent Semantic Contract Verifier that checks XiYan-generated SQL against a typed intent contract and database FK graph, performs at most one targeted semantic repair, and measures the effect on the same frozen 100 BIRD cases.

**Architecture:** Extract a candidate-independent semantic contract from question + evidence + selected schema, parse SQL deterministically with SQLGlot, compare the contract to normalized SQL semantics plus the catalog FK graph, and emit typed violations. Integrate SCV after existing deterministic preflight, fail open on unavailable semantic evidence, and allow one violation-guided repair before re-verification.

**Tech Stack:** Python 3.11, dataclasses, JSON, SQLGlot, SQLite, LangGraph, pytest, Ollama/XiYanSQL 7B, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-02-semantic-contract-verifier-design.md`

## Global Constraints

- SCV is feature-gated and off by default until the frozen-100 ablation is complete.
- Contract extraction input is only question, BIRD evidence/hint if present, and selected physical schema context; candidate SQL is never included.
- Malformed/low-confidence contracts and AST parse failures must fail open and must not modify the candidate.
- v1 verifies projection, aggregation/entity grain/DISTINCT, grouping, ranking/top-k, explicit FK join edges, and high-confidence literal omission only.
- Semantic repair budget is exactly one attempt; there is no semantic repair loop.
- Existing deterministic preflight remains responsible for read-only/single-statement/SQLite compile checks.
- Historical guarded behavior must remain reproducible with SCV disabled.
- The frozen-100 benchmark must reuse the exact same XiYan candidate in baseline and SCV arms.
- Report the project scorer accurately as frozen paired execution accuracy unless the official BIRD evaluator is actually used.
- Stop after this experiment if there is no meaningful net gain; no SCV-v2 belongs in this plan.

## Review Focus

- Ambiguous or low-confidence contract fields must abstain rather than create hard violations; Task 1 pins this with low-confidence/unresolved tests.
- SQL aliases and qualified/unqualified columns must normalize consistently; Task 2 pins alias resolution and ambiguous-column behavior.
- Correct joins expressed in reverse equality order must pass, while unrelated identifier joins must fail; Task 3 pins both directions and false-positive controls.
- A correct SQL candidate must not be changed merely because a contract field is absent/unresolved; Tasks 3 and 4 pin no-op behavior.
- Semantic repair must occur at most once even when the repaired SQL still violates SCV; Task 4 pins call count and terminal behavior.

---

## File map

### New files

- `src/semantic_sql/semantic_contract.py` — contract dataclasses, JSON parsing/validation, independent model extraction.
- `src/semantic_sql/sql_semantics.py` — SQLGlot parsing and deterministic normalized SQL semantics.
- `src/semantic_sql/schema_graph.py` — FK graph construction and explicit join-edge validation.
- `src/semantic_sql/contract_verifier.py` — typed SCV violations and contract-vs-SQL checks.
- `tests/test_semantic_contract.py` — contract parsing/extraction/abstention tests.
- `tests/test_sql_semantics.py` — AST normalization tests.
- `tests/test_contract_verifier.py` — semantic verifier + schema-graph tests.
- `tests/test_scv_inference.py` — runtime integration and one-repair tests.
- `experiments/xiyan_scv_frozen100.py` — paired XiYan vs XiYan+SCV benchmark harness.
- `.github/workflows/xiyan-scv-frozen100.yml` — manual-only four-shard benchmark.

### Modified files

- `pyproject.toml` — add pinned-major SQLGlot dependency.
- `src/semantic_sql/contracts.py` — add SCV result/status data structures only if needed by runtime result reporting.
- `src/semantic_sql/inference.py` — insert SCV nodes/routing without changing default historical flow.
- `tests/test_guarded_direct_preflight.py` — preserve historical graph assertions and add SCV-disabled control if needed.

---

### Task 1: Semantic contract model and independent extraction

**Files:**
- Create: `src/semantic_sql/semantic_contract.py`
- Create: `tests/test_semantic_contract.py`

**Interfaces:**
- Consumes: `ModelProvider.complete_text(*, system: str, user: str) -> str`.
- Produces:
  - `ContractConfidence = Literal["high", "medium", "low"]`
  - `AggregationSpec`, `RankingSpec`, `FilterSpec`, `SemanticContract` frozen dataclasses.
  - `ContractExtraction(status: Literal["ok", "skipped"], contract: SemanticContract | None, reason: str)`.
  - `parse_semantic_contract(text: str) -> ContractExtraction`.
  - `extract_semantic_contract(*, provider: ModelProvider, question: str, schema_context: str, evidence: str | None = None) -> ContractExtraction`.

- [ ] **Step 1: Write failing tests for valid contract parsing**

Tests:
- `test_parse_semantic_contract_accepts_typed_high_confidence_contract`
- `test_parse_semantic_contract_normalizes_enum_fields`

Assertions:
- projection/group_by/join_entities become tuples;
- aggregation/ranking nested fields are typed;
- unknown extra JSON keys do not become model fields;
- valid `high` and `medium` confidence values are preserved.

- [ ] **Step 2: Run parser tests and verify they fail**

Run: `pytest tests/test_semantic_contract.py -k "parse_semantic_contract_accepts or normalizes" -v`

Expected: FAIL because `semantic_sql.semantic_contract` does not exist.

- [ ] **Step 3: Implement the contract dataclasses and `parse_semantic_contract(text: str) -> ContractExtraction`**

Parsing rules:
- JSON object only;
- unsupported enum values make the affected field unresolved rather than inventing a value;
- malformed JSON returns `status="skipped"`;
- overall `confidence="low"` returns `status="skipped"`;
- missing optional semantic fields remain empty/None.

- [ ] **Step 4: Re-run parser tests**

Run: `pytest tests/test_semantic_contract.py -k "parse_semantic_contract_accepts or normalizes" -v`

Expected: PASS.

- [ ] **Step 5: Write failing extraction-isolation tests**

Tests:
- `test_extract_contract_prompt_never_contains_candidate_sql`
- `test_extract_contract_fails_open_on_malformed_json`
- `test_extract_contract_skips_low_confidence_contract`
- `test_extract_contract_preserves_unresolved_fields_without_hardening_them`

Use a recording provider and assert the user prompt contains question/schema/evidence only.

- [ ] **Step 6: Run extraction tests and verify failure**

Run: `pytest tests/test_semantic_contract.py -k "extract_contract" -v`

Expected: FAIL because `extract_semantic_contract` is missing.

- [ ] **Step 7: Implement `extract_semantic_contract(...)`**

Use one deterministic model call with a JSON-only system instruction. The prompt must explicitly request the spec fields and confidence, forbid final SQL, and tell the model to use `unresolved`/null when schema grounding is uncertain.

- [ ] **Step 8: Run Task 1 tests**

Run: `pytest tests/test_semantic_contract.py -v`

Expected: all PASS.

- [ ] **Step 9: Commit**

```bash
git add src/semantic_sql/semantic_contract.py tests/test_semantic_contract.py
git commit -m "feat: add semantic contract extraction"
```

---

### Task 2: Deterministic SQL AST normalization

**Files:**
- Create: `src/semantic_sql/sql_semantics.py`
- Create: `tests/test_sql_semantics.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: SQL text.
- Produces:
  - `NormalizedAggregation(function: str, target: str | None, distinct: bool)`.
  - `NormalizedOrder(expression: str, direction: Literal["ASC", "DESC"])`.
  - `NormalizedJoin(left_table: str | None, left_column: str, right_table: str | None, right_column: str)`.
  - `SQLSemantics(...)` containing projections, aggregations, group_by, order_by, limit, literals, joins, distinct.
  - `SQLSemanticsParse(status: Literal["ok", "skipped"], semantics: SQLSemantics | None, reason: str)`.
  - `parse_sql_semantics(sql: str) -> SQLSemanticsParse`.

- [ ] **Step 1: Add SQLGlot dependency and write failing basic AST tests**

Add `sqlglot>=27,<29` to project dependencies.

Tests:
- `test_parse_sql_semantics_extracts_projection_aggregate_group_order_limit`
- `test_parse_sql_semantics_detects_count_distinct`
- `test_parse_sql_semantics_extracts_filter_literals`

- [ ] **Step 2: Run basic AST tests**

Run: `pytest tests/test_sql_semantics.py -k "extracts or count_distinct" -v`

Expected: FAIL because parser module is missing.

- [ ] **Step 3: Implement minimal AST normalization**

Use `sqlglot.parse_one(sql, read="sqlite")`. Normalize identifiers case-insensitively but preserve table/column identity. Do not use regex for clause extraction.

- [ ] **Step 4: Run basic AST tests**

Run: `pytest tests/test_sql_semantics.py -k "extracts or count_distinct" -v`

Expected: PASS.

- [ ] **Step 5: Write failing alias/ambiguity/fail-open tests**

Tests:
- `test_aliases_resolve_to_physical_table_names`
- `test_reverse_join_equality_normalizes_same_edge`
- `test_unqualified_ambiguous_column_remains_unresolved`
- `test_parse_failure_returns_skipped_not_exception`
- `test_cte_query_does_not_crash_normalizer`

- [ ] **Step 6: Run alias/failure tests and verify failure**

Run: `pytest tests/test_sql_semantics.py -k "alias or reverse_join or ambiguous or parse_failure or cte" -v`

Expected: at least one FAIL.

- [ ] **Step 7: Implement alias resolution and conservative unresolved behavior**

Resolve aliases from FROM/JOIN table nodes. When physical table identity cannot be established uniquely, store the column with `table=None`; downstream verifier must abstain from checks requiring that table.

- [ ] **Step 8: Run Task 2 tests**

Run: `pytest tests/test_sql_semantics.py -v`

Expected: all PASS.

- [ ] **Step 9: Run full existing suite for dependency/regression safety**

Run: `pytest -q`

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml src/semantic_sql/sql_semantics.py tests/test_sql_semantics.py
git commit -m "feat: normalize SQL semantics with sqlglot"
```

---

### Task 3: FK graph and typed contract verification

**Files:**
- Create: `src/semantic_sql/schema_graph.py`
- Create: `src/semantic_sql/contract_verifier.py`
- Create: `tests/test_contract_verifier.py`

**Interfaces:**
- Consumes:
  - `SemanticContract` from Task 1.
  - `SQLSemantics` from Task 2.
  - `DatabaseCatalog`.
- Produces:
  - `SchemaGraph.from_catalog(catalog: DatabaseCatalog) -> SchemaGraph`.
  - `SchemaGraph.supports_fk_equality(left_table, left_column, right_table, right_column) -> bool`.
  - `SCVViolation(code, expected, actual, evidence, confidence)`.
  - `SCVVerification(status: Literal["pass", "violation", "skipped"], violations: tuple[SCVViolation, ...], reason: str = "")`.
  - `verify_semantic_contract(contract: SemanticContract, semantics: SQLSemantics, catalog: DatabaseCatalog) -> SCVVerification`.

- [ ] **Step 1: Write failing FK-graph tests**

Tests:
- `test_schema_graph_accepts_declared_fk_edge`
- `test_schema_graph_accepts_reverse_equality_form`
- `test_schema_graph_rejects_unrelated_identifier_join`

Build temporary SQLite schemas with explicit FK constraints and load them through `DatabaseCatalog.from_sqlite`.

- [ ] **Step 2: Run FK tests and verify failure**

Run: `pytest tests/test_contract_verifier.py -k "schema_graph" -v`

Expected: FAIL because `SchemaGraph` is missing.

- [ ] **Step 3: Implement `SchemaGraph`**

Store canonical FK column-pairs and accept equality in either syntactic direction. Do not infer joins from matching names or values.

- [ ] **Step 4: Run FK tests**

Run: `pytest tests/test_contract_verifier.py -k "schema_graph" -v`

Expected: PASS.

- [ ] **Step 5: Write failing verifier tests for each v1 semantic class**

Tests:
- `test_projection_mismatch_requires_resolved_high_confidence_projection`
- `test_aggregation_missing_detected`
- `test_aggregation_function_mismatch_detected`
- `test_distinct_required_for_explicit_entity_grain_count`
- `test_group_by_missing_detected`
- `test_order_metric_mismatch_detected`
- `test_order_direction_mismatch_detected`
- `test_topk_mismatch_detected`
- `test_invalid_explicit_join_edge_detected`
- `test_missing_high_confidence_literal_detected`

Each assertion checks violation code plus `expected`, `actual`, and nonempty `evidence`.

- [ ] **Step 6: Run verifier tests and verify failure**

Run: `pytest tests/test_contract_verifier.py -k "detected or mismatch or distinct_required" -v`

Expected: FAIL because `verify_semantic_contract` is missing.

- [ ] **Step 7: Implement conservative v1 verifier**

Rules:
- only check resolved/high-confidence fields;
- never emit a violation when required table/field identity is unresolved;
- prefer zero violations to speculative violations;
- validate explicit joins only;
- literal omission check is exact/high-confidence only.

- [ ] **Step 8: Add false-positive controls**

Tests:
- `test_correct_projection_is_untouched`
- `test_correct_aggregate_group_rank_passes`
- `test_unresolved_projection_abstains_from_projection_check`
- `test_missing_optional_contract_fields_do_not_create_violations`
- `test_valid_fk_join_with_aliases_passes`
- `test_correct_sql_with_low_contract_confidence_is_skipped_upstream`

- [ ] **Step 9: Run Task 3 tests**

Run: `pytest tests/test_contract_verifier.py -v`

Expected: all PASS.

- [ ] **Step 10: Commit**

```bash
git add src/semantic_sql/schema_graph.py src/semantic_sql/contract_verifier.py tests/test_contract_verifier.py
git commit -m "feat: add semantic contract verifier"
```

---

### Task 4: Runtime integration and one targeted semantic repair

**Files:**
- Modify: `src/semantic_sql/inference.py`
- Modify: `src/semantic_sql/contracts.py` only if public result/status fields are needed
- Create: `tests/test_scv_inference.py`
- Modify: `tests/test_guarded_direct_preflight.py`

**Interfaces:**
- Consumes all Task 1–3 interfaces.
- Produces:
  - `SCV_TREATMENT_ID = "semantic_contract_verifier_v1"`.
  - `run_guarded(..., semantic_contract_verification: bool = False, semantic_repair_budget: int = 1)`.
  - Stage names: `semantic_contract`, `semantic_verify`, `semantic_repair`.
  - No behavior change when `semantic_contract_verification=False`.

- [ ] **Step 1: Write failing backward-compatibility test**

Test: `test_scv_disabled_preserves_historical_guarded_graph_and_call_count`

Assert:
- existing graph nodes/stages remain unchanged with default arguments;
- existing semantic_revision flag still behaves as before;
- no contract model call occurs.

- [ ] **Step 2: Run backward-compatibility test**

Run: `pytest tests/test_scv_inference.py::test_scv_disabled_preserves_historical_guarded_graph_and_call_count -v`

Expected: FAIL because SCV flag/interface does not exist.

- [ ] **Step 3: Add feature-gated SCV state/context fields without routing changes**

Add default-off arguments and state slots only.

- [ ] **Step 4: Re-run backward-compatibility test**

Expected: PASS.

- [ ] **Step 5: Write failing SCV routing tests**

Tests:
- `test_scv_contract_skip_executes_original_candidate_unchanged`
- `test_scv_ast_skip_executes_original_candidate_unchanged`
- `test_scv_pass_executes_without_semantic_repair`
- `test_scv_violation_triggers_one_targeted_repair`
- `test_scv_repair_is_preflighted_and_semantically_reverified`
- `test_scv_repair_failure_never_triggers_second_semantic_repair`
- `test_correct_candidate_not_changed_when_contract_has_no_hard_violation`

Use deterministic recording providers and monkeypatch Task 1–3 boundaries so each test isolates routing rather than LLM behavior.

- [ ] **Step 6: Run routing tests and verify failure**

Run: `pytest tests/test_scv_inference.py -k "scv_" -v`

Expected: FAIL because routing/nodes are missing.

- [ ] **Step 7: Implement SCV nodes and routing**

Required runtime sequence after preflight pass:

```text
semantic_contract
→ semantic_verify
→ execute                    if pass/skip
→ semantic_repair            if violation and budget=1
→ verify                     existing deterministic preflight
→ semantic_verify            once more
→ execute or semantic failure
```

Targeted repair prompt includes only question, evidence, schema, candidate SQL, and typed violations. It instructs the provider to make the smallest change needed and return SQL only.

Do not reuse CGSR generic review as the SCV verifier.

- [ ] **Step 8: Run integration tests**

Run: `pytest tests/test_scv_inference.py tests/test_guarded_direct_preflight.py -v`

Expected: PASS.

- [ ] **Step 9: Run full suite**

Run: `pytest -q`

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/semantic_sql/inference.py src/semantic_sql/contracts.py tests/test_scv_inference.py tests/test_guarded_direct_preflight.py
git commit -m "feat: integrate semantic contract verification"
```

---

### Task 5: Real-failure regression fixtures

**Files:**
- Create: `tests/test_scv_regressions.py`

**Interfaces:**
- Consumes Task 1–4 public interfaces.
- Produces: no new production interface; this task pins known semantic failure patterns and false-positive controls.

- [ ] **Step 1: Add entity-grain regression**

Fixture pattern: “How many [entities] contain/have [lower-grain records]?” Candidate uses `COUNT(entity_id)` across a one-to-many join; contract requires entity-grain count.

Assert `DISTINCT_REQUIRED` or equivalent entity-grain violation.

- [ ] **Step 2: Add projection regression**

Fixture pattern: candidate filters correctly but projects the wrong field.

Assert `PROJECTION_MISMATCH`.

- [ ] **Step 3: Add wrong-join regression**

Fixture uses declared FKs plus an explicit equality between unrelated identifier columns.

Assert `JOIN_EDGE_INVALID`; assert the true FK equality passes.

- [ ] **Step 4: Add aggregate-ranking regression**

Fixture pattern: “which constructor/team/entity has the most total points?” Candidate orders row-level points with LIMIT 1.

Assert `AGGREGATION_MISSING` and/or `ORDER_METRIC_MISMATCH`; correct SUM/GROUP BY/ORDER BY query passes.

- [ ] **Step 5: Add top-k/tie control**

Pin:
- explicit `top 1` → LIMIT 1 required;
- “all with the maximum” contract → LIMIT 1 is not considered sufficient when ties are explicit;
- unspecified ties do not trigger speculative tie violations.

- [ ] **Step 6: Add correct-query preservation controls**

At least one correct fixture for each semantic class must produce no hard violations.

- [ ] **Step 7: Run regression file**

Run: `pytest tests/test_scv_regressions.py -v`

Expected: PASS.

- [ ] **Step 8: Run full suite**

Run: `pytest -q`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add tests/test_scv_regressions.py
git commit -m "test: pin semantic verifier regressions"
```

---

### Task 6: Paired frozen-100 XiYan vs XiYan+SCV ablation

**Files:**
- Create: `experiments/xiyan_scv_frozen100.py`
- Create: `.github/workflows/xiyan-scv-frozen100.yml`

**Interfaces:**
- Reuse:
  - frozen case IDs from `runs/bird-paired-dev-v4/paired.json`;
  - `select_schema_context(...)`;
  - XiYan generation prompt/model from the prior `experiment/xiyan-cgsr-frozen100` harness;
  - the same exact generated XiYan SQL for both arms.
- Produces:
  - per-case JSON rows;
  - aggregate artifact `xiyan-scv-frozen100.json`.

- [ ] **Step 1: Port the prior XiYan generator and database resolver without CGSR logic**

Keep model:
`hf.co/wanhin/XiYanSQL-QwenCoder-7B-2504-gguf:Q4_K_M`.

Do not regenerate SQL separately for the SCV arm.

- [ ] **Step 2: Implement paired case evaluation**

For each case record:

```text
case_id
database_id
question
xiyan_sql
baseline_execution_success
baseline_match
contract_status
contract_summary
scv_status
scv_violations
scv_sql
scv_execution_success
scv_match
sql_changed
contract_latency_s
semantic_latency_s
```

The gold SQL is used only by the scorer, never contract extraction, generation, verification, or repair.

- [ ] **Step 3: Implement accurate local scoring labels**

If retaining the existing project set-equality scorer, use field names such as `frozen_execution_match`, not `official_ex`.

Keep truncated-result behavior explicit in the artifact.

- [ ] **Step 4: Create manual-only four-shard workflow**

Workflow:
- `workflow_dispatch` only;
- Python 3.11;
- install `.[dev]`;
- install/start Ollama;
- pull XiYanSQL 7B;
- download pinned BIRD 2024-06-27 dev zip;
- verify SHA256 `cdd6d19faeb45a23970b98d3ef6c40a87987c95459c2cf12076897a60cf5a630`;
- run shards 0–3;
- upload every shard artifact even on failure.

No push trigger and no repository writes.

- [ ] **Step 5: Aggregate paired metrics**

Aggregate exactly 100 unique cases and report:

```text
baseline frozen_execution_match
SCV frozen_execution_match
execution_success for both arms
wrong_to_correct
correct_to_wrong
net_correct_delta
changed_sql
contract status counts
SCV status counts
violation counts by type
repair success by type
skip count
median contract latency
median semantic overhead
```

- [ ] **Step 6: Add harness smoke tests where practical**

At minimum unit-test pure aggregation/scoring helpers locally rather than making the workflow the first verifier of JSON schema.

- [ ] **Step 7: Run local test suite**

Run: `pytest -q`

Expected: PASS.

- [ ] **Step 8: Commit benchmark harness**

```bash
git add experiments/xiyan_scv_frozen100.py .github/workflows/xiyan-scv-frozen100.yml tests/
git commit -m "experiment: add paired XiYan SCV frozen-100 ablation"
```

- [ ] **Step 9: Run the GitHub Actions benchmark once**

Run the manual workflow on the implementation branch.

Acceptance checks before interpreting quality:
- all 4 shards complete;
- aggregate contains exactly 100 unique frozen cases;
- baseline arm reproduces the stored XiYan candidates/scoring contract;
- no gold SQL appears in generation/contract/SCV prompts;
- baseline and SCV arms share each case’s exact initial XiYan SQL.

- [ ] **Step 10: Apply the stop/go gate**

Promote SCV as a successful project mechanism only if:

```text
SCV frozen_execution_match > baseline
wrong_to_correct > correct_to_wrong
correct_to_wrong <= 2
preferred: net_correct_delta >= +3
```

If the result is approximately `57→57` or `57→58`, or false-positive intervention is high, record the negative result and freeze feature development.

- [ ] **Step 11: Commit only durable experiment evidence, not generated large artifacts**

Keep the workflow artifact as the raw run evidence. If a small summary belongs in-repo, add it in a separate evidence commit after verifying the run.

---

## Final verification before branch review

- [ ] Run `pytest -q`.
- [ ] Run `python -m compileall src evaluation experiments`.
- [ ] Confirm `semantic_contract_verification=False` preserves historical behavior.
- [ ] Confirm no automatic benchmark push trigger exists.
- [ ] Review all SCV prompts for accidental candidate leakage into contract extraction.
- [ ] Review all benchmark prompts for gold-answer leakage.
- [ ] Compare implementation against the design spec section-by-section.
- [ ] Request whole-branch code review before merge.
- [ ] Do not redesign README until the frozen-100 decision is known.
