# P1 Benchmark-First Text-to-SQL Implementation Plan

> **Historical, superseded implementation record.** Do not execute this planner/grounding plan as the current architecture. See `../../architecture.md` and `../../architecture/p1-dev-decision.md` for the implemented one-graph system and frozen paired DEV decision.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace demo-specific semantic constraints with an explicit general query contract, strengthen deterministic verification, and create a fair BIRD Direct/Structured/Guarded evaluation seam.

**Architecture:** Keep the current planner -> grounding -> generator -> verifier -> bounded repair pipeline, but make the semantic/grounded contracts explicit enough for real benchmark questions. Demo fixtures remain smoke tests; BIRD-derived DEV/FINAL subsets are external, deterministic evaluation inputs.

**Tech Stack:** Python 3.11+, dataclasses, SQLite, pytest, existing Ollama `ModelProvider` interface.

**Spec:** `docs/superpowers/specs/2026-08-25-p1-benchmark-first-design.md`

## Global Constraints

- No new orchestration framework or agent abstraction.
- No BIRD data/model weights committed to the repository.
- Existing demo fixtures are smoke tests only, never portfolio results.
- Keep read-only execution and bounded repair behavior.
- Do not tune on a final holdout.
- Preserve backwards compatibility only where it does not retain demo-specific semantics in the core contract.
- The supplied snapshot has no `.git`; verify each task independently and package the result instead of committing.

---

### Task 1: Make query semantics explicit and lossless

**Files:**
- Modify: `src/semantic_sql/contracts.py`
- Modify: `src/semantic_sql/planner.py`
- Modify: `src/semantic_sql/grounding.py`
- Modify: `src/semantic_sql/generator.py`
- Test: `tests/test_contracts.py`
- Test: `tests/test_grounding.py`
- Test: `tests/test_planner_provider.py`

**Interfaces:**
- Produces: `Aggregation`, `SemanticSelect`, `SemanticOrder`, updated `SemanticPlan`, `GroundedSelect`, `GroundedOrder`, updated `GroundedPlan`.
- Later tasks consume the grounded aggregation/filter/order/limit fields for verification and generation.

- [ ] **Step 1: Add failing tests for explicit aggregation and preserved ordering/limit**

Add tests that construct a plan for `AVG(salary)`, an ordered field, and a limit; assert grounding preserves aggregation, order direction, and limit. Add a planner payload test that expects `select`, `order_by`, and `limit` rather than a single demo metric string.

- [ ] **Step 2: Run focused tests and confirm failure**

Run:
```bash
python -m pytest -q tests/test_contracts.py tests/test_grounding.py tests/test_planner_provider.py
```
Expected: failures because the new contract types/fields do not exist.

- [ ] **Step 3: Implement the minimal explicit contracts**

Use:
```python
Aggregation = Literal["none", "count", "sum", "avg", "min", "max"]

@dataclass(frozen=True)
class SemanticSelect:
    field: str
    aggregation: Aggregation = "none"

@dataclass(frozen=True)
class SemanticOrder:
    field: str
    direction: Literal["asc", "desc"] = "asc"
```

Update `SemanticPlan` to store `select`, `filters`, `group_by`, `order_by`, and `limit`. Ground the select/order fields using the existing conservative field resolver and carry `limit` through unchanged. Keep temporal conditions as ordinary typed filters rather than a separate free-form string.

- [ ] **Step 4: Update generator payload to serialize the new grounded contract**

The model receives explicit aggregation/order/limit fields in the grounded plan; do not infer them from semantic names.

- [ ] **Step 5: Run focused tests**

Run the same focused pytest command. Expected: PASS.

---

### Task 1B: Remove arbitrary 50-value grounding dependence

**Files:**
- Modify: `src/semantic_sql/catalog.py`
- Modify: `src/semantic_sql/grounding.py`
- Test: `tests/test_grounding.py`

**Interfaces:**
- Produces: `DatabaseCatalog.find_value(table, column, semantic_value) -> object | None`.
- Grounding uses exact/case-insensitive lookup before failing; it does not depend on the first N sampled values.

- [ ] **Step 1: Add a failing regression test where the required value occurs after 50 distinct rows**

- [ ] **Step 2: Run the focused test and verify current grounding fails**

- [ ] **Step 3: Add exact/case-insensitive DB lookup and use it from `_physical_value`**

- [ ] **Step 4: Run grounding tests**

---

### Task 1C: Give the semantic planner a non-physical field vocabulary

**Files:**
- Modify: `src/semantic_sql/planner.py`
- Modify: `src/semantic_sql/inference.py`
- Test: `tests/test_planner_provider.py`

**Interfaces:**
- `create_semantic_plan(..., semantic_fields: tuple[str, ...] = ())` may receive natural-language field labels only.
- `run_structured` supplies `GroundingHints.fields.keys()` so the planner chooses groundable semantic labels without seeing physical table/column mappings.

- [ ] **Step 1: Add a failing capture test proving semantic labels are visible but physical names are not**

- [ ] **Step 2: Run the focused test and confirm failure**

- [ ] **Step 3: Add the optional semantic vocabulary to planner input and wire it through Structured/Guarded**

- [ ] **Step 4: Run planner and agent tests**

---

### Task 2: Strengthen deterministic verifier invariants

**Files:**
- Modify: `src/semantic_sql/verifier.py`
- Test: `tests/test_generator_verifier.py`

**Interfaces:**
- Consumes: explicit grounded aggregation/filter/order/limit from Task 1.
- Produces: existing `VerificationResult` with additional issue codes such as `wrong_operator`, `wrong_group_by`, `wrong_join`, `wrong_order_by`, and `wrong_limit`.

- [ ] **Step 1: Add failing regression tests**

Tests must prove the verifier rejects:
```sql
-- expected country_code = 'CZE'
WHERE country_code != 'CZE'

-- expected GROUP BY customers.name
GROUP BY orders.id

-- expected orders.customer_id = customers.id
JOIN customers ON orders.id = customers.id

-- expected ORDER BY total_amount DESC LIMIT 5
ORDER BY total_amount ASC LIMIT 10
```

Also add positive controls for the correct clauses.

- [ ] **Step 2: Run the new tests and verify they fail against current verifier**

Run:
```bash
python -m pytest -q tests/test_generator_verifier.py
```
Expected: the incorrect SQL examples are currently accepted or fail with missing explicit-contract support.

- [ ] **Step 3: Implement conservative clause-aware checks**

Parse normalized clause slices (`WHERE`, `GROUP BY`, `ORDER BY`, `LIMIT`, `JOIN ... ON`) instead of searching the entire SQL string. Validate the required operator adjacent to the grounded filter column/literal, required group expressions inside the GROUP BY clause, required join equality inside an ON clause, required order field/direction inside ORDER BY, and exact required LIMIT value.

Do not add a full SQL parser dependency in this slice.

- [ ] **Step 4: Run focused verifier tests**

Expected: all focused tests PASS.

---

### Task 3: Preserve service behavior while exposing Structured without verification

**Files:**
- Modify: `src/semantic_sql/agent.py`
- Create: `src/semantic_sql/inference.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Produces:
```python
run_structured(..., verify: bool, max_repairs: int) -> AgentResult
```
(or an equivalently small internal seam) so Guarded and Structured use the same planner/grounding/generator implementation.
- Existing `SemanticSQLService.ask()` remains the Guarded public path.

- [ ] **Step 1: Add a failing test showing Structured executes the initial candidate without repair while Guarded preserves current repair behavior**

Use deterministic providers; assert both paths share the same initial generation semantics and differ only at verification/repair.

- [ ] **Step 2: Run focused test and confirm failure**

- [ ] **Step 3: Extract the smallest shared inference seam**

Do not create framework abstractions. Keep stage records and error semantics unchanged for the public service.

- [ ] **Step 4: Run agent tests and full P1 tests**

Run:
```bash
python -m pytest -q
```
Expected: all tests PASS.

---

### Task 4: Add a fair Direct baseline

**Files:**
- Create: `src/semantic_sql/direct.py`
- Test: `tests/test_direct.py`

**Interfaces:**
- Produces:
```python
generate_direct_sql(question: str, schema_context: str, provider: ModelProvider) -> SQLCandidate
```
- Direct generation receives the same catalog/schema descriptions made available to the structured planner/grounder benchmark harness.

- [ ] **Step 1: Add a failing provider-capture test**

Assert the direct prompt contains the question and supplied schema context, contains no gold SQL/answer, and returns a fence-stripped read-only candidate string.

- [ ] **Step 2: Run test and confirm failure**

- [ ] **Step 3: Implement the minimal Direct generator**

Reuse the existing provider and SQL fence stripping behavior. Do not add repair or verification to Direct.

- [ ] **Step 4: Run direct tests and full P1 tests**

---

### Task 5: Replace the placeholder evaluator with benchmark protocol utilities

**Files:**
- Modify: `evaluation/bird_loader.py`
- Modify: `evaluation/metrics.py`
- Modify: `evaluation/run_bird.py`
- Create: `evaluation/protocol.py`
- Test: `tests/test_evaluation_isolation.py`
- Create: `tests/test_bird_protocol.py`

**Interfaces:**
- Produces deterministic disjoint sampling utilities and per-case result records.
- CLI modes: `direct`, `structured`, `guarded` for DEV; final protocol can compare `direct` with one frozen winner.

- [ ] **Step 1: Add failing protocol-isolation tests**

Using tiny synthetic metadata only, prove:
- deterministic seed/hash selection is stable,
- DEV and FINAL case IDs are disjoint,
- labels/gold SQL are not passed into inference functions,
- per-case records contain arm, status, SQL, executability, latency, and result-correctness fields,
- final manifest records model/config/dataset identifiers supplied by the caller.

- [ ] **Step 2: Run tests and confirm failure**

- [ ] **Step 3: Implement protocol utilities and evaluator CLI**

External dataset/database paths are required inputs; no benchmark files are copied into the repo. Exact SQL string match may remain a diagnostic metric, but result correctness should be supplied by the benchmark execution comparator when available and clearly labeled when it is a proxy.

- [ ] **Step 4: Run evaluation tests and full suite**

---

### Task 6: Align documentation with evidence boundaries

**Files:**
- Modify: `README.md`
- Modify: `evaluation/README.md`
- Modify: `docs/architecture.md`

**Interfaces:**
- Documents only claims implemented by Tasks 1-5.

- [ ] **Step 1: Remove or qualify claims that the repo already contains a completed one-shot comparison if it does not**

- [ ] **Step 2: State clearly that demo fixtures are smoke tests and BIRD DEV/FINAL runs are the performance evidence path**

- [ ] **Step 3: Document Direct/Structured/Guarded and the no-retuning-after-final rule**

- [ ] **Step 4: Run docs/hygiene tests and full suite**

---

### Task 7: Final verification and package

**Files:**
- No new runtime files expected.

- [ ] **Step 1: Run compile verification**

```bash
python -m compileall -q src evaluation tests
```
Expected: exit 0.

- [ ] **Step 2: Run full tests**

```bash
python -m pytest -q
```
Expected: 0 failures.

- [ ] **Step 3: Inspect for benchmark leakage and generated junk**

Confirm no BIRD labels/answers are imported by runtime modules and no `.venv`, dataset, cache, or model artifacts are packaged.

- [ ] **Step 4: Build a clean ZIP containing source, tests, evaluation code, docs, and small demo fixtures only**
