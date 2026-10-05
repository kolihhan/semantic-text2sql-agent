# Semantic Sketch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in pre-SQL semantic sketch and validate it first on 20 frozen Qwen 9B failures, then on full frozen-100 only if the pilot gate passes.

**Architecture:** Reuse the existing provider and generation path. One new semantic-sketch helper parses a JSON intent contract; guarded generation optionally runs that helper before SQL generation and passes the sketch into the existing direct prompt. Evaluation reuses exact v5 inputs and prior Qwen 9B direct outputs as the paired baseline.

**Tech Stack:** Python 3.11, existing Ollama provider, pytest, GitHub Actions.

**Spec:** `docs/semantic-sketch-ablation.md`

## Global Constraints

- No new dependency.
- Existing behavior remains default; semantic analysis is opt-in until validated.
- Same model for pilot/full: `qwen3.5:9b-q4_K_M`.
- Pilot gate: positive net official BIRD EX and no execution-success regression.

## Review Focus

- Invalid/non-JSON analysis must fail closed to an empty sketch rather than crash generation.
- Sketch must not replace the original question/evidence/schema.
- Existing direct generation must remain byte-for-byte prompt-compatible when no sketch is supplied.
- Semantic layer must add exactly one analysis call before initial SQL generation, not before every repair.
- Evaluation must not use gold SQL in the semantic-analysis or generation prompt.

---

### Task 1: Semantic sketch contract

**Files:**
- Create: `src/semantic_sql/semantic_sketch.py`
- Create: `tests/test_semantic_sketch.py`

**Interfaces:**
- Produces: `SemanticSketch`, `analyze_semantics(...)`.

- [ ] Write failing tests for valid JSON parsing and invalid-response fail-closed behavior.
- [ ] Run CI and confirm RED because the module does not exist.
- [ ] Implement the minimal parser/prompt with no new dependencies.
- [ ] Run full test suite and confirm GREEN.

### Task 2: Wire sketch into SQL generation

**Files:**
- Modify: `src/semantic_sql/direct.py`
- Modify: `src/semantic_sql/inference.py`
- Modify: `src/semantic_sql/agent.py`
- Modify: `tests/test_direct.py`
- Modify: `tests/test_guarded_direct_preflight.py`

**Interfaces:**
- Consumes: `SemanticSketch`, `analyze_semantics(...)`.
- Produces: opt-in `semantic_analysis` path.

- [ ] Write failing tests proving the direct prompt includes a supplied sketch and guarded flow calls analysis once before generation.
- [ ] Run CI and confirm RED for missing wiring.
- [ ] Add the smallest opt-in wiring; leave existing default unchanged.
- [ ] Run full test suite and confirm GREEN.

### Task 3: Pilot and full benchmark workflow

**Files:**
- Create: `evaluation/semantic_sketch_ablation.py`
- Create: `tests/test_semantic_sketch_ablation.py`
- Create: `.github/workflows/qwen9b-semantic-sketch.yml`

**Interfaces:**
- Consumes: prior Qwen 9B direct outputs, exact v5 frozen source/cases, semantic sketch generation.
- Produces: pilot JSON; if gate passes, full-100 JSON.

- [ ] Write failing unit tests for deterministic pilot selection, paired aggregation, and gate logic.
- [ ] Run CI and confirm RED.
- [ ] Implement the smallest evaluation helpers/workflow.
- [ ] Run full test suite and confirm GREEN.
- [ ] Trigger pilot; inspect result.
- [ ] If gate passes, trigger/run full 100 and aggregate; otherwise stop and report pilot failure.
