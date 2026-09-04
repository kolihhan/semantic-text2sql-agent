# P1 Benchmark-First Text-to-SQL Design

> **Historical, superseded design record.** This SemanticPlan/grounding proposal is not the current product architecture or benchmark protocol. See `../../architecture.md` and `../../architecture/p1-dev-decision.md` for the implemented one-graph system and frozen paired DEV decision.

## Goal

Turn the current demo-oriented semantic SQL pipeline into a benchmark-oriented reliability experiment whose claims are supported by real BIRD cases rather than handcrafted demo fixtures.

## Portfolio claim

The project tests whether a structured semantic representation plus deterministic verification/repair improves a small model's Text-to-SQL reliability over direct SQL generation. Complexity is kept only if it wins on a fixed development comparison and transfers to an unseen final subset.

## Non-goals

- Do not optimize the demo domain.
- Do not add LangGraph, agents, critics, multi-model voting, or broad orchestration.
- Do not vendor BIRD data or model weights into the repository.
- Do not tune on the final holdout.
- Do not claim official BIRD leaderboard results unless the official protocol is actually reproduced.

## Current problems to correct

1. `SemanticPlan.metric` is a free-form demo concept but the shipped demo/provider collapses questions into a tiny analytics vocabulary; this is not a general enough contract for BIRD.
2. `order_by` and `time_constraint` exist in `SemanticPlan` but disappear during grounding.
3. The verifier infers aggregation from words such as `count`, `sales`, or `spend` instead of an explicit aggregation field.
4. The verifier checks filter columns/literals but not comparison operators.
5. GROUP BY and JOIN checks are substring-level and can accept structurally wrong clauses.
6. The project evaluation entry point does not implement a fair Direct vs Structured vs Guarded benchmark.
7. Demo fixtures are useful only as smoke/regression tests and must not be used as portfolio evidence.

## Architecture

### Semantic representation

Use explicit query operations instead of demo-specific metric names:

- `SemanticSelect(field, aggregation)` where `aggregation` is one of `none`, `count`, `sum`, `avg`, `min`, `max`.
- `SemanticFilter(field, operator, value)`.
- `SemanticOrder(field, direction)`.
- `SemanticPlan(select, filters, group_by, order_by, limit)`.

The representation remains database-agnostic: semantic fields are names/descriptions, not physical table/column identifiers.

### Grounded representation

Ground every semantic field independently to a physical `(table, column)` and preserve:

- select aggregation,
- filter operator/value,
- grouping,
- ordering direction,
- explicit limit,
- temporal predicates represented as ordinary filters where possible.

The grounded contract, not prompt prose, is the source of truth for SQL generation and deterministic verification.

### Verification scope

The verifier is a focused invariant verifier, not a SQL theorem prover. It must reject at least:

- wrong aggregation,
- missing/incorrect comparison operator for required filters,
- missing required literal,
- missing filter column,
- missing or incorrect required GROUP BY expression,
- missing or incorrect required join equality,
- missing/incorrect ORDER BY expression/direction,
- missing/incorrect LIMIT when required.

Verification may use conservative SQL text parsing/regex for SQLite, but every claimed invariant must have a regression test that demonstrates an incorrect query is rejected.

### Experiment seam

Expose three comparable inference arms without duplicating runtime logic:

- **Direct:** question + same schema context -> model -> SQL -> same read-only executor.
- **Structured:** question -> semantic plan -> grounding -> SQL -> executor; no verifier/repair.
- **Guarded:** Structured + verifier + bounded repair.

All arms use the same provider/model, database, schema context, decoding settings, row limit, and result comparator.

### Evaluation protocol

Development and final evidence are separate:

1. Existing/frequently inspected BIRD-derived cases are development-only.
2. Run Direct/Structured/Guarded on DEV to choose the architecture.
3. Freeze prompts/contracts/configuration.
4. Build a deterministic disjoint FINAL subset from remaining eligible BIRD cases.
5. Run Direct vs the frozen winner once on FINAL.
6. Persist per-case outputs, run configuration, and aggregate results.

No benchmark data is committed. Dataset roots are external CLI/config inputs.

## Result interpretation

The project succeeds as evidence even if the complex system loses. If Direct is as good or better after accounting for latency/model calls, simplify the runtime and report that result. Do not tune until Structured/Guarded wins.

## Testing policy

- Existing demo tests remain smoke/regression tests only.
- Add focused contract/verifier tests before implementation changes.
- Add benchmark-harness tests using tiny synthetic stubs only to validate protocol isolation; they do not become reported performance evidence.
