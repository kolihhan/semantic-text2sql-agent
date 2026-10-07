# Design decisions

## Why one LangGraph workflow?

The project previously exposed two incompatible architectures: a SemanticPlan/grounding pipeline in the product and a direct-SQL Guarded graph in evaluation. That made benchmark evidence irrelevant to the shipped path. One graph now owns application, CLI, and benchmark orchestration, so tests and measurements describe the same system.

## Why direct SQL instead of a semantic planner?

The extra planner/grounder contracts created failure modes and duplicated architecture without validated incremental benefit. The model now receives the physical schema and emits SQL directly. Deterministic code then checks only properties it can know.

## Why optional evidence?

BIRD cases provide external evidence; ordinary application requests may not. Evidence is therefore request-scoped and optional, not a hidden product dependency.

## Why verify before execution?

Preflight can reliably reject writes, malformed SQL, unresolved schema references, and invalid execution boundaries. It cannot prove intent correctness. Keeping that boundary explicit prevents safety checks from being marketed as a semantic verifier.

## Why bounded repair and fail-closed termination?

Repair can recover an invalid query, but model calls must be finite and observable. The graph allows at most two repairs in the evaluated treatment and refuses after the budget is exhausted. Every repair is reflected in trace, latency, and call counts.

## Why a shared initial candidate in evaluation?

Two independent generations confound the treatment with sampling variance. The paired runner generates once, freezes the candidate, sends the same bytes to Direct and Guarded, and counts that shared call and latency in both arms. Only deterministic verification and any repair calls differ.

## Why a BIRD EX-compatible execution-match scorer?

The older local result proxy was not suitable for a portfolio correctness claim. The current runner follows the set-of-tuples execution-match semantics of the pinned BIRD evaluator revision and fails closed on truncated executions. Evaluator URL and revision are stored in the artifact. The implementation is project-local rather than a claim that the upstream evaluator is invoked directly at runtime.

## Why SIMPLIFY GUARDED? — historical 4B decision

The original frozen 100-case DEV decision used the earlier 4B configuration. In that run, Guarded improved the BIRD EX-compatible match from 30% to 33% and execution success from 70% to 88%, with three favorable EX transitions and none in the reverse direction. It also increased model calls from 100 to 150, median latency from 3.236s to 5.491s, and p95 latency from 79.398s to 147.364s.

The mechanism clearly improved executability, but only 3 of 30 Direct execution failures became correct. The decision was therefore to keep the minimal LangGraph safety/recovery skeleton while rejecting a stronger claim that the full automatic-repair treatment clearly earns its cost. Direct remains the cost baseline; repair is a narrow bounded escalation, not semantic proof.

The later checked-in 9B frozen summary reports a separate 33% → 39% EX-compatible transition and 75% → 90% conservative execution-success transition. Those figures are not a replacement for the historical 4B decision record; they belong to the newer 9B artifact and carry the workflow-provenance limitation documented in the root README.

## Why are demos and failed runs not results?

Fixtures protect wiring and regression behavior only. The first live attempt timed out after 20 cases and is preserved as an `incomplete` operational artifact, not merged into the historical result. Failed, incomplete, and backup artifacts remain provenance, not headline evidence.
