# Design decisions

> [!NOTE]
> The current portfolio headline is the frozen **Qwen3.5 9B** paired result: BIRD EX **33% → 39%** and conservative execution success **75% → 90%**. The earlier **Qwen3.5 4B** 30% → 33% / 70% → 88% result remains useful historical evidence for the original guarded-design decision, but it is no longer the canonical headline result.

## Why one LangGraph workflow?

The project previously exposed two incompatible architectures: a SemanticPlan/grounding pipeline in the product and a direct-SQL Guarded graph in evaluation. That made benchmark evidence irrelevant to the shipped path. One graph now owns application, CLI, and benchmark orchestration, so tests and measurements describe the same system.

## Why direct SQL instead of a semantic planner?

The extra planner/grounder contracts created failure modes and duplicated architecture without validated incremental benefit. The model now receives the physical schema and emits SQL directly. Deterministic code then checks only properties it can know.

## Why schema grounding by default?

The default service passes compact physical-schema context into generation. Richer value-grounding and decomposed schema-linking ideas remain experiments rather than default behavior: the frozen XiYan DGSL v1 ablation improved some cases but regressed the same number and produced no net correctness gain.

## Why optional evidence?

BIRD cases provide external evidence; ordinary application requests may not. Evidence is therefore request-scoped and optional, not a hidden product dependency.

## Why verify before execution?

Preflight can reliably reject writes, malformed SQL, unresolved schema references, and invalid execution boundaries. It cannot prove intent correctness. Keeping that boundary explicit prevents safety checks from being marketed as a semantic verifier.

## Why bounded repair and fail-closed termination?

Repair can recover an invalid query, but model calls must be finite and observable. The graph allows at most two repairs in the evaluated treatment and refuses after the budget is exhausted. Every repair is reflected in trace, latency, and call counts.

## Why a shared initial candidate in evaluation?

Two independent generations confound the treatment with sampling variance. The paired runner generates once, freezes the candidate, sends the same bytes to Direct and Guarded, and counts that shared call and latency in both arms. Only deterministic verification and any repair calls differ.

## Why BIRD EX-compatible scoring?

The older local result proxy was not suitable for a portfolio correctness claim. The repository's local BIRD EX scorer follows the set-of-result-rows comparison semantics of the BIRD evaluator pinned at revision `483554eae102996f5ec1f4feab4e78ef29c2a394`, while keeping the scorer implementation local and inspectable. The README therefore describes this as BIRD EX-compatible scoring rather than claiming that the benchmark directly invokes upstream evaluator code.

## Why keep Guarded bounded instead of adding more semantic machinery?

The current frozen Qwen3.5 9B result improved BIRD EX from **33% to 39%** and conservative execution success from **75% to 90%**, with six favorable EX transitions and no correct-to-wrong EX regressions. A later one-pass semantic-revision ablation moved guarded EX from 39% to 38%, with zero rescues and one regression, so semantic revision is not enabled by default.

The earlier Qwen3.5 4B run reached 30% → 33% EX and 70% → 88% execution success. That historical run also made the cost trade-off visible: model calls increased from 100 to 150, median latency from 3.236s to 5.491s, and p95 latency from 79.398s to 147.364s. Only 3 of 30 Direct execution failures became correct.

Together, the evidence supports the same narrow design choice: keep deterministic verification and bounded repair as a reliability layer, but do not market repair as semantic proof and do not add another semantic agent layer without positive net evidence.

## Why is the Qwen 9B workflow not described as a clean pass?

Provenance workflow run `37307108955` completed every case in its failing 25-case shard, but strict validity marked that shard invalid after one capped guarded result and the job exited non-zero. The checked-in frozen summary explicitly reports the conservative 90/100 guarded execution figure. The repository therefore treats the summary as the reporting artifact while disclosing that the overall Actions run concluded `failure` rather than presenting it as a clean successful benchmark run.

## Why are demos and failed runs not results?

Fixtures protect wiring and regression behavior only. Earlier incomplete live attempts remain operational/debugging artifacts and are not merged into headline metrics. Frozen result claims must be tied to an explicit reporting artifact, with invalid or conservative rows disclosed rather than silently repaired into a cleaner story.
