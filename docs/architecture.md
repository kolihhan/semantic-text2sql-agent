# Architecture

`SemanticSQLService` is the product boundary shared by CLI, Streamlit and FastAPI. For local Ollama models it uses the structured SQL pipeline with a bounded three-call budget, explicit query/refusal/clarification outputs, read-only AST validation, error correction, and an empty-result recheck. The existing LangGraph Guarded workflow remains available for the offline demo and explicit `mode="guarded"` consumers. Historical paired BIRD runners preserve their original Guarded baseline and frozen initial-candidate contract.

```text
Normal product                         Paired DEV evaluation
question + catalog                    question + schema + BIRD evidence
        |                                         |
        +---- optional evidence                    |
        |                                         v
        |                                  generate exactly once
        v                                         |
  LangGraph generate                              +---- Direct: execute
        |                                         |
      verify                                      +---- Guarded: same graph,
     /      \                                            frozen candidate
execute      repair
               |
             verify
            /      \
       execute    refuse
```


## Product routing

`SemanticSQLService` chooses the structured engine automatically when the
provider supports `complete_messages` and the request does not inject a frozen
SQL candidate. `mode="structured"` explicitly requires that transport;
`mode="guarded"` forces the original LangGraph path. The experimental
semantic-revision path remains on the original graph. Structured mode caps
ordinary validation repairs at one and caps all LLM calls at three. In both
modes, the final output passes the project's read-only SQLite verification
before reaching CLI/API/UI callers. The included offline demo uses the original
deterministic provider with no model download.

The integrated runtime preserves the original structured-engine policies,
but the previously reported DEV300 result is an experimental evaluation,
not a new product-level end-to-end benchmark.

## Graph state and context

The dynamic state contains the question, optional evidence, current SQL candidate, verification outcome, execution result, terminal status, and trace stages. Nodes return partial state updates.

Static dependencies live in LangGraph runtime context: the database path, model provider, repair limit, and row limit. The schema context is derived from the catalog by the service and passed as request data. A frozen initial candidate can be supplied by the benchmark so Guarded does not generate a second starting query.

## Responsibility boundary

LangGraph owns:

- state and conditional routing;
- bounded repair and termination;
- execute-versus-refuse decisions;
- traceable stage transitions.

SQLite/native deterministic code owns:

- URI `mode=ro`, `query_only`, and authorizer enforcement;
- one read-only `SELECT`/CTE boundary;
- compilation and referenced-schema validation;
- `EXPLAIN QUERY PLAN` where useful;
- bounded execution and truncation detection.

The model owns initial SQL generation and repair. Deterministic verification does not claim semantic equivalence to the user's request.

## Entry points

- `src/semantic_sql/agent.py`: supported `SemanticSQLService` application boundary.
- `src/semantic_sql/inference.py`: the sole compiled LangGraph and its runner.
- `src/semantic_sql/cli.py`: `demo` and `ask` commands through the service.
- `app.py`: Streamlit adapter through the service.
- `evaluation/run_bird.py`: paired DEV runner using one shared candidate and the same Guarded graph.

## Evaluation boundary

Gold SQL is evaluation-only. The runner generates both arm outputs before scoring, uses a project-local scorer that follows the pinned BIRD set-of-tuples execution-match contract, and records calls and latency including shared generation. Truncation of a prediction or gold result invalidates the artifact rather than silently scoring partial rows.

Only manifest-selected DEV IDs are runnable from the primary CLI. Existing output paths are refused, atomic checkpoints remain `incomplete` after a crash, and the benchmark records model/runtime/evaluator provenance.

## Historical boundary

The former SemanticPlan, GroundedPlan, planner, grounder, semantic regex verifier, and duplicate Guarded graph are not part of supported application, CLI, or benchmark control flow. Dated plans and older benchmark artifacts are retained only to explain project history. Historical 4B metrics and the newer 9B checked-in summary are separate evidence generations and should not be mixed. The current architecture and decision are documented here, in `docs/architecture/p1-dev-decision.md`, and in the root README provenance note.
