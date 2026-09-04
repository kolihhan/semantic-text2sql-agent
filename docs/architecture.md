# Architecture

`SemanticSQLService` is the product boundary and delegates to one authoritative LangGraph workflow. The CLI and Streamlit app are thin adapters over that service. The paired benchmark invokes the same Guarded workflow from a frozen initial candidate and gives Direct the same candidate without the Guarded treatment.

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

Gold SQL is evaluation-only. The runner generates both arm outputs before scoring, uses the pinned official BIRD set-of-tuples execution-match contract, and records calls and latency including shared generation. Truncation of a prediction or gold result invalidates the artifact rather than silently scoring partial rows.

Only manifest-selected DEV IDs are runnable from the primary CLI. Existing output paths are refused, atomic checkpoints remain `incomplete` after a crash, and the benchmark records model/runtime/evaluator provenance.

## Historical boundary

The former SemanticPlan, GroundedPlan, planner, grounder, semantic regex verifier, and duplicate Guarded graph are not part of supported application, CLI, or benchmark control flow. Dated plans and older benchmark artifacts are retained only to explain project history. The current architecture and decision are documented here and in `docs/architecture/p1-dev-decision.md`.
