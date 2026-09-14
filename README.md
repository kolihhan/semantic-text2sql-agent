# Semantic Text-to-SQL Reliability

A local LangGraph Text-to-SQL workflow focused on **safe execution, bounded recovery, and measurable trade-offs**. Generated SQL is checked with SQLite-native guards, executed read-only, and repaired at most twice when execution fails.

## What it does

```text
question + schema + optional evidence
                 |
              generate
                 |
               verify
              /      \
         execute     repair
                       |
                     verify
                    /      \
               execute    refuse
```

- **LangGraph** owns state, conditional routing, bounded repair, termination, and stage traces.
- **SQLite/native code** owns read-only enforcement, compilation/schema checks, `EXPLAIN QUERY PLAN`, bounded execution, and the authorizer.
- **The model** owns SQL generation and repair.
- BIRD evidence is optional runtime context: the benchmark supplies it, while the normal application path does not require it.

## Measured result

Frozen paired DEV evaluation: **100 manifest-selected BIRD DEV cases** with `qwen3.5:4b`.

| Metric | Direct | Guarded | Delta |
|---|---:|---:|---:|
| Official BIRD EX | 30/100 (30%) | 33/100 (33%) | +3 pp |
| Execution success | 70/100 (70%) | 88/100 (88%) | +18 pp |
| Total model calls | 100 | 150 | +50% |
| Median latency | 3.236 s | 5.491 s | 1.70x |
| P95 latency | 79.398 s | 147.364 s | 1.86x |

The guard recovered **18 execution failures**, but only **3** became correct under official BIRD EX. Fifty extra repair calls bought three additional correct answers: **16.7 extra calls per added correct case**.

**Decision: `SIMPLIFY GUARDED`.** Keep the deterministic safety boundary, bounded repair, traceability, and fail-closed behavior. Do not treat executability as semantic correctness, and do not restore the earlier planner-heavy architecture.

## Why this project exists

A model can emit SQL that is unsafe, invalid, or executable-but-wrong. Deterministic verification is good at establishing safety and executability; it cannot prove that the query answers the user's intent.

This project tests the useful middle ground: how much reliability a small local Text-to-SQL workflow gains from deterministic checks and bounded repair, and what that recovery costs in model calls and latency.

## Evaluation design

Each benchmark case generates SQL **once** from the question, schema, and optional BIRD evidence. That byte-identical candidate then enters two arms:

- **Direct:** execute the shared candidate under the same SQLite/resource limits.
- **Guarded:** verify the shared candidate, execute it when accepted, or attempt at most two repairs before refusing.

Both arms include the shared generation call and latency; Guarded additionally includes repair work. Correctness uses the pinned [official BIRD evaluator contract](https://github.com/AlibabaResearch/DAMO-ConvAI/blob/483554eae102996f5ec1f4feab4e78ef29c2a394/bird/llm/src/evaluation.py), including set-of-tuples result comparison.

The first operational attempt stopped after 20 cases when its 180-second HTTP read timeout expired. That incomplete checkpoint was archived and excluded. The replacement run restarted all 100 cases from scratch with a recorded 600-second transport timeout.

Valid paired DEV artifact: `runs/bird-paired-dev-v3/paired.json`  
SHA-256: `309b20b08ba682e6cd9df5da16ddb2873022088dcd927632700c360fceea884a`

Review-v2 keeps the same LangGraph workflow and changes only the Guarded repair prompt to the predeclared internal step-by-step CoT treatment. Its output path is `runs/bird-paired-dev-v4/paired.json`; no v4 result is claimed until rerun.

## Run locally

```bash
uv sync
uv run semantic-sql demo
uv run semantic-sql --provider ollama --model qwen3.5:4b ask "Show total sales from Czech Republic in 2024"
uv run streamlit run app.py
```

The bundled demo is a wiring smoke test, not benchmark evidence. BIRD data, databases, model weights, virtual environments, and caches are external prerequisites and are not vendored.

## Limitations

- The result is one unseeded local-model run on 100 DEV cases; three favorable wrong-to-correct transitions are not strong evidence of a general semantic-quality improvement.
- Preflight establishes safety and executability, not user-intent correctness; valid-but-wrong SQL can pass.
- Latency depends on the local Ollama runtime and hardware. The recorded values are paired-run evidence, not universal service-level claims.
- The sealed FINAL/holdout split was not inspected or run, so this is an architecture decision rather than a leaderboard claim.
- Historical modules and artifacts remain for provenance and should not be mistaken for supported product paths.

## Development

```bash
uv run --extra dev pytest -q -p no:cacheprovider --basetemp <fresh-temp-directory>
uv run python -m compileall -q src evaluation tests
```

See `docs/architecture.md`, `docs/design-decisions.md`, `docs/architecture/p1-dev-decision.md`, and `evaluation/README.md`.
