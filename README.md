# Semantic Text-to-SQL Reliability

One small LangGraph workflow tests a narrow engineering question: does deterministic preflight plus bounded model repair improve local Text-to-SQL enough to justify its cost?

## Problem

A model can emit SQL that is unsafe, invalid, or executable-but-wrong. Native SQLite checks can establish read-only safety, compilation, schema resolution, and execution boundaries; they cannot prove that a query answers the user's intent. This project measures the useful middle ground instead of presenting a verifier as a semantic oracle.

## Architecture

The application, CLI, and benchmark use one authoritative LangGraph workflow:

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

LangGraph owns state, conditional routing, the bounded repair loop, termination, and stage traces. SQLite/native code owns read-only enforcement, compilation/schema checks, `EXPLAIN QUERY PLAN`, bounded execution, and the authorizer. The model owns generation and repair.

BIRD evidence is optional runtime context: the benchmark supplies it, while the normal product path does not require it. The former SemanticPlan, grounding framework, planner-driven flow, semantic regex verifier, and duplicate graph are retired from primary runtime use; dated design documents and old benchmark artifacts are provenance only.

## Experiment

The frozen benchmark uses 100 manifest-selected BIRD DEV cases and `qwen3.5:4b`. Each case generates SQL exactly once from the question, schema, and optional BIRD evidence. That byte-identical candidate then enters two arms:

- **Direct:** execute the shared candidate under the same SQLite/resource limits.
- **Guarded:** verify the shared candidate, execute it when accepted, or attempt at most two repairs before refusing.

Both arms include the shared generation call and latency; Guarded additionally includes repair work. Correctness uses the pinned [official BIRD evaluator contract](https://github.com/AlibabaResearch/DAMO-ConvAI/blob/483554eae102996f5ec1f4feab4e78ef29c2a394/bird/llm/src/evaluation.py), including set-of-tuples result comparison. No FINAL/holdout selector is exposed by the primary runner.

The first operational attempt stopped after 20 cases when its 180-second HTTP read timeout expired. Its `incomplete` checkpoint was archived and excluded. The replacement run restarted all 100 cases from scratch with a recorded 600-second transport timeout; no partial generations were selected into the valid result.

## Measured result

Valid paired DEV artifact: `runs/bird-paired-dev-v3/paired.json`  

The v3 artifact is preserved historical evidence. Review-v2 keeps LangGraph but changes only the Guarded repair prompt to the predeclared internal step-by-step CoT treatment; its new output path is `runs/bird-paired-dev-v4/paired.json` and is not claimed until rerun.  
SHA-256: `309b20b08ba682e6cd9df5da16ddb2873022088dcd927632700c360fceea884a`

| Metric | Direct | Guarded | Delta |
|---|---:|---:|---:|
| Official BIRD EX | 30/100 (30%) | 33/100 (33%) | +3 pp |
| Execution success | 70/100 (70%) | 88/100 (88%) | +18 pp |
| Total model calls | 100 | 150 | +50% |
| Median latency | 3.236 s | 5.491 s | 1.70x |
| P95 latency | 79.398 s | 147.364 s | 1.86x |

Paired transitions were 3 Direct-wrong to Guarded-correct, 0 Direct-correct to Guarded-wrong, 18 Direct-execution-fail to Guarded-success, and 0 reverse execution regressions. Every Direct candidate used one model call, every initial SQL pair was identical, and Guarded call counts matched `1 + repair_attempts`.

## Operational cost and tradeoff

Thirty Direct candidates failed execution. Repair made 18 executable, but only 3 became correct under official EX. Across the sample, 50 extra repair calls bought three added correct answers: 16.7 extra calls per added correct case. The guard is effective at recovering executability, but executability is a weak proxy for intent correctness, and its long-tail latency is substantial.

The zero correct-to-wrong count should not be oversold: accepted candidates pass through unchanged, while repair is triggered on candidates that already failed execution and therefore could not pass EX.

## Decision

**SIMPLIFY GUARDED.** Keep LangGraph as the orchestration framework and keep the single minimal, fail-closed workflow, deterministic SQLite safety boundary, bounded repair, and traceability. Do not restore the semantic-planning/grounding architecture or claim that deterministic verification establishes semantic correctness.

The measured repair loop remains reproducible as the evaluated treatment, but the evidence does not support presenting broad automatic repair as a clear default winner: the +3-point EX gain is small relative to calls and tail latency. Treat repair as a narrow recovery/escalation mechanism and preserve the simpler Direct path as the cost baseline.

## Run locally

```bash
uv sync
uv run semantic-sql demo
uv run semantic-sql --provider ollama --model qwen3.5:4b ask "Show total sales from Czech Republic in 2024"
uv run streamlit run app.py
```

The bundled demo is a wiring smoke test, not benchmark evidence. BIRD data, databases, model weights, virtual environments, and caches are external prerequisites and are not vendored.

## Limitations

- The result is one unseeded local-model run on 100 DEV cases; three favorable discordant pairs are not strong evidence of a general semantic-quality improvement.
- Preflight establishes safety and executability, not user-intent correctness; valid-but-wrong SQL can pass.
- Latency depends on the local Ollama runtime and hardware. The recorded values are useful for the paired run, not universal service-level claims.
- The sealed FINAL/holdout split was not inspected or run, so this is an architecture decision rather than a leaderboard claim.
- Historical modules and artifacts remain for provenance and should not be mistaken for supported product paths.

## Development

```bash
uv run --extra dev pytest -q -p no:cacheprovider --basetemp <fresh-temp-directory>
uv run python -m compileall -q src evaluation tests
```

See `docs/architecture.md`, `docs/design-decisions.md`, `docs/architecture/p1-dev-decision.md`, and `evaluation/README.md`.
