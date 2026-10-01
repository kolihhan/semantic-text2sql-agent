# Semantic Text-to-SQL Agent

A local Text-to-SQL service that treats LLM-generated SQL as a software-reliability problem: generate a query, validate it deterministically, execute it against SQLite, and use a bounded repair loop when the first attempt fails.

## What it does

```text
natural-language question
        ↓
LLM generates SQL
        ↓
deterministic verification
        ↓
read-only SQLite execution
        ↓
execution error?
   ├─ no  → return result
   └─ yes → bounded repair → verify → retry
```

The service is exposed through FastAPI and uses a local `qwen3.5:4b` model through Ollama.

## Frozen evaluation

The latest frozen run evaluates **100 BIRD DEV queries** and compares direct generation with a guarded workflow that adds deterministic verification and at most two repair attempts.

| Metric | Direct | Guarded + repair |
|---|---:|---:|
| Execution success | 63% | **82%** |
| Official BIRD EX | 31% | **34%** |
| Median latency | 28.1 s | 46.7 s |

The guarded workflow recovered **19 direct execution failures**. It also increased latency and model calls, so the result is a reliability trade-off rather than a claim that repair universally improves answer quality.

### Execution success is not correctness

These two metrics are intentionally reported separately:

- **Execution success**: the final SQL executed successfully against SQLite.
- **Official BIRD EX**: the executed result matched the benchmark answer under the official BIRD evaluator.

A query can execute successfully and still be wrong. The benchmark therefore reports both.

Source of truth: [`runs/bird-paired-dev-v4/paired.json`](runs/bird-paired-dev-v4/paired.json).

## Reliability mechanisms

- deterministic SQL verification before execution
- read-only SQLite execution
- bounded repair budget (`max_repairs=2`)
- structured failure states and per-case traces
- official BIRD evaluator for correctness
- frozen run artifacts with model/evaluator identity and latency

## Stack

- Python
- FastAPI
- LangGraph
- SQLite
- Qwen3.5 4B via Ollama
- pytest

## Why this project exists

The goal is not just to produce SQL. It is to make failure behavior measurable:

- Does the generated query compile?
- Can it be executed safely?
- Can a bounded repair step recover execution failures?
- Does higher execution reliability improve benchmark correctness?
- What latency does the reliability layer cost?

## Limits

- The evaluation is a 100-query BIRD DEV sample, not the full benchmark.
- Execution success is not semantic correctness.
- The current workflow targets SQLite rather than arbitrary production databases.
- Results are specific to this frozen local-model configuration.
