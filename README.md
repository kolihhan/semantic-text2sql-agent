<h1 align="center">Semantic Text-to-SQL</h1>

<p align="center">
  <strong>Local NL → SQL with deterministic verification, bounded repair, and read-only execution.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-111827?style=flat-square&logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-111827?style=flat-square&logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Streamlit-111827?style=flat-square&logo=streamlit" alt="Streamlit" />
  <img src="https://img.shields.io/badge/LangGraph-111827?style=flat-square" alt="LangGraph" />
  <img src="https://img.shields.io/badge/SQLite-111827?style=flat-square&logo=sqlite" alt="SQLite" />
</p>

> [!NOTE]
> The point of this project is not just to generate SQL. It makes the **reliability layer around the model visible and measurable**.

## Demo

```cmd
run-demo.cmd
```

The Streamlit demo puts the whole path on one screen:

```text
Question
  ↓
Generate SQL
  ↓
Verify ✓ ── failure ──→ Repair (max 2)
  ↓                         │
Execute read-only ←─────────┘
  ↓
Result table + trace
```

Try:

```text
How many orders were placed in 2024?
```

The built-in offline provider can demonstrate the same service flow without Ollama. It exists for the product demo only; benchmark claims below come from the frozen local-model run.

## Measured result

Frozen paired evaluation on **100 BIRD DEV queries** using local `qwen3.5:4b` through Ollama:

| Metric | Direct | Guarded + repair |
|---|---:|---:|
| Execution success | 63% | **82%** |
| Official BIRD EX | 31% | **34%** |
| Median latency | **28.1 s** | 46.7 s |

The guarded path recovered **19 direct execution failures**. It improves execution reliability, but costs latency and does **not** guarantee semantic correctness.

Source of truth: [`runs/bird-paired-dev-v4/paired.json`](runs/bird-paired-dev-v4/paired.json).

## Architecture

```mermaid
flowchart LR
    Q[Question] --> G[Generate SQL]
    G --> V[Deterministic verification]
    V -->|pass| E[Read-only SQLite execution]
    V -->|fail| R[Bounded repair]
    E -->|execution error| R
    R --> V
    E --> O[Rows + columns + trace]
```

## Engineering choices

- **Deterministic checks before execution** instead of asking another model to judge obvious failures.
- **Read-only SQLite execution** so generated SQL cannot mutate the database.
- **Bounded repair** with `max_repairs=2`, avoiding an open-ended agent loop.
- **One service layer** behind Streamlit, FastAPI, and CLI.
- **Paired evaluation** on the same frozen BIRD cases.
- **Separate reliability from correctness**: execution success and official EX are reported independently.

## Quickstart

### Offline visual demo

```bash
uv sync
uv run streamlit run app.py
```

### Local-model path

```bash
ollama pull qwen3.5:4b
uv run streamlit run app.py
```

Choose **Ollama** in the sidebar.

### CLI

```bash
uv run semantic-sql ask "Which customer spent the most?"
```

### FastAPI

```bash
uv run uvicorn semantic_sql.api:app --reload
```

## Evaluation note

> [!IMPORTANT]
> **Execution success is not Text-to-SQL accuracy.** A query can execute successfully and still answer the wrong question. The 82% figure therefore represents execution reliability, while official BIRD EX is the correctness-oriented metric.

## Limits

- Frozen 100-query BIRD DEV sample, not the full benchmark.
- SQLite only in the current implementation.
- Results depend on the frozen local-model configuration.
- The offline provider is demo infrastructure, not evaluation evidence.
