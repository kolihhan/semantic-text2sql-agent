# Semantic Text-to-SQL

A local SQL agent built around one principle: **a query that runs is not necessarily a query that answers the question.**

It uses Qwen3.5, the actual SQLite schema, deterministic validation, and bounded recovery to turn natural-language questions into safe, executable SQL.

[![CI](https://github.com/kolihhan/semantic-text2sql-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/kolihhan/semantic-text2sql-agent/actions/workflows/ci.yml)

## Demo

Run the offline Streamlit demo without a model or API key:

\`\`\`bash
git clone https://github.com/kolihhan/semantic-text2sql-agent.git
cd semantic-text2sql-agent
python -m pip install -e ".[dev]"
streamlit run app.py
\`\`\`

To use a local model instead:

\`\`\`bash
ollama pull qwen3.5:9b
semantic-sql --provider ollama --model qwen3.5:9b ask "Which customer spent the most?"
\`\`\`

The demo displays the generated SQL, validation steps, and returned rows. The same service powers the CLI, Streamlit UI, and FastAPI endpoint.

\`\`\`text
Question
   ↓
Schema-grounded SQL generation
   ↓
Deterministic preflight ── invalid ─→ Bounded repair ─┐
   ↓ valid                                           │
Read-only execution ←─────────────────────────────────┘
   ├── Rows → Answer
   ├── Empty → Optional, verified recheck
   └── Error → Bounded recovery or recorded failure
\`\`\`

The offline provider is a deterministic demonstration, **not** a benchmark model.

## Results

### Frozen BIRD DEV100

The original paired experiment used **100 fixed BIRD DEV questions**, local Qwen3.5-9B Q4_K_M through Ollama, and the same evaluation protocol for both approaches.

| Metric | Direct SQL | Verified + bounded repair |
|---|---:|---:|
| Execution-match correctness | 33/100 | **39/100** |
| Conservative execution success | 75/100 | **90/100** |
| Wrong → correct | — | **6** |
| Correct → wrong | — | **0** |

The initial reliability pipeline recovered six wrong answers without changing a previously correct answer in this sample. The execution-success count is deliberately conservative: the raw figure was 91/100, but one capped result is excluded for strict comparison.

Source: [frozen DEV100 artifact](runs/qwen9b-frozen100/summary.json). Its original workflow was not a fully clean end-to-end run; the checked-in artifact is the source for these historical numbers.

### Matched BIRD DEV300

A later experiment evaluated **300 fixed BIRD DEV questions** with the same Qwen3.5-9B Q4_K_M model and a shared scoring protocol. All three runs used a three-call maximum.

| Approach | Correct / 300 | Execution success | Model calls |
|---|---:|---:|---:|
| Direct SQL | 148 (**49.33%**) | 278 | 300 |
| Verified + bounded repair | 151 (**50.33%**) | **294** | **332** |
| Structured generation + recovery variant | **161 (53.67%)** | 274 | 392 |

The structured recovery variant recovered **33** cases that the guarded baseline missed, but lost **23** that the guarded baseline answered correctly: a net gain of 10, with **60 additional model calls**. The improvement is not statistically established on this 300-question sample, and it comes with fewer executable queries.

This was an **experimental variant**, not a benchmark of the current merged service. The main branch integrates bounded structured recovery, but has **not** been rerun on DEV300. Do not treat 53.67% as a production score, or compare DEV100 and DEV300 percentages as if they used identical cohort and runtime configurations.

Source: [successful GPU run](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/45748311494) · [DEV300 scorecard and protocol](docs/dev300-results.md).

## Two failures worth inspecting

### 1. Valid SQL can still be wrong

On a previously inspected BIRD case, the query compiled, passed verification, and returned **23 rows**, but failed the execution-match correctness test. SQLite proved that the query could run—not that it captured the user’s intent.

That is why the project tracks **correctness** and **execution success** as separate metrics. [Failure analysis in the original frozen study](runs/qwen9b-frozen100/summary.json).

### 2. Recovery can improve accuracy and reduce reliability

The matched DEV300 run exposed a trade-off: structured recovery reached **161 correct answers** versus **151**, but produced only **274 executable outputs** versus **294**. Its 33 rescues came with 23 regressions.

A recovery mechanism needs a bounded call budget, deterministic safety checks, and a record of both rescued and broken cases. More model calls alone are not a correctness guarantee.

## Architecture

\`\`\`mermaid
flowchart LR
    Q[Question] --> S[Native SQLite schema]
    S --> G[Qwen3.5 SQL generation]
    G --> V{Read-only preflight}
    V -->|Invalid| R[Bounded repair]
    R --> V
    V -->|Valid| E[SQLite execution]
    E -->|Nonempty| O[Rows + trace]
    E -->|Empty or error| C[Optional recovery]
    C --> V
    V -->|Budget exhausted| F[Recorded failure]
\`\`\`

The pipeline keeps one LangGraph service and one read-only execution boundary. The local Ollama provider can request schema-constrained JSON responses; the offline demo and text-only providers retain their original interface. A recovery candidate must pass the same checks before replacing a query.

## Engineering choices

- **Native SQLite DDL grounding:** use real stored table definitions instead of imagined tables or columns.
- **Deterministic preflight:** check the statement and compile it before execution, without spending another model call.
- **Read-only execution:** SQLite read-only connections and an authorizer reject writes.
- **Bounded repair:** at most two repair opportunities after the first generation; no unrestricted loops.
- **Conservative empty-result recovery:** preserve the existing answer unless a safe alternative executes successfully and returns rows.
- **Paired evaluation:** separate answer correctness, execution success, model calls, rescues, and regressions.

## Reproduce

Run the CPU tests without downloading a language model:

\`\`\`bash
python -m pip install -e ".[dev]"
python -m pytest -q
\`\`\`

The frozen benchmark files and evaluation scripts are kept in the repository. The 300-question GPU experiment also has an [immutable Actions run and downloadable raw outputs](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/45748311494). Reproducing those numbers requires the recorded model and runtime settings, not just the Streamlit demo.

## Evaluation note

The system currently targets SQLite. The historical DEV100 and DEV300 experiments used different frozen cohorts and configurations. The integrated main-branch recovery code is tested for correctness and safety, but its semantic EX has not been independently measured yet. **Executing successfully is not the same as answering correctly.**
