# Semantic Text-to-SQL

A local Text-to-SQL service built with LangGraph, FastAPI, and SQLite.

The flow is simple: generate SQL, check it, run it read-only, and retry up to two times if the database returns an error. The CLI and API both use the same service layer.

## Results

I compared a direct path with the guarded path on 100 BIRD DEV questions using `qwen3.5:4b`.

| Metric | Direct | Guarded |
|---|---:|---:|
| Official BIRD EX | 30/100 | 33/100 |
| Execution success | 70/100 | 88/100 |
| Model calls | 100 | 150 |
| Median latency | 3.236 s | 5.491 s |
| P95 latency | 79.398 s | 147.364 s |

The repair loop recovered 18 failed executions, but only 3 of those became correct under the BIRD evaluator. In other words, getting SQL to run is useful, but it is not the same as getting the answer right.

That is why the current version keeps the database checks and small repair loop instead of adding more planner logic.

## How it works

```text
question + schema
      |
   generate
      |
    verify
   /      \
run      repair
           |
         verify
        /      \
      run     refuse
```

LangGraph handles the workflow. SQLite handles the database checks and read-only execution. The model handles SQL generation and repair.

## Run locally

```bash
uv sync
uv run semantic-sql demo
uv run streamlit run app.py
uv run uvicorn semantic_sql.api:app --reload
```

The default API demo does not need a running LLM server. For local model use, the CLI supports Ollama and `qwen3.5:4b`.

## Notes

The direct and guarded runs start from the same initial SQL for each question. The 100-case result is one local-model DEV run, so I treat the small EX gain cautiously. The final/holdout split has not been run.

The result artifact is in `runs/bird-paired-dev-v3/paired.json`. More detail on the workflow and evaluation is in `docs/architecture.md`, `docs/design-decisions.md`, and `evaluation/README.md`.

## Development

```bash
uv run --extra dev pytest -q -p no:cacheprovider --basetemp <fresh-temp-directory>
uv run python -m compileall -q src evaluation tests
```
