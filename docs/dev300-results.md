# BIRD DEV300 — Paired GPU Evaluation

This record preserves the completed 300-question experiment separately from the currently shipped service.

- **Evidence:** [GitHub Actions run #45748311494](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/45748311494) (successful).
- **Raw outputs:** [artifact #15092182143](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/45748311494/artifacts/15092182143), including `ablation-dev300.json`, `runtime.json`, SQLite request cache and logs.
- **Runtime:** Kaggle two Tesla T4 GPUs, Ollama 0.40.16, `qwen3.5:9b-q4_K_M` (digest `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`), temperature 0, seed 42, thinking disabled, context 8192, maximum output 1024 tokens.
- **Cohort:** frozen BIRD DEV300 manifest, identical case IDs and reference SQL across all nine arms.
- **Scoring:** paired set-of-tuples execution match (called `set_ex` below). It is distinct from SQL execution success (`exec`). Gold was used only for scoring, not prompting.
- **Controls:** at most three logical model calls per question/arm. Exact-request cache avoided repeated physical generation; it does not reduce the logical per-arm budget.

| Arm | Set EX / 300 | Bag EX / 300 | Executable / 300 | Logical model calls |
|---|---:|---:|---:|---:|
| Direct | 148 | 136 | 278 | 300 |
| Guarded | 151 | 139 | 294 | 332 |
| Structured recovery reference | **161** | 149 | 274 | 392 |
| Projection guidance | 146 | 136 | 292 | 337 |
| Benchmark guidance | 142 | 129 | 294 | 325 |
| Identifier guidance | 153 | 142 | 296 | 326 |
| Empty-result recovery | 154 | 142 | 294 | 357 |
| Execution-error recovery | 151 | 139 | 294 | 332 |
| Combined guidance + recovery | 147 | 133 | 295 | 352 |

The 161/300 variant recovered 33 cases missed by Guarded but regressed on 23 cases Guarded got right. Exact paired McNemar testing does not establish a reliable win at conventional thresholds (two-sided p ≈ 0.23). It also required 392 rather than 332 logical model calls.

**Important:** this was a self-contained benchmark of distinct implementations. The main-branch service subsequently integrated the bounded structured-output and recovery concepts, while retaining its LangGraph graph, SQLite checks and public API. The 161/300 result must not be represented as the integrated main-branch score.

The reference implementation used for one ablation was a third-party MIT-licensed research codebase, pinned in the raw experiment workflow. The public service does not bundle that implementation.
