# Experimental XiYan-first SQL routing

**Not the default service.** This is an explicit, read-only candidate built on the existing `run_guarded` path, with **no extra semantic judge**.

## Why

The successful frozen BIRD DEV100 2-arm experiment yielded Qwen9B Guarded 41/100 set-EX, XiYanSQL 7B with its native prompt 50/100, with 11 XiYan rescues and 2 regressions. A third gold-blind policy returned XiYan whenever it passed preflight + nontruncated execution, otherwise Qwen: **51/100 in offline replay**. The separate 40-case, 20-database BIRD TRAIN transfer cohort yielded Qwen 16/40, XiYan 27/40, with no extra rescue by the fallback.

That is evidence for switching the *generator*, **not** evidence that a deterministic SQL verifier knows the right semantics. A valid but wrong XiYan query will be returned unchanged. Never present 51/100 as an online or hidden-test score.

## Run the actual two models

```bash
ollama pull qwen3.5:9b-q4_K_M
ollama pull hf.co/wanhin/XiYanSQL-QwenCoder-7B-2504-gguf:Q4_K_M
```

```python
from semantic_sql.xiyan_first import XiYanFirstSQLService
from semantic_sql.providers import OllamaProvider

settings = dict(timeout_s=900, temperature=0, seed=42, num_predict=1024)
service = XiYanFirstSQLService(
    "demo/demo.db",
    xiyan_provider=OllamaProvider(
        model="hf.co/wanhin/XiYanSQL-QwenCoder-7B-2504-gguf:Q4_K_M",
        **settings,
    ),
    fallback_provider=OllamaProvider(
        model="qwen3.5:9b-q4_K_M",
        **settings,
    ),
    max_rows=100, fallback_repairs=2,
)
result = service.ask("How many orders are there?")
print(result.status, result.candidate.sql, result.rows)
print(result.stages[0])  # xiyan_primary_valid or qwen_fallback:...
```

For large schemas, pass a schema string obtained independently from the database through `service.ask(question, schema_context=...)`. The reported BIRD experiment used `evaluation.schema_context.select_schema_context`, not the full-catalog schema used by the small demo; this matters for comparable model behavior.

## Route and limitations

1. Send the unchanged native XiYan prompt to XiYan provider.
2. Run the resulting candidate through existing deterministic SQLite preflight and read-only bounded execution. There is no repair in the primary arm.
3. If the primary query passes and returns a nontruncated result, use it even if it answers the wrong question.
4. If preflight/execution fails, rows truncate or the model raises, generate a new guarded query with Qwen, allowing at most two repairs.
5. Route is recorded in `result.stages[0]`; the default `SemanticSQLService`, CLI and API are not modified.

These outcomes measure distinct things: SQL preflight (safety/validity), nontruncated execution (operational reliability), and set execution match against gold (correctness). A successful SQL run is not semantic correctness. A two-model fallback has unmeasured end-to-end latency overhead in the BIRD cohort and should not be made the default without further validation.

Source: [BIRD DEV100 matched run](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43550578654), [gold-blind retrospective audit](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43556258516), [live CPU demo smoke](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43566979426).
