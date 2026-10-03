# DGSL v1 frozen-100 experiment

This branch tests one narrow hypothesis:

> Better pre-generation grounding can help the same XiYan 7B model avoid schema, value, and join mistakes.

It does **not** add a critic, semantic repair loop, candidate voting, fine-tuning, or a new generator model.

## Treatment

```text
question
  -> retrieval-only decomposition
  -> schema/description/value grounding
  -> FK shortest-path bridge expansion
  -> compact Grounding Pack
  -> same XiYanSQL 7B
  -> SQL
```

The decomposition only provides retrieval anchors:

- entity / relation
- metric / aggregation
- filters
- ranking
- time constraints

Gold SQL is never passed to decomposition, grounding, or generation.

## Frozen baseline

The baseline source of truth is GitHub Actions run **37090833784**, the SCV run whose 100-case aggregate is:

- XiYan baseline: **58/100**
- SCV: **33/100**
- wrong -> correct: **0**
- correct -> wrong: **25**

The DGSL experiment should reuse the `xiyan_sql` from those four shard artifacts via `--baseline-results`, rather than regenerating the baseline.

Model:

```text
hf.co/wanhin/XiYanSQL-QwenCoder-7B-2504-gguf:Q4_K_M
```

## Metrics

Gold SQL is used **after generation only** for offline diagnostics.

Reported per-case and macro metrics:

- table recall / precision
- column recall / precision over the schema context exposed to XiYan
- anchor-column recall / precision
- declared-FK bridge recall
- DB-value grounding recall for gold string literals that are observable in the bounded string-like value index
- frozen execution match (the project's set-equality scorer)

The schema metrics use a lightweight dependency-free SQL extractor. They are engineering diagnostics, not official BIRD schema-linking metrics.

## Success gate

For the frozen 100 cases:

```text
DGSL >= 61 correct
AND grounding metrics improve materially
```

If DGSL stays around 58-59, stop this Text-to-SQL optimization line instead of adding another architecture layer.

## Run one shard

First combine the four baseline SCV shard JSON files into one payload with a `cases` list, or pass an existing combined baseline result.

```bash
python experiments/xiyan_dgsl_frozen100.py \
  --source /path/to/dev.json \
  --database-root /path/to/dev_databases \
  --frozen runs/bird-paired-dev-v4/paired.json \
  --model hf.co/wanhin/XiYanSQL-QwenCoder-7B-2504-gguf:Q4_K_M \
  --baseline-results /path/to/xiyan-baseline-58-frozen100.json \
  --shard-index 0 \
  --num-shards 4 \
  --output dgsl-shard-0.json
```

Repeat for shards 1-3.

## Merge shards

```bash
python experiments/merge_dgsl_results.py \
  --inputs dgsl-shard-0.json dgsl-shard-1.json dgsl-shard-2.json dgsl-shard-3.json \
  --output dgsl-frozen100.json
```

The merged JSON contains the 61/100 gate, transition counts, and lexical-vs-DGSL grounding metrics.


## Reproducible GitHub Actions run

The branch includes `.github/workflows/xiyan-dgsl-frozen100.yml`.

It automatically:

1. downloads the pinned BIRD dev release,
2. pulls the same XiYanSQL 7B model,
3. downloads the exact baseline shard artifacts from Actions run `37090833784`,
4. runs four DGSL shards,
5. merges them,
6. asserts that the recovered baseline is exactly **58/100**, and
7. uploads `xiyan-dgsl-frozen100.json`.

The workflow is manual-only so ordinary pushes do not spend model-benchmark compute.
