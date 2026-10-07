# BIRD paired DEV evaluation

This directory is the performance-evidence path. The bundled demo is a smoke test only.

The paired runner selects the manifest's frozen `dev_case_ids`, generates one Direct SQL candidate per case, and feeds that exact candidate to both Direct and Guarded. Gold SQL stays in a separate evaluation-label structure and is not consulted until both arm outputs for that case exist.

```bash
uv run python -m evaluation.run_bird paired-dev \
  --source /path/to/bird-mini-dev.json \
  --database-root /path/to/dev_databases \
  --manifest results/p1-split.json \
  --model qwen3.5:4b \
  --timeout 600 \
  --output runs/bird-paired-dev-v4/paired.json
```

The paired artifacts record protocol version, the exact treatment, model/config and evaluator provenance, per-arm execution/EX/call/latency data, repairs, and Guarded stage traces. Existing output paths are refused; a crash leaves `status: incomplete`. If either prediction or gold execution truncates at the configured row bound, the artifact is invalidated for strict execution-match comparison.

The project-local scorer follows set-of-tuples equality from the pinned BIRD evaluator contract; row order and duplicate multiplicity do not affect the score. The reference implementation is [BIRD evaluation.py](https://github.com/AlibabaResearch/DAMO-ConvAI/blob/483554eae102996f5ec1f4feab4e78ef29c2a394/bird/llm/src/evaluation.py). The upstream evaluator is a reference contract, not a claim that this repository imports it directly at runtime.

BIRD dataset/database files are external inputs and are intentionally not committed. No FINAL/holdout selector is exposed by the primary CLI.

## Evidence generations

The older paired DEV artifacts (`bird-paired-dev-v3`, `v4`, `v5`) document the historical 4B development line, including incomplete/backup operational artifacts. They should not be mixed with the later Qwen 9B frozen summary.

The current portfolio-facing 9B evidence artifact is [`runs/qwen9b-frozen100/summary.json`](../runs/qwen9b-frozen100/summary.json). Its provenance limitation is documented in the root README: the referenced GitHub Actions run had one failed treatment shard and a skipped aggregate job, so a clean end-to-end rerun is still outstanding.
