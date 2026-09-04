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

The v4 artifact records protocol version 3, the exact internal-reasoning/SQL-only repair prompt treatment, model/config and evaluator provenance, per-arm execution/EX/call/latency data, repairs, and Guarded stage traces. Existing output paths are refused; a crash leaves `status: incomplete`. If either prediction or gold execution truncates at the configured row bound, the artifact is `invalid` and official EX is false.

Official EX uses set-of-tuples equality, matching the BIRD evaluator contract; row order and duplicate multiplicity do not affect the score. The evaluator reference is [BIRD evaluation.py](https://github.com/AlibabaResearch/DAMO-ConvAI/blob/483554eae102996f5ec1f4feab4e78ef29c2a394/bird/llm/src/evaluation.py).

BIRD dataset/database files are external inputs and are intentionally not committed. No FINAL/holdout selector is exposed by the primary CLI.

The frozen valid v3 run is `runs/bird-paired-dev-v3/paired.json`; its result and **SIMPLIFY GUARDED** decision are documented in `docs/architecture/p1-dev-decision.md`. The adjacent 20-case `incomplete` timeout artifact is operational provenance only and is excluded from every metric. The new v4 target is separate and must not overwrite either artifact.
