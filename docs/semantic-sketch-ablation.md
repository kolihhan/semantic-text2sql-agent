# Semantic Sketch Ablation

Goal: add one pre-SQL semantic analysis call that turns the question, evidence, and supplied schema context into an explicit intent sketch before SQL generation.

The sketch is JSON-only and contains: `outputs`, `filters`, `relations`, `aggregation`, `grain`, `group_by`, `ordering`, and `limit`. It must not emit SQL. SQL generation receives the original question/evidence/schema plus the sketch. Existing deterministic verification/repair remains unchanged.

The feature is opt-in while being evaluated.

Evaluation gate:
1. Pilot on a deterministic 20-case subset drawn from Qwen3.5 9B direct failures in the exact v5 frozen-100 benchmark.
2. Reuse the frozen source/case identities and the same `qwen3.5:9b-q4_K_M` model.
3. Compare the existing direct SQL with semantic-sketch→SQL on the same cases using official BIRD EX and execution success.
4. Proceed to all 100 only if pilot has positive net EX (`wrong→correct > correct→wrong`) and no execution-success regression.
5. Full-100 result, not the pilot, is the resume-quality result.
