# BIRD TRAIN physical-database exposure and protected OOD set — 2026-11-18

## Verified source facts

SHA-256-checked historical original source rows from ten immutable GitHub Actions source cohorts. Independent GitHub Actions source certification:
https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44296448327
Artifact 14424600461; full source-only manifest SHA-256:
c9f1098a81541a393dfad19bc4b4600e6c421902149375c830c7673f600c632d.

- Pinned BIRD TRAIN contains **51** physical SQLite database families.
- Historical original frozen cohort union covers **49** physical databases, not merely 49 question IDs. Cohorts: TRAIN242, VAL64, FK32, two value16 runs, role8, preview8, column8, grain16 and latest fanout16. The last fanout16 repeated three already used DBs; only movie was a new physical DB.
- The **only two unexposed physical databases** are simpson_episodes and works_cycles. simpson_episodes has 209 original training questions and a 753,664-byte SQLite file; works_cycles has 474 questions and a 103,100,416-byte SQLite file.
- Existing 96 MiB physical-source cap is 100,663,296 bytes: works_cycles exceeds it. Do not silently relax the limit. If evaluating works_cycles, preregister the resource-only change BEFORE any new model inference/results and preserve the pinned DB object SHA.
- Exactly 32 original question IDs per remaining database were frozen by SHA-sort without reading predictions or gold SQL. The permanent IDs, cohort SHA and provenance are in manifests/bird-train-last-two-db-protected-holdout-20261118.json. The complete action artifact stores only original indices and question/evidence SHA, not raw question text, reference SQL, model output, or answers.

## Research and evaluation decision

The Base Model remains the same Qwen3.5-9B. Exact original Ollama digest:
56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90.
SQL QLoRA stays paused. No stronger models/teachers. BIRD DEV500 and
reserved HOLDOUT300 were not accessed or scored.

Do not use the final two physical DBs as a development set, or adjust any
prompt, hypothesis, error pattern, or heuristic threshold using outcomes
from them. A new mechanism must be preregistered and developed on the
49 already-exposed DBs or a distinct corpus (clearly labelled as a
source shift). Only after prospective mechanism evaluation should this
two-DB protected set be used for a one-time last-stage confirmation.

Two independent DB clusters do NOT support a strong cluster-level confidence
interval on an effect as small as five percentage points, even if 64 source
questions were frozen. Avoid treating them as 64 independent databases.
Public-model pretraining contamination remains unknown.

## Established stop-loss

- Real value-grounding independent replica: original favorable 8→11/16,
  disjoint physical DB replica adverse 9→5/16.
- Equal-budget real Qwen3.5-9B physical FK fanout generation: alternative
  second candidate 8→7/16; pass@2 13→13/16, selected EX 12→12/16.
- Exact physical SQLite filter-literal substitution: 3 valid guarded edits,
  0 EX rescues (31→31/64).
- Candidate count/mean, table visibility, explicit evidence and tie-break
  lexical screens did not achieve their predeclared error-selection gates.

## Next testable semantic lever

A future relation-aware experiment must propose a **question-grounded causal
invariant**: examples include entity-level count invariance under duplicate
child rows, scope of filters, or relational-set membership under a controlled
SQLite mutation. The expected answer behavior must follow from the natural
language question and known physical schema, NOT from reading publisher gold.
Validate trigger coverage and false rejections on training controls first,
then new same-model generation/repair only when the mechanism has sufficient
source coverage. Report actual frozen paired strict set/bag EX, independent
DB spread, both rescues and regressions, generation/SQLite cost, and a
deployable gold-blind selection mechanism. Never promote oracle pass@K.

**This completed stage improves benchmark provenance and prevents evaluation
leakage; it demonstrates NO new semantic EX improvement.**
