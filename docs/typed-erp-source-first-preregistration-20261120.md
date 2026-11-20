# Typed Entity–Relation–Population: source-first preregistration (2026-11-20)

**Status:** research-only, no production pipeline changes, no model generation or training.
**Base:** PR #125 head e38c521e8b9ad52fbf102facbf094736930f43e2.
**Model preserved:** Qwen3.5-9B q4_K_M, digest 56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90.

## Source and protection

The sole question corpus is original BIRD TRAIN JSON pinned to Hugging Face revision
cd2c4ea321075aa53ce7dfe52deaaffa505d046a and SHA-256
cd5884489dffd7c76d73509b396bd919fbe8b486efbc4ba2f48f89460de7d69f.
This publisher file physically contains gold SQL, but the source-only program
never indexes, prints, or serializes the SQL field. **This is a data-handling
limitation, not a claim the original TRAIN asset has no gold SQL.**

The 49/51 physical database exposure ledger is independently SHA-certified
from ten immutable historical question-source artifacts. The two still-unexposed
physical databases, simpson_episodes and works_cycles, are **not touched or
included**. The 425 prior source question indices and all 48 original K4 source
indices are excluded from the development cohort (set union, not sum).
Normalization additionally removes duplicate natural-language questions. The
12 sampled physical DBs are from the already-exposed 49, so this is **not**
an independent DB-generalization test.

Selection is SHA ordering, not based on correctness or SQL. Among exposed
databases with at least 20 remaining unique-source questions and a pinned
SQLite binary at or under the historical 96 MiB limit, select 12 databases
with SHA(seed,db), then 20 questions each with SHA(seed,original_source_index).
The cohort has 240 sampled questions and 12 physical database clusters.
Other source-only remaining TRAIN questions receive coarse question-only type
census; they do **not** receive physical schema validation. Report these two
denominators separately.

## Mechanism and abstention contracts

**Question-only candidate types** (priority fixed before reading model outputs):
1. Ratio/Percentage: explicit ratio, percentage, fraction, rate/proportion text.
2. Entity Count: explicit "how many", "number of", "count of".
3. Relation/Set: explicit both/either/neither/without/whose/belong etc.
4. Other: no recognized typed-mechanism request.
5. Unknown: insufficient natural-language content.

For physical SQLite schemas, use only PRAGMA table_info and foreign_key_list.
Entity count gets a **candidate** target only when the count noun phrase
maps uniquely to a physical table and the table has exactly one declared PK
column. Relation/Set gets a **candidate** typed FK only when exactly one
declared FK connects table names mentioned in the question. Ratio numerator,
denominator and population remain null/ABSTAIN until the source explicitly
proves their scopes; do not guess denominators or join directions.

The automatic source labels are **NOT expert ground truth**. Missing,
ambiguous, composite PK, multiple FKs, inferred FK and underspecified scope
are valid abstentions. No rule depends on question ID, old SQL or score.

## Strict stage barriers in Actions

A. **source-only**: SHA-check original training source and old source histories;
run independent unit tests; freeze 240 candidate records + 48 historical
mixed-controls, schema and code/commit digests. Upload immutable artifact
BEFORE any original Qwen candidate artifact is downloaded.

B. **SQL-only probe**: depends on A; restore the actual prior original
Qwen3.5-9B TRAIN48 *selected* SQL (all eight K4 shards), with exact source
and model digest checks. Probe narrow explicit unique target-PK vs counted
distinct child-PK. This is a **candidate contradiction**, not verified
semantic failure. Any unqualified, non-simple, grouped, ratio, nested or
ambiguous SQL must abstain. Upload independent artifact BEFORE downloading
old correctness outcomes.

C. **score-only**: depends on B; restore eight immutable original strict
physical SET EX score shards. Include all originally correct/wrong/censored
controls, report score-conditional false alarms, coverage, abstentions, and
database spread. Do NOT equate wrong EX with proven relational mismatch.
There is no publisher gold SQL inspection, no new model inference and no
SQL repair.

## Fixed decision rule before seeing B/C outcomes

A new Qwen candidate-generation experiment is **REJECTED** unless:
- A: ≥12 source-eligible *grounded candidates* in ≥4 physical DBs, with
  explicit case-level source/schema evidence and a defensible task type.
- B/C: ≥3 **independently source/SQ semantics-validated** wrong-unit/relation
  conflicts across ≥2 different DBs, not just wrong-EX flags.
- C: **zero** conflicts among all originally correct controls, with
  ≥20 correct controls and no post-hoc case removal.
- Validation: independent source-disjoint confirmation beyond reused TRAIN48
  is preregistered and respects the two protected DBs; an Observer approves
  design before any new model A/B.

Where manual/independent semantic review is missing, the number of **verified**
mismatches is 0 (unknown is not true-positive). All SQL anomalies are candidate
diagnostics. Even a convincing score correlation on reused TRAIN48 cannot
satisfy independent promotion. Do not weaken this rule after results.

No default Guarded Pipeline changes, no K4 rerun, no selector tweaking, no
DISTINCT rewrite, no DEV500 or HOLDOUT300, no QLoRA. Any result is expressly
not an improvement in BIRD strict execution match.

## Reproduction

Actions workflow: .github/workflows/cpu-typed-erp-source-first.yml

Local (provided the pinned TRAIN and historical source files are already
present):
- python -m pytest -q tests/test_typed_erp_source_first.py
- python -m scripts.typed_erp_source_first --help
- python -m scripts.typed_erp_historical_score --help

The Actions artifact, rather than this text, must be consulted for actual
sample counts, code SHA, question hashes, per-database status, and verdict.
