# Qwen3.5-9B physical literal normalization — failed independent-ish transfer gate

**2026-11-22 · STATUS: REJECT automatic default promotion; retain diagnostic mechanism only.**

## Goal / frozen controls

Investigate executable-but-Strict-EX-wrong SQL by replacing **only same-column values whose original exact equality is absent and whose uniquely observed real SQLite value differs solely in casing/boundary whitespace**. This is a mechanical physical observation, never an LLM self-review, SQL intent proof or a semantic equivalence theorem.

- Baseline original Qwen3.5-9B Q4_K_M digest `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`.
- All proposal generation uses only frozen Qwen selected SQL, SQL schema and SQLite in read-only mode; publisher Gold and original EX labels arrive only in a **subsequent GitHub job after immutable candidate artifact upload**.
- Fixed `scripts/experimental_literal_normalization_replay.py` candidate algorithm transported **unchanged** into cross-DB TRAIN64 via `scripts/transfer_normalization_birdtrain64.py`. No second-pass rule tuning after looking at score.
- No new Qwen calls, no QLoRA, no default LangGraph changes, no HOLDOUT300, no sealed `simpson_episodes` or `works_cycles`, no credential spend.
- Historical selected SQL, not new matched Qwen inference; BIRD DEV and TRAIN64 original cohorts were previously exposed, so **not independent pristine performance evidence**.

## Executed GitHub Actions and immutable evidence

1. [Label-blind BIRD DEV physical risks](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44930911891) — SUCCESS; 75 originally complete-shard cases, 13 flagged originally executable cases, 11 originally Strict-EX-wrong, 2 originally right; **not** semantic error classification precision.
2. [Physical literal binding disambiguation](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44931503021) — SUCCESS; 11 original absent/case flags + 18 label-free hash-selected controls, 29 evaluated. Among originally complete shards, normalized same-column matched 3 historical EX-wrong cases, substring-only matched 3 historical EX-wrong cases; unsafe substring-only proposals withheld.
3. [DEV exposed counterfactual SQL replay](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44931856689) — SUCCESS; frozen 4 proposed SQLs among 11 risks, 7 abstained. Original complete shards: **3 proposals, 2 Strict-EX rescues (#287 toxicology, #513 card_games), 0 regressions**, 1 nonrescue (#28 california_schools), 2 rescues across 2 DBs. Historically invalid shard 3 separately: **1 proposed rescue (#93 financial)**; **do not combine** invalid shard with valid promotion counts. The reference historical 39/100 remains the published baseline; **do not claim a new paired 42/100**. Fully scored SET/BAG for modified SQL only, no model-generation rerun.
4. [Locked rule transported to four disjoint physical BIRD TRAIN64 DBs](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44932346821) — **all 3 jobs SUCCESS**: prepare strips original publisher Gold/EX; detect stage receives only original Qwen SQL/questions and SHA-checked publisher SQLite files; after candidate freeze, score stage accesses original labels. Four databases `movie_3`, `music_tracker`, `superstore`, `world` (physical DBs disjoint from BIRD DEV, historically exposed). TRAIN64 original Qwen Guarded SET EX **31/64**; 4 frozen modifications; **2 rescues (#1561 music_tracker, #7059 movie_3), 1 regression (#6864 movie_3), 1 nonrescue (#6855 movie_3)**; counterfactual SET EX **32/64**, net +1; bag **2 rescues, 1 regression**. This **fails** preregistered >=3 net rescues across >=3 DBs with no bag regression.
5. [Q4 vs Q8 model precision elimination](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44927795828) — both models Direct/Guarded 2/6; Q8 slower, no observed improvement.

## Mechanism-level failure

The helpful cases have independently observed nonmatching physical string casing/whitespace in the correct bound column. The harmful counterexample #6864 asked for customer `Lillie Kim`; original predicates `first_name='Lillie'` and `last_name='Kim'` matched the historical original publisher result, while replacing with physically stored `LILLIE`, `KIM` made original Strict EX **true→false**. This is a warning that physical value existence is **not sufficient evidence for benchmark answer correctness**, including empty-result/reference consistency issues. Do not create a post-hoc exception based on #6864 or consult Gold SQL to tune detector.

# Decision

**REJECT automatic literal normalization rewrite as a default path.** The mechanism demonstrably can rescue wrong SQL on several databases, but it also changes an original-right SQL and degrades BAG on independent physical databases. The result cannot be promoted as a better Text2SQL system. Source-diverse/historically exposed cross-db evaluation supports *diagnostic mechanism*, not causal aggregate improvement or a production-ready LangGraph node.

The only defensible next research problem: **generate verifiable evidence of *intended* predicate/answer semantics before changing SQL**, e.g. source-question-supported entity/attribute binding, schema FK/value provenance, and disambiguating empty-result/reference anomalies with human-adjudicated questions. Use original-right controls and SOURCE/DB-disjoint preregistration. If such evidence cannot be obtained at bounded small-model cost, prefer abstain/retain baseline. Never accept generic LLM self-review nor heuristic regex only.

**Scheduler continuation marker:** inspect this ledger and both run artifacts first; do not rerun the same four source-exposed cases with threshold tweaks. No verified EX increase over 39/100; keep automated research enabled.
