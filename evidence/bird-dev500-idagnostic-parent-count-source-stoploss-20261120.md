# BIRD DEV500 — ID-agnostic FK parent-COUNT trigger: completed source/physical/postblind audit (2026-11-20)

**Decision: REJECT AS GENERIC AUTO-REWRITE; preserve evidence and leave Guarded unchanged.**

## Verified physical run and frozen controls

[Actions #44654273040](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44654273040), all jobs successful; source-only artifact **14592222291** and postblind physical scorer artifact **14591638799**.

- Frozen source: previous public DEV500 Qwen3.5-9B K8 81-source-case physical-key snapshot from [#44555827205](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44555827205). Same fixed base model; **zero new Qwen calls**, no training, no verifier/Guarded changes, HOLDOUT300 reserved.
- Detector code: `scripts/dev500_parent_count_generic_source_gate.py`. It does **not** check question ID, publisher SQL, scorer outcome or a trained classifier: English “How many [entity]” + root `COUNT(*)` + physical one-column PK of named parent table + direct JOIN to a declared child FK matching parent PK. Conservative scope, no compound PK or grouped query. Proposes `COUNT(DISTINCT parent.pk)`.
- Runs physical read-only original SQL and candidate SQL **before opening publisher reference**; the second independent job verifies the source artifact and physically checks both SQL results against original pinned publisher SQL.
- **Not an independent prospective effect:** this detection pattern was formulated after reviewing previously failed DEV questions. The 81 source cases were all already known to have 8 wrong candidates.

## Actual physical measured result — all 81 selected known-wrong source SQLs

| Item | Count |
|---|---:|
| All K8-executable/wrong candidate source cases inspected | **81** |
| Structural rule triggered | **4 / 81** |
| Triggered SQL that actually changed result | **3 / 4** |
| Publisher-result scorable triggers | **4 / 4** |
| Original wrong → proposed right | **1 / 4** |
| Original wrong → still wrong | **3 / 4** |
| New model calls / training / Guarded modifications | **0** |

Per triggered question:

- **#109 (financial):** original wrong → still wrong. Physical `COUNT(*)` and `COUNT(DISTINCT client.pk)` returned the same result.
- **#234 (toxicology):** original wrong → still wrong. Parent FK rewrite altered the SQLite result, but not to publisher answer.
- **#593 (codebase_community):** original wrong → still wrong. `IN ('Supporter','Teacher')` does NOT enforce requiring both badges; DISTINCT alone does not fix the Boolean conjunction.
- **#1256 (thrombosis_prediction):** original wrong → right. Physical multiple lab rows per patient, `COUNT(DISTINCT Patient.ID)` matches reference. **This was previously identified**, so not a new discovery of a fresh case.

## Crucial correction: alleged 'correct controls' were not necessarily first SQL correct

Prior control source records #168, #1091, #1305 certified only the **K8 plurality selected result**, not that their **first original Qwen SQL** was right. The second postblind scorer physically confirmed:

| Control | First original SQL match | Generic trigger | Rewritten SQL match |
|---|---|---|---|
| #168 financial | **wrong** | no | not attempted |
| #1091 european_football_2 | **right** | no | not attempted |
| #1305 thrombosis_prediction | **wrong** | yes | still wrong |

**Genuinely correct original first-SQL controls = 1; triggered among these = 0.** It is mathematically invalid to claim a 0% regression rate on correct first-SQL triggers: **the denominator is zero**. Also selection of only three controls is weak even if some had triggered.

## Interpretation and protocol

1. A physical FK join + named parent is a plausible *multiplicity signal*, not a sufficient semantic-correctness condition. Only one correction matched the reference.
2. Three changed-result candidates do not equal three rescues; two remained semantically wrong.
3. Zero regressions in this cohort were **guaranteed by sampling only first-SQL wrong tasks**, not evidence of safety.
4. Real task efficiency is poor if the gate requires physical side-by-side SQLite execution for most new tasks; costs, tool-call budget and error budgets must be measured only after a safe trigger exists.
5. Dissimilar DB families do not establish a single cross-DB rescue rule. The earlier 3/3 selected manual rewrites were question-ID-targeted; their success is not evidence this automatic detector will work.
6. Current source-only gate does not address ambiguous natural-language units, Boolean conjunction, date scope, denominator semantics or outdated attribute snapshots.

**Stop-loss:** Do not merge this rule into production/default Guarded and do not start prospective Qwen or QLoRA training based on this outcome. If research resumes, first build independently human-adjudicated first-SQL-correct and wrong examples covering all 11 DBs, no oracle-at-inference; freeze one task-independent semantic proposal, then paired original-Qwen max2-repair BIRD EX with real rescues, regressions, uncertainty and runtime. HOLDOUT300 stays untouched.
