# BIRD DEV500 81 fully executable wrong SQL — new 3-DB entity-grain probe (2026-11-20)

**Status: verified diagnostic, NOT new Guarded EX. Default unchanged, no SQL training.**

## Frozen scope and provenance
- Original Qwen3.5-9B digest `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`.
- [Original 81 source-first physical schemas and SQL, no reference](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44555827205), run artifact `14546155987`. Population: BIRD DEV500 K8 zero-oracle **215**, all-eight-executable/wrong **97**, 16 earlier-pilot exclusions → **81 in 11 DBs**.
- [New single run with separate source-only and postblind jobs](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44569182407). Source-first evidence artifact `14553596411`; postblind physical-set scorer artifact `14553771540`. The source job receives neither the publisher reference SQL nor any trained labels; the postblind job checks the immutable source-first SQL hashes before comparing actual SQLite result-set hashes.
- Source and scorer: [`scripts/dev500_3db_question_first_entity_grain.py`](https://github.com/kolihhan/semantic-text2sql-agent/blob/research/dev500-81-count-grain-stoploss-20261119/scripts/dev500_3db_question_first_entity_grain.py).
- The source cases were **deliberately chosen among already known K8 failures**. They are not a random prospective eval cohort, not independent source databases relative to other DEV work, and not evidence that a deployable Guarded would improve.

## Frozen question-first hypotheses and verified postblind result

| ID | Physical DB | Requested unit vs first original Qwen SQL | Source-first proposed semantic correction | Original→proposed publisher-set match |
|---|---|---|---|---|
| 219 | toxicology | carcinogenic **molecules** among triple-bond molecules vs original fraction of **bonds** among carcinogenic records | distinct molecule numerator and denominator, filter triple-bond existence first | wrong→right |
| 571 | codebase_community | user 24's **posts** relative to the user's own **votes** vs original join of the user's posts to votes *on their posts* | count each relation independently, divide with NULLIF zero handling | wrong→right |
| 1080 | european_football_2 | distinct **players** vs repeated `Player_Attributes` snapshots | count distinct documented player identifier | wrong→right |

Each of the three original and revised SQLs physically executed and produced different results before publisher SQL was opened. Postblind scorer: **0/3 original correct, 3/3 source-first changes correct, 3 rescues, 0 regressions, 3 of 3 scorable**, no additional Qwen calls. These are **three different SQL rewrites for three different semantics**, not one validated automatic transform.

Previous eight selectively checked count/grain candidates ([prior stop-loss](./bird-dev500-81-count-grain-crossdb-stoploss-20261119.md)): **2/8 rescues, 6/8 still wrong** across several DBs. Combined *nonrandom selected diagnostic population*, 11 case rows: original 0/11→5/11, 5 rescues/0 regressions in this selected set. **Never report 5/11 as true overall semantic improvement or promoted EX**.

## Contradictory controls and decision

- Earlier correctly answered controls #168, #1091 and #1305 demonstrate `COUNT(*)` on a JOIN can already have correct semantics. Incorrect #109, #1028, #1289, #1441, #463 and #593 did not become reference-correct from simple DISTINCT substitutions.
- [Previous genuinely independent BIRD TRAIN16 conservative inner-join COUNT parent-PK guard](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44053157511) activated **0/16**, strict EX **7/16→7/16**. It cannot be cited as a positive generalization result. The current three repairs do not resolve its zero-coverage problem.
- **Do not add a broad production count/distinct or SQL rewriting policy**. Source-first interventions for `bond→molecule`, `posts/votes` and `snapshot→player` have distinct preconditions; some are more than one AST edit.
- Next admissible gate: pre-register a *single* question/schema/SQL trigger using only source-visible features and physical PK/FK, establish nonzero coverage and false-positive behavior on independent correct controls, then run the existing original Qwen+max2-repair paired BIRD SET EX with database-cluster uncertainty and cost; HOLDOUT300 stays sealed.
- QLoRA for Text2SQL forbidden by current owner scope; original Qwen3.5-9B fixed.

**Metrics boundary:** zero new Qwen model calls, zero trained adapters, zero production Guarded changes, **zero new whole-system BIRD EX**, GAIA not applicable. These three cases are a *causal SQL intervention on selected source inputs*, not an end-to-end generalization finding.
