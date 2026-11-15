# PRO research gates — Qwen-only semantic Text-to-SQL

[Canonical PRO method](https://github.com/kolihhan/kolihhan/blob/main/skills-search/pro-research-decision-gates.md): **P**roven failure → **R**eturn on experiment → **O**ut-of-sample proof. This file is the project-specific decision record, not a new model architecture.

## Real target and frozen evidence (verified)

**Target:** increase strict BIRD semantic execution-match beyond the historically reproduced **39/100** Qwen3.5-9B Guarded baseline; eventually demonstrate the mechanism on databases/source absent from development. **No SQL training. Qwen-only.** SQL execution success, prompt format, and time-to-valid-SQL are operational/proxy metrics and must never substitute for semantic EX.

| Evidence | Original → treatment | Decision |
| --- | --- | --- |
| Original 100 BIRD DEV questions, 11 physical DBs | Direct **33/100** → Guarded **39/100**; 6 rescues/0 regressions | Historical starting baseline; **not** a clean new model-generation workflow |
| [Original prediction replay](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43679115833) | **100/100 stored strict EX labels reproduced** without new model calls; Guarded: 39 correct, 52 executed-but-wrong, 9 verification failures; case 305 capped at 50k rows | Diagnostic evidence, not new uplift |
| Semantic revision on same frozen100 | Guarded 39 → revision **38**; 0 rescues/1 regression | **REJECT**, do not enable default |
| [Typed intent IR / compiler](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43667332961) | Guarded **16/20** → typed-IR **15/20**; 0 rescues/1 regression | **REJECT** on tested mechanism |
| [Stochastic 4-candidate consensus](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43675678201) | Guarded **16/20** → consensus **16/20**; hindsight oracle@4 **16/20** | **REJECT**: absent candidate headroom |
| [Cross-DB lexical 2-shot ICL](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43680102605) | Guarded **8/20** → 2-shot **7/20**; 1 rescue/2 regressions, bag EX tied 6/20 | **REJECT**; [PR #51 closed unmerged](https://github.com/kolihhan/semantic-text2sql-agent/pull/51) |
| [Native Qwen thinking](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43605551107) | Guarded **8/16** → native think **7/16**, materially higher latency | **REJECT** for this paired setting |
| [Physical SQLite question-value grounding, Spider DEV12](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43685240570) | Original Qwen Guarded **9/12** → grounded **9/12**; 0 rescues, 0 regressions, 6 physical DBs, bag EX tied **9/12** | **REJECT**, [PR #52 closed unmerged](https://github.com/kolihhan/semantic-text2sql-agent/pull/52); not BIRD EX |

**Scope warning:** Spider TRAIN, BIRD DEV16/20, the historical frozen100, and other public seen datasets are different populations and not interchangeable. These results **do not** show that all IR, reasoning or few-shot schemes fail; they reject the **specific tested variants**. Public split reuse/model pretraining contamination remains possible.

## Newly verified scoring/data-quality boundary (2026-11-15)

The source-disjoint *within BIRD DEV* physical-DB value-grounding screen **stopped before any inference**: historical frozen100 already covered **all 11 physical BIRD DEV databases**, so demanding 10 never-seen BIRD DEV databases was impossible. [Gate run 43684181898](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43684181898). Do **not** quietly relax this constraint; new physical DBs must come from another corpus and must not be described as a new BIRD DEV holdout.

An independent source-shift pilot on Spider DEV12 did generate actual Qwen predictions. The pre-registered evidence intervention did **not** increase semantic set-EX: **9/12 → 9/12**. [Full physical run 43685240570](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43685240570). [CPU-only physical original-reference audit 43688758256](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43688758256) found:
- **One true relational operator error:** `voter_1:700` asks for *area codes* shared by vote sets for two contestants (reference `INTERSECT`, 14 rows); Qwen imposed a false *same phone number* requirement, yielding 0.
- **One question family appearing twice:** `course_teach:387` / `:388` ask to exclude teachers from a hometown. Actual SQLite uses case-preserved `Little Lever Urban District` (1 matching record); Qwen's exact-case filter returns 6 rows. The provided reference SQL uses `little lever urban district` (lowercase), returning 7 under SQLite's ordinary case-sensitive equality. This is a **source reference/data inconsistency candidate**, not evidence of two independent Qwen semantic failures. Keep source strict EX **9/12** unchanged; annotate, do not silently rescore.
- This 12-case data set contains **at most two distinct failing database families**, and these observations are post-hoc. They cannot justify a special-case `INTERSECT` prompt patch or revised evaluation labels.

New evaluation reports should retain **source-SQL EX** and separately flag **reference-data consistency and semantic review**, especially literal collation/case, duplicated paraphrases, and incompatible result units. Use *fresh independent questions/physical databases* before testing a general relational-set mechanism.

## Next allowed experiment: prove generation-level headroom first

1. **P:** Take the original independently replayed 52 executable-but-wrong examples as a *post-hoc failure taxonomy only*: wrong result entity/projection, filter literal/value grounding, join ownership/cardinality, aggregation/grain, missing/extraneous rows, or scorer/cap artifact. A keyword mismatch alone does **not** prove semantic causality. Don't reuse their gold SQL for prompt construction.
2. **Check mechanism:** Before model calls, state what *new evidence/capability* the generator receives and why an error could be corrected without gold at inference time. Reject another reviewer, selector or extra sampling if the proposed candidate pool has no demonstrated reachable right answer.
3. **Freeze a new population:** select by question and physical database/source SHA **before reading labels or any baseline outcomes**. Avoid previous benchmark IDs and **database overlap where possible**; if the source is already exposed, label it exploratory.
4. **R:** Same pinned Qwen9B digest, original Guarded controls, seed, deterministic verifier and max 2 repairs; one intentional variable. Report strict set EX **and bag**, matched repairs/rescues/regressions, schema/DB breakdown, token/latency/call cost and capped/unscorable rows. Include no-change A/A if the source has generation instability.
5. **Screen (new experiments only):** declare the sample size and threshold before running. For a 20-task/10-DB *pilot*, demand **≥3 net EX gains, ≤1 regression, gains spread over ≥3 DBs and no bag decline**; otherwise reject without tuning on the 20 cases. This screen is **not** permission to merge.
6. **O:** Independently test a locked mechanism on a separate source/physical-DB family with unchanged code/parameters. If source independence or reproducibility is unavailable, mark **BLOCKED**, not “improved”.
7. **Promotion:** Only after real independent improvement, sensible latency/call ROI and consistent failure mechanism; smallest default-code change then full regression/real execution validation. Keep all negative trials closed/unmerged.

## Immediately actionable state

- Clean historical Qwen9B **offline SQL replay completed**: [PR #50](https://github.com/kolihhan/semantic-text2sql-agent/pull/50) (CPU-only; original inference was not a clean workflow).
- Detailed post-hoc structural comparison completed: [run #43679575503](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43679575503).
- Latest 2-shot lexical retrieval **failed** (8→7/20), [PR #51 closed](https://github.com/kolihhan/semantic-text2sql-agent/pull/51). New physical-value evidence also **failed** (9→9/12), [PR #52 closed](https://github.com/kolihhan/semantic-text2sql-agent/pull/52).
- **Do not start another prompt sweep now.** The next change must expose a previously absent *generation-level* semantic signal and a source-disjoint verification route; otherwise stop costly experiments. The reported current target remains **>39/100 strict EX**, not “90% SQL execution success.”

## Published schema definitions on KaggleDBQA

The LLM Prompt Recovery fifth-place team compared generated and public prompt
distributions, combined different candidate sources, and traced a failed local
metric gain to the production tokenizer. The CMI sensor winning team grouped
validation by subject and modeled sensor orientation and missing modalities.
These are team-reported experiments. Our application is to split by physical
database and test information that is absent from the baseline schema.
Sources: [Prompt Recovery](https://www.kaggle.com/competitions/llm-prompt-recovery/writeups/z-d-z-5th-place-solution)
and [CMI](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/cmi-1st-place-solution).

**P.** The historical BIRD replay contains executable-but-wrong queries that use
the expected operator family and choose a different result entity or column
(e.g. card IDs for a question about set names and IDs). This differs from the
confirmed Spider set-identity error and the case-sensitive reference anomaly.
An earlier declaration-only primary/foreign-key experiment delivered new schema
metadata on ten BIRD TRAIN DBs and scored **13/18 → 12/18** (0 rescues, 1
regression, 2 unscorable). [Run and raw results](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43626767080).
Keys describe joinability. The new treatment supplies publisher-authored field
meanings, such as units, roles and abbreviated labels that the DDL omits.

**R.** Freeze three official test questions per DB by hash ahead of inference.
Use the same visible tables, original Qwen3.5-9B digest, generation options,
Guarded repairs and SQLite execution in both arms. Add only the publisher's
column definitions for visible physical columns. Compare strict set and bag EX,
paired outcomes, resolved column use, model calls and elapsed time. Read gold
SQL after freezing predictions. Require at least 16/18 scorable, at least
three net set-EX gains, at most one regression, rescues in at least three DBs,
and nondecreasing bag EX to retain for independent confirmation.

**O.** KaggleDBQA is a separate public corpus. Exclude its baseball and soccer
databases conservatively because both topic families occur in prior evaluation.
Use six original publisher SQLite files and the pinned official test split.
Possible base model pretraining exposure is unknown. Any positive screen still
requires a separate physical-DB confirmation and a matched BIRD100 rerun.

**Result — REJECT column definitions as a standalone treatment.**
All 18 references were scorable. Strict set EX and bag EX both moved
**3/18 → 2/18** (one rescue, two regressions, rescue in one DB).
The models referenced all documented reference columns in **12/18 → 15/18**;
that proxy improved while returned tuples declined.
Both arms executed 18/18, with 19 model calls each. Summed generation and
execution time was 351.7s → 643.0s. The rescue retained exactly the same
gold-referenced columns: documentation helped the model choose the stored
value `OR` and a top-one query. For Colorado funding it found both correct
program columns but used an absent `CO` fact-table code. On another question
it found the correct location but leaked its ranking count into the returned
columns. The other regression changed single-fire ranking to sum-by-year.
[CPU run](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43699471878) ·
[Paired predictions](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43699471878/artifacts/14156200917).

**Post-hoc physical BIRD replay.** Among 51 uncapped executable errors,
19 had a result-width mismatch. Gold-aware deletion or reordering of
already-produced columns could match both the exact set and the bag for
**6/51** (four databases). The other 45 did not match under this oracle. In 15/51 the verified physical
base-column set was already identical to the reference, yet the result differed.
The oracle uses reference rows and changes no predictions; the original
**39/100** is the measured Guarded score. Some projection differences reflect
reference ambiguity, including a question explicitly asking for full names
whose SQL returns superhero aliases. Independently review ambiguous labels
before pursuing a question-only result contract. The original evidence and
read-only procedure are preserved in [PR #53](https://github.com/kolihhan/semantic-text2sql-agent/pull/53).

## One physical observation before Qwen SQL generation

The prior schema and metadata studies measured more column grounding without
better execution match. A physical check of a stored code, value representation,
or entity multiplicity supplies a different signal. The diagnostic comparison
uses one Qwen-generated read-only probe with three arms. Original Guarded uses
no probe. A matched inference control receives only the probe SQL. The
treatment gets the identical probe SQL and up to five actual rows with column
names. Invalid probes use the original generation path. All three arms keep the
same original Qwen digest, schema selection, repair limits and read-only checks.

Freeze four medium and four hard questions by identity per database from the
independently authored [Chinook and Sakila questions](https://github.com/VinayakMokashi/text2sql-analyst/tree/d5420f1ad73c6608c1b2869d7fba662ea4709920/eval).
Use original physical sample DBs and exact pinned source versions. The two
domains are fresh to this repository. The source is public and the sample
database result is conditional on these difficulty strata. Score predictions
against source SQL only after freezing them. Measure strict set and bag
execution match, paired flips, probe visibility, and total model calls and
latency. Screening requires at least 14 scorable, net gain at least 3 versus
Guarded with at most one regression, rescues on both DBs, net gain at least
2 versus the matched no-rows control, and no bag decline versus Guarded.
Any passing mechanism still needs a matched BIRD evaluation to measure the
target 39/100 baseline.

**Result — REJECT the physical-row intervention.** All 16 probes returned
nonempty observations. Strict set and bag EX were **5/16 Guarded, 5/16
probe SQL alone, 4/16 probe SQL plus physical rows**. The row arm rescued
zero and regressed one against either control. Its calls increased from
18 to 32 versus Guarded. In Chinook `t13`, all three arms averaged a line
item instead of aggregating invoice totals first. The probe reinforced the
wrong grain. In `t19` both probe arms returned the ranking total alongside
the requested year. In Sakila `s14` the SQL-only arm returned just the film
title, while the row arm also returned the unrequested ranking count.
This trial supplies physical evidence that a valid probe can preserve a
wrong grain or induce a projection error.

The first Actions attempt reached real generation before encountering the
source's list-valued equivalent SQL references. After supporting all official
alternatives, the four inference shards completed in [run
43703246942](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43703246942).
The uploader preserved all per-case model and probe SQL. The original
Sakila import triggers use the current time, so the two Sakila binary
fingerprints differed and workflow aggregation stopped. A read-only replay
on one physical file per domain exactly reproduced 16/16 original probe
payloads and 48/48 prediction execution statuses and scores, reusing the
original Qwen outputs. The original fingerprints, single-file verification,
and replay code are saved with [PR #54](https://github.com/kolihhan/semantic-text2sql-agent/pull/54).
BIRD100 was outside this sample.

A separate read-only counterfactual diagnosed a hidden filter error in
Chinook `t19`. The original fixture has zero invoices where customer
country differs from billing country, so the baseline's billing predicate
happened to match the reference year (2012). Changing only invoice 264's
billing country in an isolated copy changed the baseline to 2010, while
the customer-country reference still returned 2012. This is mechanism
evidence and leaves the official paired scores intact.

## Distinguish grain detection from SQL correction (2026-11-15)

The independent six-physical-Spider-DB [Qwen answer-grain planning experiment 43705751417](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43705751417) **finished** on 12/12 scorable questions: original Guarded **9/12 strict set and bag EX**, Qwen-generated answer entity/row grain/relation input **9/12**, with **zero rescues, zero regressions**, **13→25 model calls** and **215.2→496.57 seconds**. [PR #55](https://github.com/kolihhan/semantic-text2sql-agent/pull/55) **REJECTED unmerged**. A new `semantic constraints` name must not replay that rejected one-extra-Qwen-plan mechanism.

Separately, a post-hoc physical [Chinook t13 counterfactual run 43707817012](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43707817012) verifies an existing aggregation-grain failure at the exact original Chinook **v1.4.5** source blob. The *previously generated* Qwen9B query `AVG(InvoiceLine.Quantity)` computes **1.00** both before and after adding 100 purchased line items to one already existing invoice. The source reference `AVG(SUM(Quantity) per InvoiceId)`, correctly rounded, shifts **5.44→5.68** across unchanged **412** invoices. The treatment updates `Invoice.Total` for accounting consistency. **Decision: CONFIRMED_GRAIN_FAILURE_NOT_A_REPAIR**; [PR #56](https://github.com/kolihhan/semantic-text2sql-agent/pull/56) retains the small source-pinned diagnostic. This is a known, already-exposed question on the old Chinook DB, not out-of-sample EX, a Qwen repair, or a BIRD100 gain.

A prospective mechanism must first be detectable from **question + available schema/data only**, with a justified *answer-preserving or answer-changing* database mutation (not simply any perturbation). Test its ability to reject incorrect relational semantics on new physical databases before attempting correction. Only a frozen generator change that produces paired independent strict-EX rescues without regressions can justify altering the Guarded default; otherwise stop.


## Conservatively remove unrequested ranking metrics — genuine out-of-previous-DB result (2026-11-15)

[Free GitHub CPU Qwen run 43709584252](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43709584252) physically completed; all three source-disjoint Spider DEV shards and aggregate succeeded. Six physical databases never used in previous Spider experiments: concert_singer, cre_Doc_Template_Mgt, network_1, student_transcripts_tracking, world_1, wta_1. **18/18** independently frozen entity-superlative questions were scorable. All predictions made by the unchanged original Guarded Qwen3.5:9B model, pinned digest. The treatment only edits already generated SQL when a ranking aggregate is included in the SELECT results even though the question requests just the winning entity. It preserves SQL relationship, grouping and the aggregate in ORDER BY. The **same 18 model calls** suffice for both arms; source references are read after predictions freeze.

Measured strict set EX and bag EX: **Guarded 9/18 → conservative projection repair 10/18**, **one rescue, no regressions, one triggering case**. The rescued source-independent case was Spider DEV student_transcripts_tracking:537: ranking degree program by COUNT student enrolments was already correct, but the answer wrongly projected both degree name and count. Projection-only rewrite changed only returned columns and retained ORDER BY COUNT(...). This is a real paired one-task improvement on an untouched physical DB, NOT evidence of broad stable EX improvement.

**Decision: positive narrow mechanism signal, but REJECT promotion under preregistered screen.** The screen required >=2 net rescues across >=2 databases; observed 1 rescue on 1 database. Do not alter production Guarded, expand triggers based on this now-exposed cohort, or claim BIRD100 improvement. [PR #57](https://github.com/kolihhan/semantic-text2sql-agent/pull/57) retains reproducible exploratory code/tests. [Full paired artifact](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43709584252/artifacts/14159887010). For stronger evidence, run exact frozen treatment on a genuinely different original physical source, not a retuned Spider subset.


## Cross-source failure of the original frozen structural projection repair — BIRD100 (2026-11-15)

The original Spider18 Qwen9B conservative projection-only rule, exact source blob \`fa6ad187c2c0c331753f1cb1ddbd0da7840f23b1\`, obtained a single narrow Spider 9/18→10/18 rescue without regression on six previously untouched physical Spider DEV DBs, but **FAILED** the more important physical BIRD DEV100 replay. [Actions 43712110068](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43712110068) successfully repeated the *same deterministic rule* against all 100 historical frozen original Qwen9B Guarded SQL strings, without any new model calls or training, with identical pinned database/source hashes and model digest. Strict set EX **39/100→38/100**, strict bag EX **35/100→34/100**, **4 edits, zero rescues, one regression**; historical model EX labels matched exactly. The original shard 3 invalidity flag is preserved. [PR #58](https://github.com/kolihhan/semantic-text2sql-agent/pull/58) closed REJECTED unmerged. [Original per-case artifact](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43712110068/artifacts/14161271127).

The regression was BIRD case 1224, \`thrombosis_prediction\`: the question explicitly requested **highest total bilirubin level and patient ID/sex/birthday**; the existing correct SQL returned \`MAX(l."T-BIL")\` alongside patient attributes. The same frozen projection-only rule incorrectly deleted this explicitly requested numeric value. The alternative metric-only-first guard had no reliable question-to-output contract, thus raw syntax projection is not a safe semantic repair. Other triggers (1196, 84, 131) did not rescue. **Do not patch case-specific BIRD keywords, claim fresh generalization, or merge this operator as the default.** Primary BIRD100 still **39/100**. Future mechanism must distinguish result columns using user intent with source-independent evidence and protect original correct outputs before editing; new physical DB confirmation is required.


## Structural foreign-key source mechanisms: SOURCE HAS NO MISSING INFORMATION (2026-11-15)

A distinct candidate beyond prompt/IR/projection fixes was to expose real **multi-hop FK join routes** omitted by the one-hop original lexical schema selector. [GitHub Actions #43716748190](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43716748190) restored original SHA-pinned Spider physical SQLite source, excluded the 18 previously exposed Spider DEV database IDs and all TRAIN DB identities, and ran four successful focused FK-path unit tests. It found only **three eligible questions in one database** (\`real_estate_properties\`), rather than the preregistered 12 questions on six untouched physical DBs; **model inference never started**. This falsifies the proposed gate's feasible source coverage; do not manufacture extra questions by changing selection after seeing cohort counts.

An independent competing data-source hypothesis was that first-party published Spider \`tables.json\` foreign-key annotations contain *additional real relationships* missing in SQLite's declared constraints, and can enrich Qwen's schema without extra model planning. [GitHub Actions #43717344286](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/43717344286) passed four source-integrity tests and scanned **166 physical Spider DBs** against the publisher's original \`tables.json\`: **795** published FK annotation edges; **793** mapped to actual physical table/column pairs; SQLite already declared **800** FK edges, with **zero** extra publisher edges absent from the original physical FK catalog. Hence there are **0/12 eligible unexposed augmentation questions across 0 DBs**, no paired Qwen inference, no strict-EX change. The unmatched two annotations need not be assumed valid; no missing physical relationship was verified.

**Decision:** both concrete extra-FK-info implementations are rejected on *source feasibility before model budget*; this does **not** reject graph semantics generally, but a join-path hint cannot add facts when the publisher/SQLite graph has zero independent missing edges or when no sufficiently distributed cohort can be frozen. Need a new transferable relation/aggregation semantic mechanism that does more than restate available schema (e.g. source-disjoint relational invariants or changing candidate generation), and actual strict EX gain before modifying defaults. BIRD100 Qwen Guarded stays 39/100; SQL model remains untrained.
