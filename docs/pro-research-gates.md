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
