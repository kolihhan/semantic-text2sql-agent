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

**Scope warning:** Spider TRAIN, BIRD DEV16/20, the historical frozen100, and other public seen datasets are different populations and not interchangeable. These results **do not** show that all IR, reasoning or few-shot schemes fail; they reject the **specific tested variants**. Public split reuse/model pretraining contamination remains possible.

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
- Latest 2-shot lexical retrieval **failed the gate** (8→7/20), [PR #51 closed](https://github.com/kolihhan/semantic-text2sql-agent/pull/51).
- **Do not start another prompt sweep now.** The next change must expose a previously absent *generation-level* semantic signal and a source-disjoint verification route; otherwise stop costly experiments. The reported current target remains **>39/100 strict EX**, not “90% SQL execution success.”
