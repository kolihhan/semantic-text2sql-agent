# Original Qwen3.5-9B TRAIN96 — posthoc candidate-coverage / regression mechanism audit

## Provenance (same frozen physical 96)
Source: [run #44476207751](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44476207751). Read **12 original model-generated SQL shards + 12 original physical score shards**, all by immutable source index. This is a retrospective diagnostic, **not new evaluation/gain, not independent holdout**, and not causal proof from changes in SQL text. 96 total, 88 publisher-reference-scorable, 8 unscorable in `retail_world` due to publisher SQL `OperationalError`.

### Four-candidate exact-coverage decomposition
| Original fixed Qwen35-9B candidate source | Oracle SET EX on same 88 |
|---|---:|
| Two independent direct original SQLs (seeds 42/43) | **42/88** |
| Two independent role/grain prompt SQLs (same seeds) | **42/88** |
| Correct under **both** K2 pools | 34/88 |
| Correct **only** under direct K2 | 8/88 |
| Correct **only** under semantic K2 | 8/88 |
| Union of **all four** independent calls | **50/88** |

*Crucial:* 50/88 is a gold-knowing hindsight **oracle@4**, **not actual deployable performance**, and uses twice as many model calls as each K2 arm. The real first-candidate selected scores are **35/88 direct vs 29/88 semantic**, 7 rescues and 13 regressions, net -6.

### What changed in 13 regressions
- In **6/13**, treatment's *first selected* SQL was physically **not executable** (compile/table/column/syntax problems). Remaining **7/13** executed but did not match reference results. Therefore simply making SQL executable can fix only one subset and cannot prove the result semantically correct.
- Examples: source #128 removed `Mountain` and `Billing disputes` scope; #129 replaced district FK join with a nonexistent/direct `state` filter; #4450 discarded the `Illinois` geographic constraint; #2563 built a different table path that is not executable; #6095 projected a country name in place of the requested organization name; #5114 dropped the actual team membership join.
- Some genuine rescues: #5037 removed an extra selected name column (height requested only); #5103 selected just the player's name rather than including the assist numeric value; #2491 repaired a game-to-platform table path; #1844 reversed the role of GeneID1/GeneID2. These examples are **posthoc descriptions of pairs**, not a verified exhaustive root-cause taxonomy.

### Cross-check with other independently run historical mechanisms
- Previous same-base grounded-plan TRAIN96 [#44316964908](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44316964908): 36/95 direct vs 39/95 plan (+3), 7 rescues, 4 regressions, DB CI lower below 0, different call budget; **not promoted**.
- Previous equal-2-call physical relation-preview TRAIN96 [#44333082630](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44333082630): 51/91 direct vs 48/91 probe, 5 rescues and 8 regressions, more wall time, **not promoted**.
- Cross-DB template-binding feasibility4 [#44450472093](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44450472093): template bound **1/4**, template arm 0/4 vs direct 0/4, position control 1/4; **far too small and non-independent** to justify scale-up.

## Structural next step — no automatic model-prompt rerun
The main obstruction is **creating the correct SQL structure reliably**, not finding one through repeated generic prompts. Repeated “role/grain” instructions generated as many additional uniquely correct outcomes as uniquely lost outcomes. A new candidate proposal must:
1. Start with *actual* grounded table/column/FK paths and entity-grain cardinality, not a guessed join.
2. Preserve every source question and evidence filter/literal in generated SQL, perform read-only compile preflight, and abstain rather than introduce an invalid join or remove a geographic/team constraint.
3. Demonstrate **increased candidate-set strict-execution oracle under matched model calls** *before* testing selection. This oracle is diagnostic only; a deployment claim also needs a gold-blind selector and separate fresh question-level score, latency and DB-cluster analysis.
4. Not reuse any of these 96 source IDs to tune and claim prospective performance; use new frozen source IDs, original model digest and original database checksums.

**Decision:** reject blind role/grain prompt variant, reject any statement that oracle@4 50/88 is a real improvement. Do not modify the production SQL architecture or touch reserved BIRD HOLDOUT300.
