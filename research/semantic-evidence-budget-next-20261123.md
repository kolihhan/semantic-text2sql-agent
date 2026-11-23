# Three-part BIRD research decision — semantic correctness, LangGraph, actual cost (2026-11-23)

**Goal:** Fixed original Qwen3.5-9B; raise strict BIRD SET/BAG EX and demonstrate actual benefit from stateful conditional LangGraph, under bounded measured compute. BIRD HOLDOUT300 and two protected TRAIN DBs remain untouched. No other models, Spider or SQL training. Core code stays on research branch.

## 1. Real semantic bottleneck: mostly missing good candidates, not more verifier rules

Original source-frozen Qwen9B DEV500 K8 retrospective [PR #122](https://github.com/kolihhan/semantic-text2sql-agent/pull/122): deployed 233/500 correct, oracle-any 285/500; 215/500 contain **no correct SQL among eight**, 52/500 had a correct candidate but chosen result was wrong. Thus 215/267 (80.5%) of K8 final failures are generator-frontier errors. Reranking alone cannot solve them.

Prior paired failures: Adaptive Graph TRAIN16 9/16→9/16 (+2 calls); Physical K4 second TRAIN16 9/16→9/16 (2.73x wall); ALL-of TRAIN11 2/11→2/11; COUNT-grain TRAIN19 11/19→11/19. Global-count post-hoc TRAIN19 11/19→12/19 after an exposed wrong case; truly question-frozen TRAIN32, 32 historical-exposed DBs, [Actions #45103219648](https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/45103219648) 15/32→15/32 SET/BAG, 0 changed SQLs, 0 rescues/regressions; +5.46s graph wall over 1151.59s shared model wall. Retire low-coverage lexical rewrite.

Outside work: [SQLens NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c57812dee8acade8c5e385260b2cde28-Abstract-Conference.html) uses clause-level semantic error signals; [TS-SQL EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.156/) develops test-driven refinement; [CHASE-SQL ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/974ff7b5bf08dbf9400b5d599a39c77f-Abstract-Conference.html) separates SQL generation diversity from candidate selection. They show prior art, not proof for this small-model project.

## 2. Meaningful LangGraph research: Evidence-Triggered Structural Alternative (ETSA)

Hypothesis: before asking original Qwen to regenerate, **type the requested entity, relation direction, filter/value, aggregation grain and answer shape**; independently ground the disputed SQL clause in physical schema FK and minimal database evidence. Only when there is a concrete mismatch that an alternative can address, route to exactly one *structurally different* SQL generation; otherwise accept/abstain. This targets missing correct SQL, rather than generic prompt-based self-review of an executable SQL.

Minimal workflow:
    Question + schema/evidence
         -> original Qwen K1 -> read-only verifier
         -> typed clause contract + DB evidence
             no witness -> execute original
             supported witness -> one targeted alternative SQL
                 -> SQL safety preflight -> gold-blind selector -> execute
                 -> on failure: original fallback, never loop indefinitely

LangGraph State keeps original SQL, typed obligations with source provenance, schema hash, exact witness, attempted alternatives, verified results, Qwen call/token/wall budget and reason for routing. Conditional edges skip unnecessary tools/calls; optional persistence/checkpoints permit resuming expensive work but do not improve EX themselves. Human-in-the-loop for production ambiguity only; **no human label input** in BIRD benchmarking. Official references: [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api), [Thinking in LangGraph](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph), [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence).

## 3. Matching experiment and cost contract

New source-only BIRD question-ID freeze before inference; exclude prior trial IDs, reserved HOLDOUT300/two TRAIN DBs. Most physical BIRD DB families already exposed, so never call question-disjoint a true DB-OOD test. Pre-gold freeze typed clause contracts / evidence, route decisions, SQL proposals and final selection. Then score publisher reference separately.

Three arms with same Qwen digest and original SQL:
- G0 original Guarded.
- G1 one *generic* additional Qwen SQL on the SAME routed cases with exactly matched call/context budget.
- G2 one evidence-conditioned *structural* additional Qwen SQL on those SAME routed cases, same call/context budget.
Compare Oracle@2 (generation frontier) AND deployed Strict SET/BAG EX (selector actually chooses it), rescues/regressions, original-right preservation, route activation and per-DB breakdown. A small feasibility gate can require >=2 net selected SET rescues, <=1 regression and gains on >=2 DBs; not statistical proof or automatic promotion. Stop zero activation or negative EX without prompt-sweeping exposed gold.

Track exact Qwen calls and actual Ollama prompt_eval_count / eval_count tokens, shared and incremental model wall, graph/tool wall, end-to-end p50/p95, cache hits. Exact paired McNemar uses rescue/regression discordances. Local Ollama is *not* zero-cost compute; GitHub Actions USD cannot be inferred from model token counts and must be explicitly observed or marked UNKNOWN.

**This research branch implements opt-in real Ollama token and nanosecond telemetry (same SQL outputs), plus a reusable same-cohort paired SET/BAG scorecard with exact McNemar, per-database slices and missing-cost-is-UNKNOWN semantics.** The CPU-only BIRD32 replay computes historical accuracy and wall, but properly reports missing historical tokens/dollars as null. No model calls or protected data used by this scorecard.

**Decision:** stop old narrow semantic-verifier treatments; spend inference only after new candidate-generation mechanism is falsifiable and source-frozen; differentiate it from equal-cost generic sampling and require both semantic EX gain and acceptable incremental costs.
