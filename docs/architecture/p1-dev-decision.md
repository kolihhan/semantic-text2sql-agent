# P1 paired DEV architecture decision — 2026-08-29

## Verdict

**SIMPLIFY GUARDED.** Keep the one authoritative LangGraph workflow and its deterministic, read-only, fail-closed boundary. Treat bounded model repair as a narrow recovery/escalation mechanism; do not claim that the full Guarded treatment is an unqualified winner or that preflight proves semantic correctness.

The benchmarked code, prompts, repair budget, verifier, model, manifest, and valid artifact are frozen. No post-result tuning or FINAL/holdout run occurred.

## Compared treatment

- **Shared generation:** question + physical schema + optional BIRD evidence → one SQL candidate.
- **Direct:** execute that shared candidate under the same SQLite and row limits, with no Guarded verification or repair.
- **Guarded:** start from the byte-identical candidate, deterministically verify, execute when accepted, otherwise make at most two repair calls and verify again before execute/refuse.

There is no Structured arm, SemanticPlan, grounding treatment, second initial generation, or alternate graph in the current paired protocol.

## Valid paired DEV result

Sample: 100 frozen DEV IDs. Model: `qwen3.5:4b`. Transport timeout: 600 seconds per model request. Maximum repairs: 2. Maximum rows: 50,000.

| Metric | Direct | Guarded | Delta |
|---|---:|---:|---:|
| Official BIRD EX | 30/100 (30%) | 33/100 (33%) | +3 pp |
| Execution success | 70/100 (70%) | 88/100 (88%) | +18 pp |
| Total model calls | 100 | 150 | +50 calls |
| Median latency | 3.236 s | 5.491 s | +69.7% |
| P95 latency | 79.398 s | 147.364 s | +85.6% |

Paired transitions:

- Direct wrong → Guarded correct: 3
- Direct correct → Guarded wrong: 0
- Direct execution fail → Guarded success: 18
- Direct execution success → Guarded fail: 0

Repair distribution was 69 cases with no repair, 12 with one repair, and 19 with two repairs. Of the 30 Direct execution failures, Guarded made 18 executable but only 3 correct under official EX. The 50 extra calls therefore bought three added correct cases, or 16.7 extra calls per added correct case.

## Integrity checks

Independent artifact validation confirmed:

- status `complete`, 100 cases, and 100 unique case IDs;
- case IDs exactly match the manifest's DEV order;
- byte-identical initial SQL for every pair;
- exactly one Direct model call per case;
- every Guarded call count equals `1 + repair_attempts`;
- stored rates and all four transition counts recompute from per-case records;
- runtime provenance records `qwen3.5:4b`, timeout 600, DEV-only protocol version 2;
- official evaluator revision `483554eae102996f5ec1f4feab4e78ef29c2a394`.

## Artifact provenance

- Valid complete artifact: `runs/bird-paired-dev-v3/paired.json`
- Valid artifact SHA-256: `309b20b08ba682e6cd9df5da16ddb2873022088dcd927632700c360fceea884a`
- Source SHA-256: `0310b7ca2dfb6247234e2073f93e7a75106ee41ba55231f75d06d2ef68efe2ec`
- Split manifest SHA-256: `b3b4ec070e63a7106d023917e0f0cc61ea75e2be2f2888b8c7bdaf2064fb3ae1`
- Official evaluator: `https://github.com/AlibabaResearch/DAMO-ConvAI/blob/483554eae102996f5ec1f4feab4e78ef29c2a394/bird/llm/src/evaluation.py`

An earlier attempt used a 180-second transport timeout and stopped at case 21. Its atomic checkpoint was `incomplete` with 20 cases and was excluded in full:

- Failed-attempt artifact: `runs/bird-paired-dev-v3/paired.failed-timeout-after-20.json`
- Failed-attempt SHA-256: `9eef4c58123cb330d30531968091881a03dc1d95aa4f1426e23035449533bee0`

The replacement run restarted from case 1 with the 600-second timeout; it did not reuse or select partial predictions from the failed attempt.

## Steel-man against the result

The Guarded treatment has a genuine favorable paired signal: three EX improvements, no observed EX regressions, and much higher executability. However, accepted Direct candidates pass through unchanged, while repair is invoked only after execution/preflight failure; zero correct-to-wrong transitions are therefore partly structural rather than evidence of semantic discernment.

Only three favorable EX-discordant pairs in 100 cases is weak evidence for general semantic improvement. Fifteen additional queries became executable but remained wrong, twelve failed to become executable, total calls rose 50%, and tail latency rose 85.6%. A valid-but-wrong query still passes deterministic checks. These costs support **SIMPLIFY GUARDED**, not **KEEP GUARDED**.

## Remaining uncertainty

This is one unseeded local-model DEV run on one machine. No confidence claim, cross-model claim, production-SLA claim, or leaderboard claim is made. The sealed FINAL/holdout split was not inspected or run.
