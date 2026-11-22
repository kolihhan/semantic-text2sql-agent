# BIRD + Qwen3.5-9B Hybrid Research Scheduler

## One research goal

Use LangGraph's real advantages: persistent State, conditional routing,
tool evidence, bounded recovery/recheck and appropriate stopping.
Prove higher strict BIRD SET/BAG execution correctness than the original
Guarded Qwen3.5-9B pipeline while controlling actual time and model calls.
Code complexity, green CI and SQL executability do not count as semantic gain.

Hard constraints: BIRD ONLY. Original Qwen3.5-9B Q4_K_M. Never Spider.
Do not use protected BIRD HOLDOUT300 or last two protected TRAIN databases,
and do not promote experimental workflow code to default silently.

## Hybrid schedule

Research / hypothesis executor: the already enabled ChatGPT Automation runs
hourly at minute 05 (Asia/Kuala_Lumpur). This is ChatGPT's fastest
recurrence. It must develop one falsifiable mechanism, freeze BIRD identities
ahead of inference, prepare the experiment and examine actual Strict EX +
rescues/regressions and costs.

Execution / orchestration: GitHub Actions runs every 5 minutes on main using
.github/workflows/bird-hybrid-five-minute.yml and the Python standard-library
scripts/bird_hybrid_scheduler.py, plus a workflow_dispatch option.

GitHub cron may start late. The short scheduled job never downloads Ollama,
runs a model, reads gold labels or creates a flood of issues/PRs. It dispatches
an exact experiment ONLY when a reviewed main-branch BIRD worker exists and
research/bird-hybrid-plan.json has an approved, frozen, budgeted experiment.
It checks the job path/model/source SHA/cost/Strict EX gate, enforces one
approved job at a time, blocks on other active BIRD Qwen research, and
deduplicates prior dispatches (including completed runs). GitHub API failures
fail closed.

At initial deployment, the next physical-evidence causal ablation is
DESIGN_REQUIRED. Hence scheduler output is NEEDS_RESEARCH; no Qwen inference
is approved or automatically started at deployment. The hourly research
executor must complete the preregistration rather than spending resources
on the previously rejected adaptive v1 heuristic.

## Future approved worker registration

1. Implement an actual bounded original-Qwen BIRD-only worker on main under
   .github/workflows/bird-research-approved-*.yml.
2. Worker must declare workflow_dispatch, marker
   BIRD_APPROVED_EXP_HANDLER_V1, run-name prefixed with BIRD-APPROVED:
   and inputs experiment_id, frozen_source_sha256, max_model_calls,
   max_wall_minutes. Model runtime must actually enforce these caps.
3. Source is frozen GOLD-BLIND, no tested reference SQL as model input.
   Research branch, fixes and proof must pass CI before worker is approved.
4. Change one research/bird-hybrid-plan.json experiment from design_required
   to approved with fixed SHA, bounded calls/wall and a prospective strict
   SET EX improvement/regression gate. Only then will a 5-min tick dispatch
   the worker once. No default agent-code merge is implied.
5. Scoring must be after sealed candidate SQL; only strict real BIRD
   publisher SET/BAG EX, same-cohort rescues/regressions and measured time,
   tokens/model calls/financial cost establish promotion evidence. Independent
   physical database / source evidence is required for transfer.

## Reference negative and positive findings

Actions #45048336642 (same exposed BIRD TRAIN16) verified:
Original Guarded 9/16 SET, Adaptive graph 9/16 SET, extra 2 Qwen calls and
+23.58 seconds post-first SQL model wall. REJECT original Adaptive risk gate.
Ordinary K4 8/16 vs Physical K4 10/16, but Physical uses 2.16x K4 model
wall time. This is an exploratory old-DB pilot, not independent correctness
proof or new BIRD DEV100.

The next research experiment must distinguish physical SQLite evidence from
longer/more directive prompts under matched actual original-Qwen calls and
budget, and demonstrate which *conditional* LangGraph step can actually
change answer correctness. If no mechanism survives, keep old Guarded.
