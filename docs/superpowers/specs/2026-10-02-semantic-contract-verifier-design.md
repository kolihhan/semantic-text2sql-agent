# Semantic Contract Verifier (SCV) — Design

**Date:** 2026-10-02  
**Status:** Proposed  
**Repository:** `kolihhan/semantic-text2sql-agent`

## 1. Purpose

The project already has deterministic SQL safety/preflight checks and bounded repair, but those checks answer a limited question: *is this SQL safe and compilable?* They do not reliably answer: *does this SQL express the user's intended semantics?*

The previous clause-guided semantic revision (CGSR) experiment confirmed that a second LLM self-review pass is not enough. On the frozen 100-query XiYan ablation, the semantic-revision arm produced no net EX gain and introduced a regression. The next experiment must therefore add an **independent structural signal**, not another generic model critique.

SCV is a bounded semantic-verification subsystem:

> **question + schema + evidence → typed semantic contract → SQL AST / FK graph → deterministic violations → at most one targeted repair**

The goal is not to solve all Text-to-SQL semantics or chase BIRD SOTA. The goal is to test whether a small, inspectable verification layer can recover a few executable-but-wrong queries while keeping false-positive repairs low.

## 2. Success criteria

The frozen-100 evaluation is the decision gate.

Primary success criteria:

- SCV paired execution accuracy under the frozen evaluator must exceed the paired XiYan baseline.
- `wrong→correct > correct→wrong`.
- `correct→wrong <= 2/57`.
- Preferred practical bar: at least **+3 net correct cases** on the frozen 100.

Secondary diagnostics:

- violation precision on triggered cases;
- coverage by violation type;
- repair success by violation type;
- SCV skip / abstain rate;
- latency overhead;
- number of untouched correct queries.

If the result is approximately `57→57` or `57→58`, or if false-positive intervention is high, SCV is frozen. There is no SCV-v2 in this scope.

## 3. Non-goals

SCV will not add:

- candidate ensembles or self-consistency voting;
- synthetic execution databases;
- learned verifier models;
- training or fine-tuning;
- multi-agent orchestration;
- general natural-language semantic parsing;
- full value grounding;
- unrestricted iterative repair;
- leaderboard/SOTA optimization.

## 4. Current system boundary

Existing components remain responsible for their current jobs:

- `evaluation/schema_context.py`
  - deterministic lexical schema selection;
  - one-hop FK-neighbor inclusion;
  - column descriptions.
- `semantic_sql/verifier.py`
  - single-statement/read-only enforcement;
  - SQLite compilation via `EXPLAIN QUERY PLAN`.
- `semantic_sql/inference.py`
  - generation;
  - deterministic preflight;
  - bounded repair;
  - execution.

SCV must not replace these responsibilities. It is inserted only after a candidate has passed syntactic/safety preflight.

## 5. Architecture

New modules:

```text
src/semantic_sql/
├── semantic_contract.py
├── sql_semantics.py
├── schema_graph.py
└── contract_verifier.py
```

Data flow:

```text
Question + Evidence + Selected Schema
                  │
                  ▼
        semantic_contract.py
                  │
            SemanticContract
                  │
                  │
XiYan candidate ──┼─────────────────┐
                  │                 │
                  ▼                 ▼
          existing preflight   sql_semantics.py
                                  SQL → AST
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
               contract_verifier.py         schema_graph.py
                 contract ↔ AST             JOIN ↔ FK graph
                         └────────────┬────────────┘
                                      ▼
                              typed violations
                               │            │
                             none         found
                               │            │
                            execute   targeted repair ×1
```

The contract extractor never sees candidate SQL. This keeps it independent from the generator and avoids “the model agrees with its own answer” as the only signal.

## 6. Semantic contract

The first version intentionally represents only semantics that can be checked with useful precision.

Conceptual schema:

```json
{
  "confidence": "high | medium | low",
  "projection": ["entity_or_field"],
  "aggregation": {
    "function": "COUNT | SUM | AVG | MIN | MAX | null",
    "target": "entity_or_field | null",
    "entity_grain": "entity | null",
    "distinct": "true | false | null"
  },
  "group_by": ["entity_or_field"],
  "ranking": {
    "metric": "entity_or_expression | null",
    "direction": "ASC | DESC | null",
    "top_k": "integer | null",
    "ties": "single | all | unspecified"
  },
  "filters": [
    {
      "field": "entity_or_field | unresolved",
      "op": "= | != | < | <= | > | >= | BETWEEN | IN | LIKE | IS NULL | IS NOT NULL",
      "value": "scalar | list | range | unresolved"
    }
  ],
  "join_entities": ["entity_or_table"]
}
```

Rules:

1. Contract extraction input is only:
   - user question;
   - BIRD evidence/hint if present;
   - selected physical schema context.
2. Candidate SQL is never included.
3. Malformed output fails open.
4. Low-confidence or unresolved fields do not produce hard violations.
5. The contract is not chain-of-thought and must not contain hidden reasoning text.

## 7. SQL semantic representation

Add `sqlglot` as the single new parser dependency.

`sql_semantics.py` parses a candidate into a normalized structural representation containing at least:

- projected expressions / columns;
- aggregate functions and targets;
- DISTINCT use;
- GROUP BY expressions;
- ORDER BY expressions and direction;
- LIMIT;
- explicit predicates and literals;
- table aliases;
- join edges between resolved table/column references.

The normalized representation must be deterministic and testable without a model.

If AST parsing fails, SCV abstains and leaves the existing preflight/repair path in control.

## 8. Schema graph

`schema_graph.py` builds a graph from the existing catalog foreign keys.

Each FK contributes a directed canonical edge plus an undirected adjacency used for path reasoning:

```text
source_table.source_col -> target_table.target_col
```

The first version checks only explicit join predicates.

A join is structurally supported when one of the following holds:

- the exact FK edge appears;
- the reverse form of the same FK equality appears.

SCV does not infer arbitrary semantic joins from names, values, or embeddings in v1.

This intentionally catches clear failures such as joining unrelated identifiers while avoiding speculative graph repair.

## 9. Violation model

SCV returns typed, evidence-bearing violations rather than a boolean “wrong SQL” judgment.

Initial types:

- `PROJECTION_MISMATCH`
- `AGGREGATION_MISSING`
- `AGGREGATION_FUNCTION_MISMATCH`
- `ENTITY_GRAIN_MISMATCH`
- `DISTINCT_REQUIRED`
- `GROUP_BY_MISSING`
- `GROUP_BY_MISMATCH`
- `ORDER_METRIC_MISMATCH`
- `ORDER_DIRECTION_MISMATCH`
- `TOPK_MISMATCH`
- `JOIN_EDGE_INVALID`
- `FILTER_LITERAL_MISSING`

Each violation includes:

```text
code
expected
actual
evidence
confidence
```

Example:

```text
AGGREGATION_MISSING
expected: SUM(results.points) at constructor grain
actual:   row-level results.points
evidence: contract ranks constructors by total points
```

## 10. Verification policy

SCV checks only high-confidence contract fields.

### Projection

Trigger only when the requested output can be resolved to a specific entity/field and the candidate projects a clearly different resolved field.

Do not trigger for ambiguous aliases, computed expressions, or unresolved wording.

### Aggregation and entity grain

Check:

- required aggregate function exists;
- target is compatible;
- requested entity grain is represented by GROUP BY or DISTINCT semantics;
- count-at-entity-grain cases use DISTINCT when duplicate lower-grain rows are possible and the contract explicitly requires entity counting.

The verifier should prefer abstention over guessing cardinality from names alone.

### Grouping

Trigger when the contract requires grouped aggregation and the SQL either has no GROUP BY or groups at a clearly incompatible resolved entity.

### Ranking / top-k

Check:

- ORDER BY direction;
- LIMIT/top-k;
- ordering metric matches the required aggregate/raw metric;
- tie semantics only when the contract explicitly identifies “all ties”.

### Join path

Check explicit equality join predicates against the FK graph. Do not attempt general join synthesis in the verifier.

### Filters

v1 only checks high-confidence explicit literal omissions. It does not judge subtle predicate semantics, normalization, or fuzzy value grounding.

## 11. Targeted repair

Repair is triggered only when one or more hard SCV violations exist.

The repair prompt receives:

- question;
- evidence;
- selected schema;
- current SQL;
- typed violations with expected/actual/evidence.

Instruction:

> Apply the smallest SQL change required to satisfy the listed violations. Do not rewrite unrelated clauses.

Budget: **one semantic repair**.

After repair:

1. existing deterministic preflight runs again;
2. SCV runs again;
3. if hard violations remain, the system does not enter another semantic repair loop.

For the experiment, both original and repaired candidates are preserved so transitions are auditable.

## 12. Runtime integration

Proposed flow:

```text
generate candidate
→ deterministic preflight
→ if preflight fails: existing repair behavior
→ if preflight passes and SCV enabled:
     extract/load independent contract
     parse SQL AST
     verify contract + join graph
     if no hard violation: execute
     if hard violation:
         targeted semantic repair once
         preflight repaired SQL
         re-run SCV
         execute if valid, otherwise stop/fail closed for semantic treatment
→ execute
```

SCV is feature-gated and off by default until the frozen-100 ablation is complete.

The historical baseline path must remain reproducible.

## 13. Failure handling

SCV is fail-open for unavailable semantic evidence, but not silent.

Outcomes:

- `SCV_SKIPPED_CONTRACT`
  - malformed contract;
  - low overall confidence;
  - extractor failure.
- `SCV_SKIPPED_AST`
  - SQLGlot cannot parse the candidate.
- `SCV_PASS`
  - no hard violations.
- `SCV_VIOLATION`
  - one or more typed violations.
- `SCV_REPAIR_PASS`
  - one repair removes hard violations and preflight passes.
- `SCV_REPAIR_FAILED`
  - repair still violates preflight or SCV.

Skip states do not modify the candidate.

## 14. Testing strategy

Implementation follows TDD.

### Unit tests

Cover:

- contract JSON parsing and validation;
- low-confidence abstention;
- SQL AST normalization;
- projection extraction;
- aggregate extraction;
- DISTINCT detection;
- GROUP BY extraction;
- ORDER BY and LIMIT extraction;
- predicate/literal extraction;
- alias resolution;
- FK graph construction;
- valid FK joins;
- invalid join edges;
- each violation type;
- malformed contract fail-open;
- AST parse fail-open.

### Regression fixtures

Use known failure patterns from the frozen evaluation:

- entity grain / COUNT(DISTINCT);
- projection mismatch;
- invalid join key;
- row-level top-1 vs aggregate top-1;
- tie/top-k behavior;
- correct query left untouched.

Regression tests must include positive and negative controls to constrain false positives.

### Integration tests

Verify:

- SCV disabled preserves historical guarded flow;
- SCV skip states do not change SQL;
- hard violation triggers exactly one targeted repair;
- repaired SQL goes through existing preflight;
- no second semantic repair occurs.

## 15. Frozen-100 ablation

The benchmark must isolate SCV.

For every frozen case:

1. generate/store the XiYan candidate once;
2. Arm A evaluates that exact candidate as XiYan baseline;
3. Arm B evaluates the same candidate with SCV enabled.

Report:

- baseline execution accuracy under the frozen evaluator;
- SCV execution accuracy under the same frozen evaluator;
- execution success;
- wrong→correct;
- correct→wrong;
- net correct delta;
- SQL changed count;
- violation counts by type;
- repair success by type;
- SCV skip count;
- median semantic overhead.

Do not call the result “official BIRD EX” unless the evaluator exactly matches the official BIRD execution evaluation. If the project’s local set-equality scorer is used, label it explicitly as the project’s frozen paired execution-accuracy metric.

## 16. Evidence and interpretation rules

A positive SCV result supports only:

> Structural semantic verification improved this frozen paired evaluation under the stated model, schema-selection, and evaluator setup.

It does not support:

- BIRD leaderboard claims;
- general Text-to-SQL superiority;
- SOTA language;
- attribution of XiYan model gains to SCV.

A negative result is retained as an engineering decision and the project is frozen rather than expanded.

## 17. README / portfolio follow-up

README redesign happens only after the ablation.

If SCV passes the gate, the project story becomes:

> A local Text-to-SQL system that separates SQL safety from semantic correctness using an independent typed semantic contract, AST verification, FK-graph checks, and bounded targeted repair.

If SCV fails the gate, the README states that structural semantic verification was evaluated but not promoted because it did not produce a meaningful net gain.

The README visual redesign is a separate bounded task. It should use a clean hero, one architecture visual, one proof block, one representative failure/fix example, engineering decisions, evaluation, and quickstart — not emoji cards, giant badge rows, or large “At a glance” tables.
