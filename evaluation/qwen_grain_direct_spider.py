"""Prospective Qwen-only controlled direct-generation prompt, Spider TRAIN source shift.

Research question: can a *single inline relational-grain constraint* alter
Qwen's SQL generator and improve strict physical execution-match?
This is NOT another Qwen plan/reviewer call and NOT a SQL model finetune.
Both original Qwen9B Guarded and treated generator have identical question,
schema, model digest, sampling seed and repair budget. The treatment changes
only one system instruction during the first SQL generation. Gold SQL is read
only after both outputs have been physically produced and frozen.

Spider TRAIN is **not** independent of possible Qwen pretraining; scores
are strictly screening/proxy evidence, not out-of-training/OOD claims.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256, file_digest
import json
from pathlib import Path
import re
from time import monotonic

from evaluation.run_bird import CountingProvider, _ollama_model_digest
from evaluation.schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.contracts import SQLCandidate
from semantic_sql.direct import generate_direct_sql
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.inference import run_guarded
from semantic_sql.providers import OllamaProvider

PROTOCOL = "qwen9b-grain-first-initial-generation-spider-train-screen-v1"
MODEL = "qwen3.5:9b-q4_K_M"
DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
SOURCE_SHA = "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"
CASES, DBS, SHARDS = 18, 9, 3
# Prior physical Spider DEV experiments and source-structure probes.
EXCLUDED_DBS = frozenset({
    "car_1", "employee_hire_evaluation", "flight_2", "museum_visit",
    "singer", "tvshow", "pets_1", "dog_kennels", "orchestra",
    "voter_1", "course_teach", "poker_player",
    "concert_singer", "cre_Doc_Template_Mgt", "network_1",
    "student_transcripts_tracking", "world_1", "wta_1",
    "real_estate_properties",
})
_AGG = re.compile(r"\b(?:average|mean|total|sum|number|count|many|most|least|fewest)\b", re.I)
_ENTITY = re.compile(r"\b(?:each|every|per|by|group|groups|among)\b", re.I)
# One-variable frozen prompt. It is not allowed to inspect the gold SQL or
# baseline generated SQL; this is a prospective alternative generator.
_GRAIN_HINT = (
    "For each aggregation, identify the entity represented by one output row. "
    "If the question asks for a statistic of per-entity totals/counts, first "
    "aggregate at the entity key with GROUP BY, then calculate the outer "
    "statistic from those entity groups. An average per entity is NOT the "
    "average over raw child rows. For a simple per-entity count, use only "
    "one GROUP BY, not a redundant nested average. Keep only the answer "
    "columns requested in the question. Do not invent relationships."
)


def is_group_question(question: str) -> bool:
    return bool(_AGG.search(question) and _ENTITY.search(question))


def physical(root: Path, dbid: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_]+", dbid):
        raise ValueError("unsafe source db id")
    path = root / dbid / (dbid + ".sqlite")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def frozen_source(train: list[dict], root: Path) -> dict:
    def rank(value: str) -> str:
        return sha256((PROTOCOL + ":" + value).encode()).hexdigest()
    groups = defaultdict(dict)
    for i, row in enumerate(train):
        dbid = str(row.get("db_id") or "")
        question = " ".join(str(row.get("question") or "").split())
        if dbid in EXCLUDED_DBS or not is_group_question(question):
            continue
        groups[dbid].setdefault(question.casefold(), {
            "id": f"{dbid}:{i}", "db_id": dbid, "index": i,
            "question": question,
        })
    candidates = sorted(
        (dbid for dbid, items in groups.items() if len(items) >= 2),
        key=lambda db: rank("db:" + db),
    )
    rows = []
    for dbid in candidates:
        try:
            catalog = DatabaseCatalog.from_sqlite(physical(root, dbid))
        except (OSError, FileNotFoundError):
            continue
        if len(catalog.tables) < 2:
            continue
        rows.extend(sorted(groups[dbid].values(), key=lambda r: rank(r["id"]))[:2])
        if len(rows) == CASES:
            break
    if len(rows) != CASES or len({r["db_id"] for r in rows}) != DBS:
        raise ValueError(
            f"cohort failed physical diverse source gate: "
            f"{len(rows)} cases / {len({r['db_id'] for r in rows})} databases; "
            "do not relax selection post hoc"
        )
    if any("query" in case for case in rows):
        raise AssertionError("source gold SQL leaked into manifest")
    return {
        "protocol": PROTOCOL, "model": MODEL, "model_digest": DIGEST,
        "spider_archive_sha": SOURCE_SHA,
        "seed": PROTOCOL, "source_split": "train", "source_gold_read": False,
        "possible_pretraining_overlap": True,
        "excluded_prior_db_ids": sorted(EXCLUDED_DBS),
        "hint_sha256": sha256(_GRAIN_HINT.encode()).hexdigest(),
        "n": CASES, "physical_dbs": DBS,
        "cohort_sha256": sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "cases": rows,
    }


class GrainDirectProvider:
    """Wrap exactly the first direct model call; repairs use unwrapped Qwen."""

    def __init__(self, counting_provider):
        self.provider = counting_provider

    def complete_text(self, *, system: str, user: str) -> str:
        return self.provider.complete_text(
            system=system + "\n" + _GRAIN_HINT, user=user
        )


def _generate_and_run(db: Path, schema: str, question: str, *,
                      treated: bool) -> dict:
    model = CountingProvider(OllamaProvider(
        model=MODEL, timeout_s=750,
        temperature=0, seed=42, num_predict=700,
    ))
    started = monotonic()
    if treated:
        initial = generate_direct_sql(
            question, schema, GrainDirectProvider(model)
        )
        result = run_guarded(
            database=db, provider=model, question=question,
            schema_context=schema, initial_candidate=initial,
            max_repairs=2, max_rows=50_000,
        )
    else:
        result = run_guarded(
            database=db, provider=model, question=question,
            schema_context=schema, max_repairs=2, max_rows=50_000,
        )
    return {
        "sql": result.candidate.sql if result.candidate else None,
        "status": result.status, "truncated": result.truncated,
        "rows": result.rows if result.status == "ok" else (),
        "model_calls": model.total_calls,
        "elapsed_s": round(monotonic() - started, 2),
    }


def score(rows: list[dict], train: list[dict], root: Path) -> None:
    """Read reference only AFTER both generated arm SQL and rows are frozen."""
    for record in rows:
        source = train[record["index"]]
        if (source["db_id"] != record["db_id"]
            or " ".join(source["question"].split()) != record["question"]):
            raise ValueError("source question drift")
        db = physical(root, record["db_id"])
        try:
            golden = execute_readonly(db, source["query"], max_rows=50_000)
            record["scorable"] = not golden.truncated
        except (SQLExecutionError, ValueError):
            golden, record["scorable"] = None, False
        for arm in ("original", "grain_direct"):
            result = record[arm]
            valid = record["scorable"] and result["status"] == "ok" and not result["truncated"]
            result["set_ex"] = bool(valid and set(map(tuple, result["rows"])) == set(golden.rows)) if record["scorable"] else None
            result["bag_ex"] = bool(valid and Counter(map(tuple, result["rows"])) == Counter(golden.rows)) if record["scorable"] else None
            del result["rows"]
        with db.open("rb") as f:
            record["db_sha256"] = file_digest(f, "sha256").hexdigest()
        del record["index"]


def one_shard(manifest: Path, trainfile: Path, root: Path, shard: int) -> dict:
    if shard not in range(SHARDS):
        raise ValueError("invalid shard")
    train = json.loads(trainfile.read_text())
    frozen = json.loads(manifest.read_text())
    if frozen != frozen_source(train, root):
        raise ValueError("source, seed, hints or cohort changed after freeze")
    if _ollama_model_digest("http://127.0.0.1:11434", MODEL) != DIGEST:
        raise ValueError("Qwen9B model checksum drift")
    rows = []
    for item in frozen["cases"][shard * (CASES // SHARDS):(shard + 1) * (CASES // SHARDS)]:
        db = physical(root, item["db_id"])
        schema, _ = select_schema_context(
            DatabaseCatalog.from_sqlite(db), root, item["db_id"], item["question"]
        )
        original = _generate_and_run(db, schema, item["question"], treated=False)
        treated = _generate_and_run(db, schema, item["question"], treated=True)
        rows.append({
            **item, "original": original, "grain_direct": treated,
        })
        print("QWEN_GRAIN_DIRECT_FROZEN_GENERATIONS",
              item["id"], "calls", original["model_calls"],
              treated["model_calls"], flush=True)
    score(rows, train, root)
    return {
        "protocol": PROTOCOL, "shard": shard, "digest": DIGEST,
        "cohort_sha256": frozen["cohort_sha256"],
        "cases": rows,
    }


def aggregate(shards: list[dict]) -> dict:
    if (len(shards) != SHARDS
        or sorted(x["shard"] for x in shards) != list(range(SHARDS))
        or any(
            x["protocol"] != PROTOCOL or x["digest"] != DIGEST
            or len(x["cases"]) != CASES // SHARDS
            for x in shards
        )
        or len({x["cohort_sha256"] for x in shards}) != 1):
        raise ValueError("incomplete grain experiment matrix")
    rows = [
        item for part in sorted(shards, key=lambda s:s["shard"])
        for item in part["cases"]
    ]
    if (len(rows) != CASES or len({r["id"] for r in rows}) != CASES
        or len({r["db_id"] for r in rows}) != DBS
        or any(
            len({r["db_sha256"] for r in rows if r["db_id"] == db}) != 1
            for db in {r["db_id"] for r in rows}
        )):
        raise ValueError("original source cohort or physical DB mismatch")
    scorable = [r for r in rows if r["scorable"]]
    rescues = [r for r in scorable if r["grain_direct"]["set_ex"] and not r["original"]["set_ex"]]
    regress = [r for r in scorable if r["original"]["set_ex"] and not r["grain_direct"]["set_ex"]]
    scores = {
        arm: {
            "set_ex": sum(r[arm]["set_ex"] for r in scorable),
            "bag_ex": sum(r[arm]["bag_ex"] for r in scorable),
            "calls": sum(r[arm]["model_calls"] for r in rows),
            "seconds": round(sum(r[arm]["elapsed_s"] for r in rows), 2),
        }
        for arm in ("original", "grain_direct")
    }
    positive = (
        len(scorable) == CASES and len(rescues) - len(regress) >= 2
        and not regress
        and len({r["db_id"] for r in rescues}) >= 2
        and scores["grain_direct"]["bag_ex"] >= scores["original"]["bag_ex"]
    )
    return {
        "protocol": PROTOCOL, "n": CASES,
        "database_count": DBS, "scorable": len(scorable),
        "sources": "Spider TRAIN, potential LLM pretraining overlap",
        "arms": scores,
        "rescues": [r["id"] for r in rescues],
        "regressions": [r["id"] for r in regress],
        "physical_dbs": sorted({r["db_id"] for r in rows}),
        "screen_pass_only": positive,
        "promote": False,
        "bird100_score_unchanged": True,
        "training": False,
        "no_additional_semantic_planner": True,
        "cases": rows,
    }


def main():
    p = argparse.ArgumentParser()
    for key in ("train", "db-root", "manifest", "output"):
        p.add_argument("--" + key, type=Path)
    p.add_argument("--shard", type=int)
    p.add_argument("--inputs", type=Path, nargs="+")
    a = p.parse_args()
    if a.output is None or a.output.exists():
        raise FileExistsError("fresh output path required")
    if a.inputs:
        result = aggregate([json.loads(f.read_text()) for f in a.inputs])
        print("QWEN_GRAIN_DIRECT_FINAL",
              json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),
              flush=True)
    elif a.manifest:
        if any(x is None for x in (a.train, a.db_root, a.shard)):
            p.error("paired shard requires train/db-root/shard")
        result = one_shard(a.manifest, a.train, a.db_root, a.shard)
    else:
        if any(x is None for x in (a.train, a.db_root)):
            p.error("freeze requires train/db-root")
        result = frozen_source(json.loads(a.train.read_text()), a.db_root)
        print("QWEN_GRAIN_DIRECT_FREEZE",
              result["cohort_sha256"],
              [x["db_id"] for x in result["cases"]], flush=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")


if __name__ == "__main__":
    main()
