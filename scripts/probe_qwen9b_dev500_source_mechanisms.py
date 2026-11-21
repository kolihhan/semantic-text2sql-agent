"""Original Qwen3.5-9B DEV500 CPU-only semantic-source eligibility screen.

Two PREDECLARED mechanisms: explicit scalar growth-rate answer shape and nth-rank
ordinal discrepancy. This is a retrospective source-only coverage diagnostic.
No SQL repair, no model inference, no publisher gold SQL, no benchmark scorer.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

SOURCE_RAW_SHA = "ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb"
SOURCE_SHA = SOURCE_RAW_SHA
MODEL_SHA = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
CACHE_PROTO = "FROZEN-BIRD-DEV500-QWEN35-9B-KCACHE-V1"
FROZEN_PROTOCOL = "BIRD_DEV500_QWEN9B_GOLD_FREE_SOURCE_V1"
SCREEN_PROTOCOL = "SOURCE_ONLY_RATIO_ORDINAL_ELIGIBILITY_V1"

# Explicit ordinal followed by an extremum; excludes "top 5", "at least 5",
# and unspecified "first name" requests. No LLM or answer labels involved.
ORDINAL = re.compile(
    r"\b(?P<rank>\d{1,2}(?:st|nd|rd|th)|second|third|fourth|fifth|sixth|"
    r"seventh|eighth|ninth|tenth)\s+(?:\w+\s+){0,3}"
    r"(?:highest|lowest|largest|smallest|best|worst|most|least)\b",
    re.I,
)
NAMES = {
    "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}
RATIO_REQUEST = re.compile(r"\b(?:growth rate|percentage change|percent change)\b", re.I)
NUMERATOR = re.compile(r"\bas\s+[\[\"]?numerator\b", re.I)
DENOMINATOR = re.compile(r"\bas\s+[\[\"]?denominator\b", re.I)
DIVISION = re.compile(r"/|\bdivide\s*\(", re.I)
RANKS_USED = re.compile(r"\b(?:row_number|dense_rank|rank)\s*\(|\boffset\b", re.I)
FINAL_LIMIT1 = re.compile(r"\blimit\s+1\s*;?\s*$", re.I)


def classify(question: str, sql: str) -> tuple[str, ...]:
    """Return source-level *suspicions*, never verified semantic contradictions."""
    flagged = []
    if (RATIO_REQUEST.search(question) and NUMERATOR.search(sql)
        and DENOMINATOR.search(sql) and not DIVISION.search(sql)):
        flagged.append("SCALAR_GROWTH_RATE_SPLIT_OUTPUT")
    m = ORDINAL.search(question)
    if m:
        raw_rank = m["rank"].casefold()
        rank = NAMES.get(raw_rank) if raw_rank in NAMES else int(re.match(r"\d+", raw_rank).group())
        if (rank >= 2 and re.search(r"\border\s+by\b", sql, re.I)
            and FINAL_LIMIT1.search(sql) and not RANKS_USED.search(sql)):
            flagged.append("NTH_RANK_LIMIT1_WITHOUT_OFFSET")
    return tuple(flagged)


def digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def write_new(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def freeze(*, raw_source: Path, out: Path) -> dict:
    raw = raw_source.read_bytes()
    if sha256(raw).hexdigest() != SOURCE_RAW_SHA:
        raise ValueError("BIRD source SHA drift")
    full_rows = json.loads(raw)
    if len(full_rows) != 1534:
        raise ValueError("wrong original BIRD source universe")
    sanitized = []
    ids = set()
    for idx, row in enumerate(full_rows):
        cid = str(row.get("question_id") or row.get("id") or idx)
        question, db_id = row["question"], row["db_id"]
        if cid in ids or not isinstance(question, str) or not isinstance(db_id, str):
            raise ValueError("source identity/schema drift")
        ids.add(cid)
        # The only persistent source record deliberately omits reference SQL,
        # correctness, difficulty, and other original keys.
        sanitized.append({"case_id": cid, "db_id": db_id, "question": question})
    out_obj = {
        "protocol": FROZEN_PROTOCOL,
        "source_sha256": SOURCE_SHA,
        "contains_reference_or_label": False,
        "questions": sanitized,
    }
    assert all(set(x) == {"case_id", "db_id", "question"} for x in sanitized)
    write_new(out, out_obj)
    return {"questions_sanitized": len(sanitized), "gold_persisted": False}


def screen(*, source: Path, cache_dir: Path, out: Path) -> dict:
    src = json.loads(source.read_text(encoding="utf-8"))
    if (src.get("protocol") != FROZEN_PROTOCOL
        or src.get("source_sha256") != SOURCE_SHA
        or src.get("contains_reference_or_label") is not False):
        raise ValueError("gold-free source identity mismatch")
    questions = src.get("questions")
    if not isinstance(questions, list) or len(questions) != 1534:
        raise ValueError("question list absent/incomplete")
    if any(set(x) != {"case_id", "db_id", "question"} for x in questions):
        raise ValueError("source file includes extra fields: reject to protect gold blindness")
    lookup = {x["case_id"]: x for x in questions}
    if len(lookup) != 1534:
        raise ValueError("nonunique source ids")

    files = sorted(cache_dir.glob("k8-dev500-candidate-cache-*.json"))
    if len(files) != 32:
        raise ValueError("expected exactly 32 original cache shard files")
    seen_shards, cases = set(), {}
    for file in files:
        cache = json.loads(file.read_text(encoding="utf-8"))
        if (cache.get("protocol") != CACHE_PROTO
            or cache.get("status") != "CANDIDATE_CACHE_COMPLETE"
            or cache.get("source_sha256") != SOURCE_SHA
            or cache.get("model_digest") != MODEL_SHA
            or cache.get("no_gold_sql_in_cache") is not True
            or cache.get("k") != 8):
            raise ValueError("original Qwen K8 provenance mismatch")
        shard = cache["shard"]
        if shard in seen_shards:
            raise ValueError("duplicate shard")
        seen_shards.add(shard)
        for case in cache["cases"]:
            cid = str(case["case_id"])
            if cid in cases or cid not in lookup:
                raise ValueError("duplicate/missing case source")
            q = lookup[cid]
            if case["db_id"] != q["db_id"] or case["question_sha256"] != digest(q["question"]):
                raise ValueError("question or physical DB drift")
            samples = sorted(case["samples"], key=lambda x: x["index"])
            if len(samples) != 8 or [x["index"] for x in samples] != list(range(8)):
                raise ValueError("K8 source incomplete")
            if any(not isinstance(x.get("sql"), str) for x in samples):
                raise ValueError("original SQL missing")
            cases[cid] = (q, [x["sql"] for x in samples])
    if seen_shards != set(range(32)) or len(cases) != 500:
        raise ValueError("original 500/32 source coverage failed")

    families = ("SCALAR_GROWTH_RATE_SPLIT_OUTPUT", "NTH_RANK_LIMIT1_WITHOUT_OFFSET")
    flags = []
    for cid, (q, sqls) in sorted(cases.items(), key=lambda x: int(x[0])):
        triggered = [classify(q["question"], sql) for sql in sqls]
        for family in families:
            indices = [i for i, codes in enumerate(triggered) if family in codes]
            if indices:
                flags.append({
                    "case_id": cid, "db_id": q["db_id"], "family": family,
                    "k1_flag": 0 in indices, "flagged_k_indices": indices,
                    "question_sha256": digest(q["question"]),
                    "k1_sql_sha256": digest(sqls[0]),
                })
    results = {}
    for family in families:
        family_flags = [x for x in flags if x["family"] == family]
        first = [x for x in family_flags if x["k1_flag"]]
        dbs = sorted({x["db_id"] for x in first})
        results[family] = {
            "k1_source_flagged_cases": len(first),
            "k1_physical_database_count": len(dbs),
            "k1_physical_db_ids": dbs,
            "any_of_8_flagged_questions": len(family_flags),
            "gate_source_coverage_pass": len(first) >= 3 and len(dbs) >= 2,
            "repair_and_model_allowed_by_this_screen": False,
        }
    report = {
        "protocol": SCREEN_PROTOCOL,
        "source_sha256": SOURCE_SHA, "qwen_model_digest": MODEL_SHA,
        "n_original_cases": len(cases), "n_original_k8_sqls": 8 * len(cases),
        "n_cache_shards": len(files),
        "known_dev500_previously_exposed": True,
        "no_reference_sql_accessed_by_screen": True,
        "no_label_scoring": True, "no_qwen_calls": True,
        "no_sql_repair": True, "holdout300_untouched": True,
        "families": results, "flags": flags,
        "research_only_no_performance_promotion": True,
    }
    write_new(out, report)
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    subs = p.add_subparsers(dest="cmd", required=True)
    f = subs.add_parser("freeze")
    f.add_argument("--raw-source", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    s = subs.add_parser("screen")
    s.add_argument("--source", type=Path, required=True)
    s.add_argument("--cache-dir", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = (freeze(raw_source=a.raw_source, out=a.out) if a.cmd == "freeze"
              else screen(source=a.source, cache_dir=a.cache_dir, out=a.out))
    print("QWEN9B_DEV500_SOURCE_ONLY_PROBE", json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
