#!/usr/bin/env python3
"""Freeze BIRD DEV500 + untouched DEV300 from a pinned BIRD DEV source.

Requires historical frozen100 IDs and official difficulty labels (from pinned source rows or a trusted map).
Does not open gold SQL for selection, and never overwrites a manifest.

python scripts/freeze_bird_split.py --source dev.json \
 --historical-ids original_frozen100_ids.json \
 --output manifests/bird_dev500_holdout300.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ranked(case_id: str, seed: str) -> str:
    return sha(f"{seed}:{case_id}".encode("utf-8"))


def load_source(source_path: Path) -> list[dict]:
    rows = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("source must be a nonempty JSON array")
    if any(not isinstance(r, dict) for r in rows):
        raise ValueError("source rows must be JSON objects")
    return rows


def case_id(row: dict, index: int) -> str:
    for key in ("question_id", "id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return str(index)


def load_history(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = next(
            (payload[k] for k in ("case_ids", "dev_case_ids", "task_ids", "ids") if isinstance(payload.get(k), list)),
            None,
        )
    if not isinstance(payload, list) or not payload:
        raise ValueError("historical frozen IDs must be a JSON list or manifest with case_ids")
    values = [str(x).removeprefix("bird-") for x in payload]
    if len(values) != len(set(values)):
        raise ValueError("historical IDs contain duplicates")
    return values


def allocate(capacities: dict[tuple[str, str], int], amount: int) -> dict[tuple[str, str], int]:
    total = sum(capacities.values())
    if amount < 0 or amount > total:
        raise ValueError("requested split is larger than eligible source")
    if total == 0:
        return {key: 0 for key in capacities}
    floor = {k: (amount * n) // total for k, n in capacities.items()}
    remainder = amount - sum(floor.values())
    order = sorted(capacities, key=lambda k: (-(amount * capacities[k] % total), k))
    for key in order[:remainder]:
        floor[key] += 1
    assert sum(floor.values()) == amount
    assert all(floor[k] <= capacities[k] for k in capacities)
    return floor


def freeze(
    rows: list[dict],
    history: list[str],
    difficulties: dict[str, str],
    source_sha: str,
    history_sha: str,
    difficulty_sha: str,
    seed: str = "bird-metric-gate-20261116-v1",
    dev_size: int = 500,
    holdout_size: int = 300,
    historical_expected: int = 100,
) -> dict:
    if len(history) != historical_expected:
        raise ValueError(f"expected exactly {historical_expected} historical IDs; got {len(history)}")
    if not isinstance(difficulties, dict):
        raise ValueError("difficulty map must be a dictionary mapping case ID to official difficulty")
    ids = {}
    semantic_key_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for i, row in enumerate(rows):
        cid = case_id(row, i)
        if cid in ids:
            raise ValueError(f"duplicate BIRD DEV case ID: {cid}")
        if not row.get("db_id") or not row.get("question"):
            raise ValueError(f"row {cid} lacks db_id or question")
        k = (str(row["db_id"]), " ".join(str(row["question"]).split()).casefold())
        semantic_key_groups[k].append(cid)
        ids[cid] = row
    missing = set(history) - ids.keys()
    if missing:
        raise ValueError(f"historical IDs absent from source: {len(missing)}")
    # Freeze policy: remove ALL repeated exact within-database questions from BOTH
    # partitions. Group assignment could also be safe, but removing them avoids
    # dependent paraphrase weights and any overlap with a historical ID. This
    # rule uses only question text + DB; never gold SQL, model outcomes or labels.
    duplicate_groups = [members for members in semantic_key_groups.values() if len(members) > 1]
    duplicate_ids = {cid for members in duplicate_groups for cid in members}
    historical_set = set(history)
    excluded_ids = duplicate_ids | historical_set
    if dev_size + holdout_size > len(ids) - len(excluded_ids):
        raise ValueError("not enough non-duplicate novel DEV questions to meet requested sizes")
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for cid, row in ids.items():
        if cid in excluded_ids:
            continue
        value = difficulties.get(cid, row.get("difficulty"))
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"official difficulty label missing for case ID {cid}")
        groups[(str(row["db_id"]), value.strip())].append(cid)
    for group in groups:
        groups[group].sort(key=lambda cid: (ranked(cid, seed), cid))
    dev_alloc = allocate({k: len(v) for k, v in groups.items()}, dev_size)
    remain = {k: len(groups[k]) - dev_alloc[k] for k in groups}
    hold_alloc = allocate(remain, holdout_size)
    dev_ids = sorted(cid for k in groups for cid in groups[k][:dev_alloc[k]])
    hold_ids = sorted(cid for k in groups for cid in groups[k][dev_alloc[k]:dev_alloc[k] + hold_alloc[k]])
    assert len(dev_ids) == dev_size and len(hold_ids) == holdout_size
    assert not (set(dev_ids) & set(hold_ids))
    assert not ((set(dev_ids) | set(hold_ids)) & set(excluded_ids))
    db_dev = Counter(str(ids[cid]["db_id"]) for cid in dev_ids)
    db_hold = Counter(str(ids[cid]["db_id"]) for cid in hold_ids)
    return {
        "protocol": "bird-dev-stratified-question-disjoint/v1",
        "population": "BIRD DEV; question-disjoint but NOT database/source independent",
        "source_sha256": source_sha,
        "historical_ids_sha256": history_sha,
        "difficulty_map_sha256": difficulty_sha,
        "seed": seed,
        "source_rows": len(rows),
        "excluded_historical_count": len(history),
        "excluded_exact_duplicate_question_groups": len(duplicate_groups),
        "excluded_exact_duplicate_question_ids": len(duplicate_ids - historical_set),
        "eligible_novel_rows": len(ids) - len(excluded_ids),
        "dev_size": dev_size,
        "holdout_size": holdout_size,
        "dev_case_ids": dev_ids,
        "holdout_case_ids": hold_ids,
        "dev_by_db": dict(sorted(db_dev.items())),
        "holdout_by_db": dict(sorted(db_hold.items())),
        "difficulty_source": "official row labels / user-provided keyed difficulty map; never inferred from SQL",
        "holdout_policy": "freeze before inspecting outputs; one final use only",
        "warning": "Old BIRD frozen100 visited all 11 physical BIRD DEV databases. Neither new split is source-independent.",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--historical-ids", type=Path, required=True)
    p.add_argument("--difficulty-map", type=Path, default=None,
                   help="Optional trusted case-ID difficulty map; otherwise use the pinned DEV rows' native difficulty field")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", default="bird-metric-gate-20261116-v1")
    p.add_argument("--dev-size", type=int, default=500)
    p.add_argument("--holdout-size", type=int, default=300)
    args = p.parse_args()
    try:
        rows = load_source(args.source)
        history = load_history(args.historical_ids)
        difficulties = json.loads(args.difficulty_map.read_text(encoding="utf-8")) if args.difficulty_map else {}
        diff_sha = sha(args.difficulty_map.read_bytes()) if args.difficulty_map else sha(args.source.read_bytes())
        result = freeze(rows, history, difficulties, sha(args.source.read_bytes()),
                        sha(args.historical_ids.read_bytes()), diff_sha,
                        args.seed, args.dev_size, args.holdout_size)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, sort_keys=True, indent=2)
            handle.write("\n")
        print(json.dumps({"status": "FROZEN", "dev": result["dev_size"], "holdout": result["holdout_size"]}))
        return 0
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
