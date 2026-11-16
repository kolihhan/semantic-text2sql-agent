#!/usr/bin/env python3
"""Fail-closed corpus overlap audit. Does not contact networks or models.

Usage (Text2SQL):
 python scripts/check_data_contamination.py --train bird_train.json \
   --exclude bird-dev=bird_dev.json --required-role bird-dev --output report.json

Usage (agent):
 python scripts/check_data_contamination.py --train trajectories.jsonl \
   --exclude gaia-all=gaia_reference.jsonl --exclude bird-dev=bird_dev.json \
   --required-role gaia-all --required-role bird-dev --output report.json

A PASS proves only no detectable lexical overlap against the *supplied* corpora,
not that the reference corpus is complete or semantically decontaminated.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

TEXT_FIELDS = {
    "question", "Question", "instruction", "prompt", "task", "input",
    "SQL", "sql", "query", "answer", "final_answer", "Final answer",
    "content", "text", "response",
}


def load_rows(path: Path) -> list[dict]:
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f"missing or empty corpus: {path}")
    content = path.read_text(encoding="utf-8-sig")
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        obj = [json.loads(s) for s in content.splitlines() if s.strip()]
    if isinstance(obj, dict):
        obj = next((obj[k] for k in ("rows", "data", "examples", "tasks") if isinstance(obj.get(k), list)), [obj])
    if not isinstance(obj, list) or not obj or any(not isinstance(r, dict) for r in obj):
        raise ValueError(f"corpus must contain nonempty list of objects: {path}")
    return obj


def texts(obj: object) -> list[str]:
    found: list[str] = []
    def walk(value: object, include: bool = False) -> None:
        if isinstance(value, str) and include and value.strip():
            found.append(value)
        elif isinstance(value, list):
            for item in value:
                walk(item, include)
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(item, include or key in TEXT_FIELDS)
    walk(obj)
    return found


def normalize(value: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE))


def grams(value: str) -> frozenset[str]:
    compact = value.replace(" ", "")
    if len(compact) < 12:
        return frozenset()
    return frozenset(compact[i:i+5] for i in range(len(compact) - 4))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audit(train_path: Path, exclusion_paths: dict[str, Path], required_roles: set[str], threshold: float = 0.82) -> dict:
    if not required_roles or required_roles - exclusion_paths.keys():
        raise ValueError("required reference roles missing; must supply all required corpora")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("near threshold must be in (0, 1]")
    train_rows = load_rows(train_path)
    references: list[tuple[str, str, frozenset[str]]] = []
    reference_hashes = {}
    for role, path in sorted(exclusion_paths.items()):
        rows = load_rows(path)
        if role == "bird-dev" and any(not isinstance(row.get("question"), str) or not isinstance(row.get("SQL"), str) for row in rows):
            raise ValueError("bird-dev reference must contain question and SQL on every row")
        if role == "gaia-all" and any(not isinstance(row.get("Question", row.get("question")), str) for row in rows):
            raise ValueError("gaia-all reference must contain Question on every row")
        extracted = [normalize(v) for row in rows for v in texts(row)]
        extracted = sorted({v for v in extracted if len(v) >= 12})
        if not extracted:
            raise ValueError(f"no usable reference texts: {role}")
        references.extend((role, v, grams(v)) for v in extracted)
        reference_hashes[role] = {"sha256": digest(path.read_bytes()), "rows": len(rows), "texts": len(extracted)}
    index: dict[str, set[int]] = defaultdict(set)
    exact: dict[str, set[str]] = defaultdict(set)
    for ix, (role, value, shingles) in enumerate(references):
        exact[value].add(role)
        for gram in shingles:
            index[gram].add(ix)

    findings: list[dict] = []
    checked = 0
    for row_ix, row in enumerate(train_rows):
        values = sorted({normalize(t) for t in texts(row) if len(normalize(t)) >= 12})
        if not values:
            raise ValueError(f"training row {row_ix} has no auditable text")
        for value in values:
            checked += 1
            found = set()
            if value in exact:
                found |= {(role, "exact") for role in exact[value]}
            sh = grams(value)
            if len(sh) >= 6:
                candidate_counts: dict[int, int] = defaultdict(int)
                for g in sh:
                    for ix in index.get(g, ()):
                        candidate_counts[ix] += 1
                # Only a small number of candidates can satisfy Jaccard >= threshold.
                for ix, common in candidate_counts.items():
                    role, ref, ref_sh = references[ix]
                    upper = common / max(len(sh), len(ref_sh))
                    if upper < threshold:
                        continue
                    sim = common / (len(sh) + len(ref_sh) - common)
                    if sim >= threshold:
                        found.add((role, "near"))
            # Never expose reference or protected training question content.
            for role, kind in sorted(found):
                findings.append({
                    "train_row": row_ix,
                    "train_text_sha256": digest(value.encode("utf-8")),
                    "reference_role": role,
                    "overlap": kind,
                })
    # Finding count is complete even when the report's examples are capped.
    return {
        "status": "CONTAMINATED" if findings else "PASS_LEXICAL_SCREEN_ONLY",
        "train_sha256": digest(train_path.read_bytes()),
        "train_rows": len(train_rows),
        "checked_train_texts": checked,
        "reference_sources": reference_hashes,
        "threshold": threshold,
        "finding_count": len(findings),
        "findings_preview": findings[:100],
        "scope": "Supplied-corpus lexical screen only; no proof of reference completeness or semantic non-contamination.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--exclude", action="append", required=True, help="ROLE=/path/to/corpus.json[l]")
    parser.add_argument("--required-role", action="append", required=True)
    parser.add_argument("--threshold", type=float, default=0.82)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        excluded = {}
        for s in args.exclude:
            role, separator, raw = s.partition("=")
            if not separator or not role or not raw or role in excluded:
                raise ValueError("--exclude must be unique ROLE=PATH")
            excluded[role] = Path(raw)
        result = audit(args.train, excluded, set(args.required_role), args.threshold)
    except (ValueError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        result = {"status": "BLOCKED", "error": str(exc), "scope": "Do not train with an unverified audit"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit("refusing to overwrite audit report")
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "finding_count": result.get("finding_count")}))
    return 0 if result["status"] == "PASS_LEXICAL_SCREEN_ONLY" else 2


if __name__ == "__main__":
    sys.exit(main())
