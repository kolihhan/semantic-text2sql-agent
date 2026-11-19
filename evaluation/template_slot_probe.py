"""Blind, matched four-call SQL candidate-slot ablation.

Input contains only physically executed candidate result signatures; no gold SQL,
correctness flags, reference answers or DEV labels may enter selection.
Each arm has exactly four calls: A=[1,2,3,4], B=[T,1,2,3],
C=[4,1,2,3]. C isolates first-slot/tie effects from template effects.
No-template cases fall back to A in both arms.

This module does not generate SQL or claim an accuracy improvement. It seals
choices before an independent scorer supplies reference result signatures.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

PROTOCOL = "blind-template-slot-4call-v1"
CASE_KEYS = {"case_hash", "db_id", "baseline", "template"}
CANDIDATE_KEYS = {"id", "valid", "signature"}
SOURCE_KEYS = {"protocol", "manifest_sha256", "model_digest", "cases"}


def _fingerprint(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _candidate(value: dict, expected_id: str) -> dict:
    if not isinstance(value, dict) or set(value) != CANDIDATE_KEYS:
        raise ValueError("candidate shape changed: no gold, SQL, or reference fields allowed")
    if value["id"] != expected_id or type(value["valid"]) is not bool:
        raise ValueError("candidate identity or validity differs from sealed protocol")
    sig = value["signature"]
    if value["valid"]:
        if not isinstance(sig, str) or not sig:
            raise ValueError("valid candidate needs physical result signature")
    elif sig is not None:
        raise ValueError("invalid candidate must have null result signature")
    return dict(value)


def _choose(candidates: list[dict]) -> dict:
    """The project's exact plurality rule: earliest valid on a vote tie."""
    if len(candidates) != 4:
        raise ValueError("each arm must use exactly four candidates")
    support = Counter(c["signature"] for c in candidates if c["valid"])
    if not support:
        return {"chosen_id": candidates[0]["id"], "signature": None, "votes": 0}
    max_votes = max(support.values())
    chosen = next(c for c in candidates
                  if c["valid"] and support[c["signature"]] == max_votes)
    return {"chosen_id": chosen["id"], "signature": chosen["signature"], "votes": max_votes}


def seal_choices(source: dict) -> dict:
    if not isinstance(source, dict) or set(source) != SOURCE_KEYS:
        raise ValueError("source must be a frozen gold-free manifest and candidates")
    if source["protocol"] != PROTOCOL or not isinstance(source["cases"], list):
        raise ValueError("source protocol mismatch")
    for field in ("manifest_sha256", "model_digest"):
        if not isinstance(source[field], str) or not source[field]:
            raise ValueError(f"missing frozen {field}")
    cases, seen = [], set()
    for row in source["cases"]:
        if not isinstance(row, dict) or set(row) != CASE_KEYS:
            raise ValueError("case includes forbidden fields, possibly answer labels")
        case_hash, db_id = row["case_hash"], row["db_id"]
        if not isinstance(case_hash, str) or len(case_hash) != 64 or \
           not isinstance(db_id, str) or not db_id or case_hash in seen:
            raise ValueError("duplicate or invalid hashed case identity")
        seen.add(case_hash)
        if not isinstance(row["baseline"], list) or len(row["baseline"]) != 4:
            raise ValueError("four original candidates required")
        original = [_candidate(c, f"c{i}") for i, c in enumerate(row["baseline"], 1)]
        template = row["template"]
        if template is not None:
            template = _candidate(template, "template")
            # This is an ACTUAL extra generated candidate, not an oracle rewrite.
            b = [template, *original[:3]]
            c = [original[3], *original[:3]]
        else:
            b = c = original
        a = original
        cases.append({
            "case_hash": case_hash, "db_id": db_id,
            "template_activated": template is not None,
            "arm_candidates": {"A": a, "B": b, "C": c},
            "choices": {name: _choose(items)
                        for name, items in (("A", a), ("B", b), ("C", c))},
        })
    if not cases:
        raise ValueError("empty frozen evaluation")
    payload = {
        "protocol": PROTOCOL, "manifest_sha256": source["manifest_sha256"],
        "model_digest": source["model_digest"], "cases": cases,
        "source_sha256": _fingerprint(source),
        "gold_was_unavailable_to_selector": True,
    }
    payload["selections_sha256"] = _fingerprint(payload)
    return payload


def score_sealed(sealed: dict, references: dict) -> dict:
    if sealed.get("protocol") != PROTOCOL or \
       sealed.get("gold_was_unavailable_to_selector") is not True:
        raise ValueError("not a blind sealed selection")
    stamped = dict(sealed)
    signature = stamped.pop("selections_sha256", None)
    if signature != _fingerprint(stamped):
        raise ValueError("selection artifact changed after sealing")
    if not isinstance(references, dict) or set(references) != {"manifest_sha256", "cases"} \
       or references["manifest_sha256"] != sealed["manifest_sha256"]:
        raise ValueError("reference manifest mismatch")
    gold = {}
    for r in references["cases"]:
        if not isinstance(r, dict) or set(r) != {"case_hash", "scorable", "set_hash"}:
            raise ValueError("unexpected reference field")
        if r["case_hash"] in gold or type(r["scorable"]) is not bool:
            raise ValueError("duplicate or invalid reference")
        if r["scorable"] and (not isinstance(r["set_hash"], str) or not r["set_hash"]):
            raise ValueError("scorable gold must contain real bounded result hash")
        if not r["scorable"] and r["set_hash"] is not None:
            raise ValueError("unscorable reference cannot provide a result")
        gold[r["case_hash"]] = r
    if set(gold) != {r["case_hash"] for r in sealed["cases"]}:
        raise ValueError("all sealed cases must be scored, no silent filtering")
    scores = {name: 0 for name in ("A", "B", "C")}
    oracle = {name: 0 for name in ("A", "B")}
    rescue = regression = 0
    covered = scorable = 0
    by_db: dict[str, dict] = {}
    for row in sealed["cases"]:
        r = gold[row["case_hash"]]
        covered += int(row["template_activated"])
        if not r["scorable"]:
            continue
        scorable += 1
        g = r["set_hash"]
        db_counts = by_db.setdefault(row["db_id"], {"n": 0, "A": 0, "B": 0, "C": 0})
        db_counts["n"] += 1
        flags = {}
        for arm in scores:
            hit = row["choices"][arm]["signature"] == g
            flags[arm] = hit
            scores[arm] += int(hit)
            db_counts[arm] += int(hit)
        for arm in oracle:
            oracle[arm] += int(any(c["valid"] and c["signature"] == g
                                   for c in row["arm_candidates"][arm]))
        rescue += int(not flags["A"] and flags["B"])
        regression += int(flags["A"] and not flags["B"])
    return {
        "protocol": PROTOCOL, "manifest_sha256": sealed["manifest_sha256"],
        "selections_sha256": signature, "source_sha256": sealed["source_sha256"],
        "n_total": len(sealed["cases"]), "n_scorable": scorable,
        "n_unscorable": len(sealed["cases"]) - scorable,
        "template_activated": covered, "final_correct": scores,
        "oracle_at_least_one_correct": oracle,
        "B_vs_A_rescues": rescue, "B_vs_A_regressions": regression,
        "B_vs_A_net": rescue-regression,
        "by_database": by_db,
        "warning": "Not a model run. Candidate generation, physical bounded SQL execution, and an independent source-disjoint prospective test are still required.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("seal", "score"))
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--sealed", type=Path)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("do not overwrite immutable experimental artifacts")
    if args.mode == "seal":
        if not args.candidates or args.references or args.sealed:
            parser.error("seal needs only --candidates")
        output = seal_choices(json.loads(args.candidates.read_text("utf-8")))
    else:
        if not args.sealed or not args.references or args.candidates:
            parser.error("score needs --sealed and --references")
        output = score_sealed(json.loads(args.sealed.read_text("utf-8")),
                              json.loads(args.references.read_text("utf-8")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", "utf-8")
    print(json.dumps({k: v for k, v in output.items() if k != "by_database"},
                     sort_keys=True))


if __name__ == "__main__":
    main()
