from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_source_hash(path: str | Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"source hash mismatch: expected {expected_sha256}, got {actual}")


def _rank(case_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).hexdigest()


def create_split_manifest(
    case_ids: Iterable[str],
    *,
    source_sha256: str,
    seed: str,
    dev_size: int,
    final_size: int,
    fixed_dev_ids: Iterable[str] = (),
) -> dict[str, object]:
    unique_ids = sorted({str(case_id) for case_id in case_ids})
    available = set(unique_ids)
    fixed = [str(case_id) for case_id in fixed_dev_ids]
    if len(set(fixed)) != len(fixed):
        raise ValueError("fixed DEV case IDs must be unique")
    missing = [case_id for case_id in fixed if case_id not in available]
    if missing:
        raise ValueError(f"fixed DEV case IDs are missing from source: {missing[:5]}")

    ranked = sorted(unique_ids, key=lambda case_id: (_rank(case_id, seed), case_id))
    if fixed:
        dev_ids = sorted(fixed)
        remainder = [case_id for case_id in ranked if case_id not in set(dev_ids)]
    else:
        if dev_size < 0:
            raise ValueError("dev_size must be non-negative")
        dev_ids = ranked[:dev_size]
        remainder = ranked[dev_size:]

    if final_size < 0:
        raise ValueError("final_size must be non-negative")
    if final_size > len(remainder):
        raise ValueError(f"requested {final_size} FINAL cases but only {len(remainder)} remain")
    final_ids = remainder[:final_size]

    return {
        "protocol_version": 1,
        "source_sha256": source_sha256,
        "seed": seed,
        "dev_case_ids": dev_ids,
        "final_case_ids": final_ids,
    }


def selected_case_ids(manifest: dict[str, object], split: str) -> list[str]:
    if split not in {"dev", "final"}:
        raise ValueError("split must be 'dev' or 'final'")
    key = f"{split}_case_ids"
    values = manifest.get(key)
    if not isinstance(values, list):
        raise ValueError(f"manifest is missing list field {key!r}")
    return [str(value) for value in values]


def load_case_ids_from_manifest(payload: object) -> list[str]:
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        for key in ("case_ids", "dev_case_ids"):
            if isinstance(payload.get(key), list):
                values = payload[key]
                break
        else:
            raise ValueError("case-id manifest must contain case_ids or dev_case_ids")
    else:
        raise ValueError("case-id manifest must be a JSON list or object")

    normalized: list[str] = []
    for value in values:
        case_id = str(value)
        if case_id.startswith("bird-"):
            case_id = case_id[5:]
        normalized.append(case_id)
    return normalized
