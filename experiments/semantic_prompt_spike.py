from __future__ import annotations

import json
import re
import urllib.request

from semantic_revision_spike import CASES, MODEL, OLLAMA_URL

SYSTEM = """Generate exactly one read-only SQLite SELECT or CTE that answers the user's question from the supplied schema and evidence. Return SQL only.

Before writing SQL, reason internally about the requested semantics:
- OUTPUT: return exactly the entity/attribute the question asks for; do not substitute a related attribute.
- GRANULARITY: count or aggregate the requested entity, not lower-level joined rows; use DISTINCT entity keys when joins can duplicate that entity.
- RELATIONSHIPS: use only tables, columns, and join relationships supported by the supplied schema.
- EVIDENCE: treat supplied external evidence as authoritative domain semantics; do not replace it with a guess.
- LITERALS: preserve identifiers and string literals exactly unless the evidence says how to transform them.
- EXTREMA: for highest/lowest, preserve all ties unless the question explicitly asks for a single row.
- DIALECT: use valid SQLite syntax.

Do not reveal reasoning. Do not invent tables or columns."""


def call(case: dict[str, str]) -> str:
    user = f"Question: {case['question']}\n\nSchema context:\n{case['schema']}"
    if case["evidence"]:
        user += f"\n\nExternal evidence:\n{case['evidence']}"
    payload = {
        "model": MODEL,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 220},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        body = json.loads(response.read().decode("utf-8"))
    text = str(body["message"]["content"]).strip()
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def main() -> None:
    results = []
    for case in CASES:
        generated = call(case)
        results.append({
            "id": case["id"],
            "kind": case["kind"],
            "generated": generated,
            "baseline": case["candidate"],
            "gold": case["gold"],
        })
        print(f"\n=== case {case['id']} ({case['kind']}) ===")
        print("generated:", generated)
        print("gold:", case["gold"])

    with open("semantic-prompt-results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
