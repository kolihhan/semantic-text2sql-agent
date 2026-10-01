from __future__ import annotations

import json
import re
import urllib.request

from semantic_revision_spike import CASES, MODEL, OLLAMA_URL


def call_ollama(*, system: str, user: str, num_predict: int) -> str:
    payload = {
        "model": MODEL,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": num_predict},
        "messages": [
            {"role": "system", "content": system},
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
    return str(body["message"]["content"]).strip()


def strip_sql(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def main() -> None:
    results = []
    for case in CASES:
        plan_user = f"""Question:\n{case['question']}\n\nEvidence:\n{case['evidence'] or '(none)'}\n\nRelevant schema:\n{case['schema']}\n\nDo not write SQL yet. Build a concise semantic plan using exactly these labels:\nOUTPUT:\nENTITIES/LITERALS:\nTABLES:\nJOINS:\nFILTERS:\nAGGREGATION/GRANULARITY:\nORDERING/TOP-K:\n\nBe literal about what the user asks to return, what entity is counted, tie handling, and quoted identifiers/values."""
        plan = call_ollama(
            system="Translate a natural-language database question into a semantic relational plan. Do not produce SQL.",
            user=plan_user,
            num_predict=180,
        )

        sql_user = f"""Question:\n{case['question']}\n\nEvidence:\n{case['evidence'] or '(none)'}\n\nRelevant schema:\n{case['schema']}\n\nSemantic plan:\n{plan}\n\nGenerate the SQLite query that follows the semantic plan exactly. Return SQL only. Preserve all ties when the question asks for the highest/lowest value unless the question explicitly asks for one row."""
        sql = strip_sql(
            call_ollama(
                system="Generate exactly one read-only SQLite SELECT or CTE from the supplied semantic plan. No markdown or explanation.",
                user=sql_user,
                num_predict=220,
            )
        )

        result = {
            "id": case["id"],
            "kind": case["kind"],
            "plan": plan,
            "generated": sql,
            "baseline": case["candidate"],
            "gold": case["gold"],
        }
        results.append(result)
        print(f"\n=== case {case['id']} ({case['kind']}) ===")
        print("plan:", plan)
        print("generated:", sql)
        print("gold:", case["gold"])

    with open("semantic-plan-results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
