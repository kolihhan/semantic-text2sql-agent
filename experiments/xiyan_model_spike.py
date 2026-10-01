from __future__ import annotations

import json
import re
import urllib.request

from semantic_revision_spike import CASES, OLLAMA_URL

MODEL = "hf.co/wanhin/XiYanSQL-QwenCoder-7B-2504-gguf:Q4_K_M"


def call(case: dict[str, str]) -> str:
    prompt = f"""你是一名SQLite专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，并运用SQLite知识生成sql语句回答【用户问题】。
【用户问题】
{case['question']}

【数据库schema】
{case['schema']}

【参考信息】
{case['evidence'] or ''}

【用户问题】
{case['question']}

```sql"""
    payload = {
        "model": MODEL,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 220},
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=900) as response:
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

    with open("xiyan-model-results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
