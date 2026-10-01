from __future__ import annotations

import json
import re
import urllib.request

MODEL = "qwen3.5:4b"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

CASES = [
    {
        "id": "325",
        "kind": "wrong",
        "question": "How many carcinogenic molecules that consisted of Nitrogen?",
        "evidence": "nitrogen refers to element = 'n'; label = '+' mean molecules are carcinogenic;",
        "schema": "atom(atom_id, molecule_id, element); molecule(molecule_id, label); atom.molecule_id = molecule.molecule_id",
        "candidate": "SELECT COUNT(*) FROM atom WHERE element = 'n' AND EXISTS (SELECT 1 FROM molecule WHERE molecule_id = atom.molecule_id AND label = '+');",
        "gold": "SELECT COUNT(DISTINCT T1.molecule_id) FROM molecule AS T1 INNER JOIN atom AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.element = 'n' AND T1.label = '+'",
    },
    {
        "id": "476",
        "kind": "wrong",
        "question": "Please list the name of the cards in the set Coldsnap with the highest converted mana cost.",
        "evidence": "card set Coldsnap refers to name = 'Coldsnap'",
        "schema": "cards(name, asciiName, convertedManaCost, setCode, printings); sets(code, name); cards.setCode = sets.code",
        "candidate": "SELECT DISTINCT c.asciiName FROM cards c INNER JOIN sets s ON c.printings LIKE '%' || s.code || '%' WHERE s.name = 'Coldsnap' ORDER BY c.convertedManaCost DESC;",
        "gold": "SELECT T1.name FROM cards AS T1 INNER JOIN sets AS T2 ON T2.code = T1.setCode WHERE T2.name = 'Coldsnap' AND T1.convertedManaCost = (SELECT MAX(convertedManaCost) FROM cards AS T1 INNER JOIN sets AS T2 ON T2.code = T1.setCode WHERE T2.name = 'Coldsnap')",
    },
    {
        "id": "790",
        "kind": "wrong",
        "question": "Calculate the difference between Emil Blonsky's weight and Charles Chandler's weight.",
        "evidence": "difference = Emil Blonsky weight minus Charles Chandler weight; both names refer to superhero.full_name",
        "schema": "superhero(id, superhero_name, full_name, weight_kg)",
        "candidate": "SELECT s1.weight_kg - s2.weight_kg FROM superhero s1 JOIN superhero s2 ON s1.id = s2.id AND LOWER(s1.full_name) = 'Emil Blonsky' AND LOWER(s2.full_name) = 'Charles Chandler';",
        "gold": "SELECT (SELECT weight_kg FROM superhero WHERE full_name LIKE 'Emil Blonsky') - (SELECT weight_kg FROM superhero WHERE full_name LIKE 'Charles Chandler') AS CALCULATE",
    },
    {
        "id": "300",
        "kind": "wrong",
        "question": "What atoms comprise TR186?",
        "evidence": "TR186 is a molecule id",
        "schema": "atom(atom_id, molecule_id, element)",
        "candidate": "SELECT DISTINCT element FROM atom WHERE molecule_id = 186;",
        "gold": "SELECT T.atom_id FROM atom AS T WHERE T.molecule_id = 'TR186'",
    },
    {
        "id": "1130",
        "kind": "wrong",
        "question": "What are the short name of team who played safe while creating chance of passing?",
        "evidence": "played safe while creating chance of passing refers to chanceCreationPassingClass = 'Safe'; short name of team refers to team_short_name",
        "schema": "Team(team_api_id, team_short_name); Team_Attributes(team_api_id, chanceCreationPassing, chanceCreationPassingClass); Team.team_api_id = Team_Attributes.team_api_id",
        "candidate": "SELECT Team_Attributes.chanceCreationPassing, Team_Attributes.chanceCreationPassingClass FROM Team_Attributes WHERE Team_Attributes.chanceCreationPassingClass = 'Safe' AND Team_Attributes.chanceCreationPassing IS NOT NULL;",
        "gold": "SELECT DISTINCT t1.team_short_name FROM Team AS t1 INNER JOIN Team_Attributes AS t2 ON t1.team_api_id = t2.team_api_id WHERE t2.chanceCreationPassingClass = 'Safe'",
    },
    {
        "id": "622",
        "kind": "control",
        "question": "State the name of badge that the user whose display name is 'Sharpie' obtained.",
        "evidence": "'Sharpie' is the DisplayName of user; user refers to UserId",
        "schema": "users(Id, DisplayName); badges(UserId, Name); users.Id = badges.UserId",
        "candidate": "SELECT Name FROM badges WHERE UserId = (SELECT Id FROM users WHERE DisplayName = 'Sharpie');",
        "gold": "SELECT T2.Name FROM users AS T1 INNER JOIN badges AS T2 ON T1.Id = T2.UserId WHERE T1.DisplayName = 'Sharpie'",
    },
    {
        "id": "838",
        "kind": "control",
        "question": "Provide the full name of the superhero named Alien.",
        "evidence": "",
        "schema": "superhero(superhero_name, full_name)",
        "candidate": "SELECT full_name FROM superhero WHERE superhero_name = 'Alien';",
        "gold": "SELECT full_name FROM superhero WHERE superhero_name = 'Alien'",
    },
    {
        "id": "455",
        "kind": "control",
        "question": "Which of the cards that are a promotional painting have multiple faces on the same card? Please list their names.",
        "evidence": "promotional painting refers to isPromo = 1; multiple faces refers to side being present",
        "schema": "cards(name, isPromo, side)",
        "candidate": "SELECT name FROM cards WHERE isPromo = 1 AND side IS NOT NULL;",
        "gold": "SELECT name FROM cards WHERE isPromo = 1 AND side IS NOT NULL",
    },
]


def strip_sql(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def call_ollama(case: dict[str, str]) -> str:
    user = f"""Question:\n{case['question']}\n\nEvidence:\n{case['evidence'] or '(none)'}\n\nRelevant schema:\n{case['schema']}\n\nCandidate SQL:\n{case['candidate']}\n\nReview whether the SQL actually answers the question. Check requested output, entity/granularity, joins, filters, aggregation, ordering/top-k, and literal values. If it is semantically correct, return it unchanged. If it is wrong, return the smallest corrected SQLite query. Return SQL only."""
    payload = {
        "model": MODEL,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 220},
        "messages": [
            {
                "role": "system",
                "content": "You review one SQLite query for semantic correctness. Return exactly one read-only SELECT or CTE. No markdown or explanation.",
            },
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
    return strip_sql(str(body["message"]["content"]))


def compact(sql: str) -> str:
    return " ".join(sql.lower().rstrip(";").split())


def main() -> None:
    results = []
    for case in CASES:
        revised = call_ollama(case)
        changed = compact(revised) != compact(case["candidate"])
        result = {
            "id": case["id"],
            "kind": case["kind"],
            "changed": changed,
            "candidate": case["candidate"],
            "revised": revised,
            "gold": case["gold"],
        }
        results.append(result)
        print(f"\n=== case {case['id']} ({case['kind']}) changed={changed} ===")
        print("candidate:", case["candidate"])
        print("revised:  ", revised)
        print("gold:     ", case["gold"])

    wrong_changed = sum(r["kind"] == "wrong" and r["changed"] for r in results)
    controls_changed = sum(r["kind"] == "control" and r["changed"] for r in results)
    print(f"\nSUMMARY wrong_changed={wrong_changed}/5 controls_changed={controls_changed}/3")
    with open("semantic-revision-results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
