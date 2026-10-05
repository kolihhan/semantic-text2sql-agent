from semantic_sql.direct import generate_direct_sql
from semantic_sql.semantic_sketch import SemanticSketch


class CaptureProvider:
    def __init__(self) -> None:
        self.system = ""
        self.user = ""

    def complete_text(self, *, system: str, user: str) -> str:
        self.system = system
        self.user = user
        return "```sql\nSELECT COUNT(*) FROM orders\n```"


def test_direct_generation_uses_question_and_supplied_schema_without_gold():
    provider = CaptureProvider()
    candidate = generate_direct_sql(
        "How many orders are there?",
        "TABLE orders(id INTEGER, total_amount REAL)",
        provider,
    )

    assert candidate.sql == "SELECT COUNT(*) FROM orders"
    assert "How many orders" in provider.user
    assert "TABLE orders" in provider.user
    assert "Semantic analysis" not in provider.user
    assert "gold" not in (provider.system + provider.user).lower()
    assert "reference" not in (provider.system + provider.user).lower()


def test_direct_generation_includes_supplied_semantic_sketch():
    provider = CaptureProvider()
    sketch = SemanticSketch(
        outputs=("constructor name",),
        aggregation="sum points",
        grain="one row per constructor",
        group_by=("constructor",),
        ordering="total points descending",
        limit="1",
    )

    generate_direct_sql(
        "Which constructor scored the most points?",
        "constructors(id, name)\nresults(constructorId, points)",
        provider,
        semantic_sketch=sketch,
    )

    assert "Semantic analysis (must follow):" in provider.user
    assert sketch.as_prompt() in provider.user
