from semantic_sql.semantic_sketch import SemanticSketch, analyze_semantics


class StaticProvider:
    def __init__(self, response: str) -> None:
        self.response = response
        self.system = ""
        self.user = ""
        self.calls = 0

    def complete_text(self, *, system: str, user: str) -> str:
        self.calls += 1
        self.system = system
        self.user = user
        return self.response


def test_analyze_semantics_parses_grounded_intent_without_sql():
    provider = StaticProvider(
        '{"outputs":["constructor name"],"filters":["Monaco GP","1980 to 2010"],'
        '"relations":["constructor scored race points"],"aggregation":"sum points",'
        '"grain":"one row per constructor","group_by":["constructor"],'
        '"ordering":"total points descending","limit":"1"}'
    )

    sketch = analyze_semantics(
        "Which constructor scored the most points at Monaco GP between 1980 and 2010?",
        "constructors(constructorId, name)\nresults(raceId, constructorId, points)",
        provider,
        external_evidence="Monaco GP identifies the Monaco race.",
    )

    assert sketch == SemanticSketch(
        outputs=("constructor name",),
        filters=("Monaco GP", "1980 to 2010"),
        relations=("constructor scored race points",),
        aggregation="sum points",
        grain="one row per constructor",
        group_by=("constructor",),
        ordering="total points descending",
        limit="1",
    )
    assert provider.calls == 1
    assert "Which constructor" in provider.user
    assert "constructors(" in provider.user
    assert "Monaco GP identifies" in provider.user
    assert "gold" not in (provider.system + provider.user).lower()
    assert "sql" in provider.system.lower()


def test_analyze_semantics_fails_closed_on_invalid_response():
    provider = StaticProvider("I think you should GROUP BY constructor")

    sketch = analyze_semantics("question", "schema", provider)

    assert sketch == SemanticSketch()
    assert provider.calls == 1
