from fastapi.testclient import TestClient

from semantic_sql.api import create_app
from semantic_sql.contracts import AgentResult, SQLCandidate, VerificationResult


class FakeService:
    def __init__(self):
        self.calls = []

    def ask(self, question: str, *, evidence: str | None = None):
        self.calls.append((question, evidence))
        return AgentResult(
            status="ok",
            question=question,
            candidate=SQLCandidate("SELECT 1"),
            verification=VerificationResult(ok=True),
            rows=((1,),),
            columns=("value",),
        )


def test_health_and_query_delegate_to_service():
    service = FakeService()
    client = TestClient(create_app(service=service))

    assert client.get("/health").json() == {"ok": True, "service_loaded": True}

    response = client.post(
        "/query",
        json={"question": "return one", "evidence": "demo evidence"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["candidate"]["sql"] == "SELECT 1"
    assert body["rows"] == [[1]]
    assert service.calls == [("return one", "demo evidence")]


def test_portfolio_demo_page_exposes_reliability_flow():
    client = TestClient(create_app(service=FakeService()))
    response = client.get("/")
    assert response.status_code == 200
    assert "Semantic Text-to-SQL" in response.text
    assert "Generate" in response.text
    assert "Verify" in response.text
    assert "Repair" in response.text
    assert "/query" in response.text


def test_query_rejects_blank_question():
    client = TestClient(create_app(service=FakeService()))
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422
