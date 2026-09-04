from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol
import urllib.request


class ModelProvider(Protocol):
    def complete_text(self, *, system: str, user: str) -> str: ...


@dataclass
class OllamaProvider:
    model: str = "qwen3.5:4b"
    base_url: str = "http://localhost:11434"
    timeout_s: float = 60.0

    def _chat(self, *, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return str(body["message"]["content"])

    def complete_text(self, *, system: str, user: str) -> str:
        return self._chat(system=system, user=user)


class DemoProvider:
    """Deterministic provider used by the shipped offline smoke tests.

    It exercises the same service boundaries as an LLM provider but is not a
    performance benchmark and must not be used as portfolio evidence.
    """

    def __init__(self) -> None:
        self.last_question = ""

    def complete_text(self, *, system: str, user: str) -> str:
        lowered = user.lower()
        is_repair = "repair" in system.lower() or "verification issues" in lowered
        question = lowered.split("schema context:", 1)[0]
        if "salary" in question:
            return "SELECT salary FROM orders"
        if "how many" in question or "number of orders" in question:
            if not is_repair:
                return "SELECT SUM(orders.id) FROM orders WHERE orders.created_at >= '2024-01-01' AND orders.created_at < '2025-01-01'"
            return "SELECT COUNT(orders.id) FROM orders WHERE orders.created_at >= '2024-01-01' AND orders.created_at < '2025-01-01'"
        if "which customer" in question and "spent the most" in question:
            return (
                "SELECT customers.name, SUM(orders.total_amount) "
                "FROM orders JOIN customers ON orders.customer_id = customers.id "
                "GROUP BY customers.name ORDER BY SUM(orders.total_amount) DESC LIMIT 1"
            )
        clauses = []
        join = ""
        if "czech republic" in question:
            join = " JOIN customers ON orders.customer_id = customers.id"
            clauses.append("customers.country_code = 'CZE'")
        if "2024" in question:
            clauses.extend(["orders.created_at >= '2024-01-01'", "orders.created_at < '2025-01-01'"])
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        return "SELECT SUM(orders.total_amount) FROM orders" + join + where
