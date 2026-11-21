from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlsplit
import urllib.request


class ModelProvider(Protocol):
    def complete_text(self, *, system: str, user: str) -> str: ...


@dataclass
class OllamaProvider:
    model: str = "qwen3.5:4b"
    base_url: str = "http://localhost:11434"
    timeout_s: float = 60.0
    temperature: float | None = None
    seed: int | None = None
    num_predict: int | None = None

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
        options = {k: v for k, v in (
            ("temperature", self.temperature), ("seed", self.seed),
            ("num_predict", self.num_predict),
        ) if v is not None}
        if options:
            payload["options"] = options
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



@dataclass
class FreeLLMAPIProvider:
    """Opt-in OpenAI-compatible transport via a configured local FreeLLMAPI router.

    Explicit Qwen routing only: do not use 'auto' or a fallback profile for
    comparable research. The response name check cannot prove an Ollama SHA
    match; external outputs MUST be treated as a different inference backend.
    """

    model: str = "qwen3.5-9b"
    expected_response_model: str = "Qwen3.5-9B"
    expected_response_platform: str = "ovh"
    base_url: str = "http://localhost:3001/v1"
    api_key: str | None = field(default=None, repr=False)
    timeout_s: float = 90.0
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int = 512

    def complete_text(self, *, system: str, user: str) -> str:
        if not self.model.strip() or self.model.lower().startswith("auto"):
            raise ValueError("FreeLLMAPI benchmark transport requires an explicit model, not auto routing")
        key = self.api_key or os.environ.get("FREELLMAPI_API_KEY")
        if not key:
            raise ValueError("FREELLMAPI_API_KEY or api_key is required")
        uri = urlsplit(self.base_url)
        if uri.scheme != "https" and not (
            uri.scheme == "http" and uri.hostname in {"localhost", "127.0.0.1", "::1"}
        ):
            raise ValueError("FreeLLMAPI requires HTTPS unless serving on localhost")
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": self.temperature,
                "seed": self.seed,
                "reasoning_effort": "none",
                "max_tokens": self.max_tokens,
                "stream": False,
            }).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + key,
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("model") != self.expected_response_model:
            raise ValueError("FreeLLMAPI model identity mismatch; refuse unknown/substituted model")
        routed = payload.get("_routed_via")
        if not isinstance(routed, dict) or routed.get("platform") != self.expected_response_platform:
            raise ValueError("FreeLLMAPI routed provider mismatch; refuse unknown/substituted backend")
        if routed.get("model") != self.expected_response_model:
            raise ValueError("FreeLLMAPI routed provider model mismatch")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("FreeLLMAPI incomplete completion: missing choices")
        answer = choices[0]
        if answer.get("finish_reason") != "stop":
            raise ValueError("FreeLLMAPI incomplete completion: finish_reason is not stop")
        content = answer.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("FreeLLMAPI incomplete completion: missing SQL text")
        return content


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
