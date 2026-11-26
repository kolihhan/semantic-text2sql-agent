"""Local model adapter types for the bounded structured SQL pipeline."""
from __future__ import annotations
from dataclasses import dataclass

class APIConnectionError(Exception):
    pass

class APITimeoutError(Exception):
    pass

class AuthenticationError(Exception):
    pass

class RateLimitError(Exception):
    pass

ErrorCategory = str

@dataclass
class LLMResult:
    text: str
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    latency_ms: float
    retry_ms: float
    model: str
    cost_usd: float

async def complete(*args, **kwargs):
    raise RuntimeError("This local pipeline requires a configured model provider.")
