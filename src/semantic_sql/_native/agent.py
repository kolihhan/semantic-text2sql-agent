# MIT-licensed upstream-derived component; copyright notice in this directory's LICENSE.
"""Bounded text-to-SQL generation, validation, execution, and repair."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .llm import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError
from pydantic import ValidationError

from .conversation import ConversationContext, Turn
from .costs import BudgetExceeded, BudgetGuard
from .db import execute, identifier_hint, is_recoverable_rejection, is_safe, validate
from .llm import ErrorCategory, LLMResult, complete
from .prompts import (
    EMPTY_RESULT_PROMPT,
    REFUSAL_RETRY_PROMPT,
    REPAIR_PROMPT,
    SQL_SCHEMA,
    ResponseType,
    SQLResponse,
)
from .schema import DEFAULT_DB_PATH

CompleteFn = Callable[..., Awaitable[LLMResult]]


@dataclass
class AgentResult:
    question: str
    response_type: ResponseType | None = None
    message: str | None = None
    sql: str | None = None
    rows: list[tuple[Any, ...]] | None = None
    columns: list[str] | None = None
    error: str | None = None
    error_category: ErrorCategory | None = None
    repaired: bool = False
    refusal_retried: bool = False
    empty_retried: bool = False
    truncated: bool = False
    llm_calls: list[LLMResult] = field(default_factory=list)
    t_schema_ms: float = 0.0
    t_llm_ms: float = 0.0
    t_exec_ms: float = 0.0
    t_total_ms: float = 0.0

    @property
    def input_tokens(self) -> int:
        return sum(call.input_tokens for call in self.llm_calls)

    @property
    def cached_tokens(self) -> int:
        return sum(call.cached_tokens for call in self.llm_calls)

    @property
    def output_tokens(self) -> int:
        return sum(call.output_tokens for call in self.llm_calls)

    @property
    def cost_usd(self) -> float:
        return sum(call.cost_usd for call in self.llm_calls)

    @property
    def ok(self) -> bool:
        return self.error is None


class Agent:
    def __init__(
        self,
        model: str,
        conn: sqlite3.Connection,
        budget: BudgetGuard,
        *,
        db_path: str | Path = DEFAULT_DB_PATH,
        max_repairs: int = 1,
        complete_fn: CompleteFn = complete,
        max_tokens: int = 400,
        temperature: float = 0.0,
        request_options: dict[str, Any] | None = None,
        pipeline_repairs: bool = False,
    ):
        self.model = model
        self.conn = conn
        self.budget = budget
        self.db_path = db_path
        self.max_repairs = max_repairs
        self.complete_fn = complete_fn
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_options = request_options
        # Benchmark-mode repairs (off for the interactive CLI): recoverable safety rejects
        # get a "did you mean" repair, one refusal is retried as answerable, an empty result
        # gets one guarded re-check. Forbidden SQL still never retries.
        self.pipeline_repairs = pipeline_repairs

    async def ask(self, question: str, ctx: ConversationContext) -> AgentResult:
        started = time.perf_counter()
        result = AgentResult(question=question)
        messages = ctx.build_messages(question)
        repairs_left = self.max_repairs
        refusal_retry_left = 1 if self.pipeline_repairs else 0

        while True:
            try:
                llm_result = await self.complete_fn(
                    messages,
                    self.model,
                    response_format=SQL_SCHEMA,
                    max_tokens=self.max_tokens,
                    budget=self.budget,
                    request_options=self.request_options,
                    temperature=self.temperature,
                )
                result.llm_calls.append(llm_result)
                result.t_llm_ms += llm_result.latency_ms
            except BudgetExceeded as exc:
                return self._finish(result, str(exc), "budget-exceeded", started)
            except AuthenticationError as exc:
                return self._finish(result, str(exc), "authentication-failed", started)
            except RateLimitError as exc:
                return self._finish(result, str(exc), "rate-limited", started)
            except APITimeoutError as exc:
                return self._finish(result, str(exc), "timed-out", started)
            except APIConnectionError as exc:
                return self._finish(result, str(exc), "network-failed", started)

            try:
                payload = SQLResponse.model_validate_json(llm_result.text)
                if payload.response_type == "query":
                    if not payload.sql or not payload.sql.strip():
                        raise ValueError("query responses require non-empty sql")
                    if payload.message is not None:
                        raise ValueError("query responses must set message to null")
                else:
                    if payload.sql is not None:
                        raise ValueError(
                            f"{payload.response_type} responses must set sql to null"
                        )
                    if not payload.message or not payload.message.strip():
                        raise ValueError(
                            f"{payload.response_type} responses require a non-empty message"
                        )
            except (ValidationError, ValueError) as exc:
                return self._finish(
                    result, f"Structured output failed: {exc}", "structured-output-failed", started
                )

            result.response_type = payload.response_type
            result.message = payload.message
            if payload.response_type != "query":
                if refusal_retry_left:
                    refusal_retry_left -= 1
                    result.refusal_retried = True
                    messages.append({"role": "assistant", "content": llm_result.text})
                    messages.append({"role": "user", "content": REFUSAL_RETRY_PROMPT})
                    continue
                result.t_total_ms = (time.perf_counter() - started) * 1_000
                return result

            sql = payload.sql
            assert sql is not None
            result.sql = sql
            safe, reason = is_safe(sql, db_path=self.db_path)
            if not safe:
                if self.pipeline_repairs and repairs_left and is_recoverable_rejection(reason):
                    repairs_left -= 1
                    result.repaired = True
                    hinted = reason + identifier_hint(reason, self.db_path)
                    messages.append({"role": "assistant", "content": llm_result.text})
                    messages.append(
                        {"role": "user", "content": REPAIR_PROMPT.format(sql=sql, error=hinted)}
                    )
                    continue
                return self._finish(result, reason, "safety-rejected", started)

            valid, error = validate(self.conn, sql)
            if not valid:
                if repairs_left:
                    repairs_left -= 1
                    result.repaired = True
                    messages.append({"role": "assistant", "content": llm_result.text})
                    messages.append(
                        {
                            "role": "user",
                            "content": REPAIR_PROMPT.format(sql=sql, error=error),
                        }
                    )
                    continue
                category: ErrorCategory = (
                    "repair-exhausted" if result.repaired else "validate-failed"
                )
                return self._finish(result, error, category, started)

            exec_started = time.perf_counter()
            try:
                columns, rows, truncated = execute(self.conn, sql)
            except sqlite3.Error as exc:
                result.t_exec_ms += (time.perf_counter() - exec_started) * 1_000
                if repairs_left:
                    repairs_left -= 1
                    result.repaired = True
                    messages.extend(
                        [
                            {"role": "assistant", "content": llm_result.text},
                            {
                                "role": "user",
                                "content": REPAIR_PROMPT.format(sql=sql, error=str(exc)),
                            },
                        ]
                    )
                    continue
                category = "repair-exhausted" if result.repaired else "execute-failed"
                return self._finish(result, str(exc), category, started)
            result.t_exec_ms += (time.perf_counter() - exec_started) * 1_000
            if self.pipeline_repairs and not rows:
                retried = await self._retry_empty(messages, llm_result.text, sql, result)
                if retried is not None:
                    sql, (columns, rows, truncated) = retried
                    result.sql = sql
            result.columns, result.rows, result.truncated = columns, rows, truncated
            ctx.record(Turn(question, sql, len(rows), truncated))
            result.t_total_ms = (time.perf_counter() - started) * 1_000
            return result

    async def _retry_empty(
        self,
        messages: list[dict[str, str]],
        answer: str,
        sql: str,
        result: AgentResult,
    ) -> tuple[str, tuple[list[str], list[tuple[Any, ...]], bool]] | None:
        """One guarded re-check of an empty result.

        The new query replaces the original only if it passes the same safety, validation,
        and execution path *and* returns rows; any failure keeps the original (empty) answer,
        so this can add a call but never turn a delivered answer into an error.
        """
        result.empty_retried = True
        retry_messages = [
            *messages,
            {"role": "assistant", "content": answer},
            {"role": "user", "content": EMPTY_RESULT_PROMPT.format(sql=sql)},
        ]
        try:
            llm_result = await self.complete_fn(
                retry_messages,
                self.model,
                response_format=SQL_SCHEMA,
                max_tokens=self.max_tokens,
                budget=self.budget,
                request_options=self.request_options,
                temperature=self.temperature,
            )
        except (BudgetExceeded, AuthenticationError, RateLimitError, APITimeoutError,
                APIConnectionError):
            return None
        result.llm_calls.append(llm_result)
        result.t_llm_ms += llm_result.latency_ms
        try:
            payload = SQLResponse.model_validate_json(llm_result.text)
        except ValidationError:
            return None
        candidate = (payload.sql or "").strip()
        if payload.response_type != "query" or not candidate or candidate == sql.strip():
            return None
        if not is_safe(candidate, db_path=self.db_path)[0] or not validate(self.conn, candidate)[0]:
            return None
        exec_started = time.perf_counter()
        try:
            outcome = execute(self.conn, candidate)
        except sqlite3.Error:
            return None
        finally:
            result.t_exec_ms += (time.perf_counter() - exec_started) * 1_000
        return (candidate, outcome) if outcome[1] else None

    @staticmethod
    def _finish(
        result: AgentResult,
        error: str,
        category: ErrorCategory,
        started: float,
    ) -> AgentResult:
        result.error = error
        result.error_category = category
        result.t_total_ms = (time.perf_counter() - started) * 1_000
        return result
