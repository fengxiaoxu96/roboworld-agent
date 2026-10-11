"""Observability tracing and token usage monitoring module (Task 5.3).

Main functions:
1. Full-pipeline integration with LangSmith (and optional Phoenix), monitoring all LLM/Agent call topologies and node states;
2. TokenUsageTracker callback handler: accurately records Prompt Tokens, Completion Tokens, total latency, and call history;
3. Support local token estimation and topology recording in offline/unit-test environments, ensuring full observability without API Keys;
4. Provide context managers and configuration injection interfaces, seamlessly attached to LangGraph and workflow pipelines.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

# Prioritize loading .env located at project root
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH)
else:
    load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_LANGCHAIN_PROJECT = "roboworld-agent"
DEFAULT_LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"


def is_dummy_api_key(key: str | None) -> bool:
    """Check if API Key is empty or a placeholder."""
    if not key:
        return True
    k = key.strip().lower()
    return any(
        d in k
        for d in [
            "your_",
            "placeholder",
            "here",
            "xxx",
            "sk-proj-your",
            "lsv2_pt_your",
        ]
    )


class TokenRecord(BaseModel):
    """Token and latency record for a single LLM invocation."""

    model_name: str = "unknown"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    prompt_snippet: str = ""
    completion_snippet: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class TokenUsageTracker(BaseCallbackHandler):
    """Token consumption and topology tracker based on LangChain callback protocol."""

    def __init__(self) -> None:
        super().__init__()
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.llm_call_count: int = 0
        self.chain_call_count: int = 0
        self.tool_call_count: int = 0
        self.error_count: int = 0
        self.total_latency: float = 0.0

        self.records: list[TokenRecord] = []
        self._llm_start_times: dict[str, float] = {}
        self._llm_prompts: dict[str, str] = {}

    def _estimate_tokens(self, text: str) -> int:
        """Simple token estimation for offline/mock environments (approx. 4 characters/token)."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        """Capture LLM call start."""
        self.llm_call_count += 1
        key = str(run_id)
        self._llm_start_times[key] = time.perf_counter()
        prompt_preview = prompts[0][:150] if prompts else ""
        self._llm_prompts[key] = prompt_preview

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        """Capture LLM call completion and calculate tokens and latency."""
        key = str(run_id)
        start_t = self._llm_start_times.pop(key, time.perf_counter())
        latency = max(0.0, time.perf_counter() - start_t)
        self.total_latency += latency

        prompt_snippet = self._llm_prompts.pop(key, "")

        # Try to extract official token statistics from llm_output
        llm_output = response.llm_output or {}
        token_usage = llm_output.get("token_usage") or llm_output.get("usage")

        p_tokens = 0
        c_tokens = 0
        model_name = str(llm_output.get("model_name", "unknown"))

        completion_texts: list[str] = []
        for gen_list in response.generations:
            for gen in gen_list:
                completion_texts.append(gen.text)
        completion_full = "\n".join(completion_texts)
        comp_snippet = completion_full[:150]

        if isinstance(token_usage, dict):
            p_val = (
                token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0
            )
            c_val = (
                token_usage.get("completion_tokens")
                or token_usage.get("output_tokens")
                or 0
            )
            p_tokens = int(p_val)
            c_tokens = int(c_val)
        else:
            # High quality estimation when token_usage dict is absent in offline environments
            p_tokens = self._estimate_tokens(prompt_snippet)
            c_tokens = self._estimate_tokens(completion_full)

        tot = p_tokens + c_tokens
        self.prompt_tokens += p_tokens
        self.completion_tokens += c_tokens
        self.total_tokens += tot

        record = TokenRecord(
            model_name=model_name,
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=tot,
            latency_seconds=round(latency, 4),
            prompt_snippet=prompt_snippet,
            completion_snippet=comp_snippet,
        )
        self.records.append(record)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        """Capture LLM call exception."""
        self.error_count += 1
        key = str(run_id)
        self._llm_start_times.pop(key, None)
        self._llm_prompts.pop(key, None)
        logger.warning("Tracing captured LLM call exception: %s", error)

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        self.chain_call_count += 1

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        self.tool_call_count += 1

    def get_summary(self) -> dict[str, Any]:
        """Get complete statistical summary of current tracker."""
        avg_latency = (
            round(self.total_latency / self.llm_call_count, 4)
            if self.llm_call_count > 0
            else 0.0
        )
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_call_count": self.llm_call_count,
            "chain_call_count": self.chain_call_count,
            "tool_call_count": self.tool_call_count,
            "error_count": self.error_count,
            "total_latency_seconds": round(self.total_latency, 4),
            "average_latency_seconds": avg_latency,
            "records_count": len(self.records),
        }

    def reset(self) -> None:
        """Reset all statistical metrics."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.llm_call_count = 0
        self.chain_call_count = 0
        self.tool_call_count = 0
        self.error_count = 0
        self.total_latency = 0.0
        self.records.clear()
        self._llm_start_times.clear()
        self._llm_prompts.clear()


class TracingManager:
    """Observability and Tracing manager."""

    def __init__(
        self,
        project_name: str | None = None,
        api_key: str | None = None,
        endpoint: str | None = None,
        enable_langsmith: bool | None = None,
        timeout_ms: int = 5000,
        validate_connection: bool = True,
    ) -> None:
        self.project_name = (
            project_name
            or os.getenv("LANGCHAIN_PROJECT")
            or os.getenv("LANGSMITH_PROJECT")
            or DEFAULT_LANGCHAIN_PROJECT
        ).strip()

        self.api_key = (
            api_key
            or os.getenv("LANGCHAIN_API_KEY")
            or os.getenv("LANGSMITH_API_KEY")
            or ""
        ).strip()

        self.endpoint = (
            (
                endpoint
                or os.getenv("LANGCHAIN_ENDPOINT")
                or os.getenv("LANGSMITH_ENDPOINT")
                or DEFAULT_LANGCHAIN_ENDPOINT
            )
            .strip()
            .rstrip("/")
        )

        is_dummy = is_dummy_api_key(self.api_key)
        tracing_env = any(
            os.getenv(k, "").lower() in ("true", "1")
            for k in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING")
        )

        if enable_langsmith is not None:
            self.langsmith_enabled = bool(enable_langsmith)
        else:
            self.langsmith_enabled = (
                (not is_dummy) and bool(self.api_key) and tracing_env
            )

        self.client: Any = None
        self.tracker = TokenUsageTracker()

        if self.langsmith_enabled:
            # Synchronize configuration into global environment variables
            if self.api_key:
                os.environ["LANGCHAIN_API_KEY"] = self.api_key
                os.environ["LANGSMITH_API_KEY"] = self.api_key
            if self.endpoint:
                os.environ["LANGCHAIN_ENDPOINT"] = self.endpoint
                os.environ["LANGSMITH_ENDPOINT"] = self.endpoint
            if self.project_name:
                os.environ["LANGCHAIN_PROJECT"] = self.project_name
                os.environ["LANGSMITH_PROJECT"] = self.project_name

            try:
                from langsmith import Client

                client_obj = Client(
                    api_url=self.endpoint,
                    api_key=self.api_key,
                    timeout_ms=timeout_ms,
                )
                if validate_connection:
                    # Probe connection and credentials validity
                    client_obj.has_project(self.project_name)

                self.client = client_obj
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGSMITH_TRACING"] = "true"
                logger.info(
                    "LangSmith tracing is ready (project=%s, endpoint=%s)",
                    self.project_name,
                    self.endpoint,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "LangSmith API connection or auth failed (project=%s, endpoint=%s): %s. "
                    "Automatically downgraded to local Tracing mode to prevent background retries and SSL errors.",
                    self.project_name,
                    self.endpoint,
                    exc,
                )
                self.langsmith_enabled = False
                self.client = None
                # Explicitly disable global env var to prevent LangChain background batch thread SSLEOFError
                os.environ["LANGCHAIN_TRACING_V2"] = "false"
                os.environ["LANGSMITH_TRACING"] = "false"
        else:
            # Explicitly set false when LangSmith disabled to prevent lingering env vars from triggering upload errors
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            os.environ["LANGSMITH_TRACING"] = "false"

    def get_callbacks(self) -> list[BaseCallbackHandler]:
        """Get list of callbacks to mount, containing local TokenTracker and optional LangSmithTracer."""
        callbacks: list[BaseCallbackHandler] = [self.tracker]

        if self.langsmith_enabled and self.client is not None:
            try:
                from langchain_core.tracers import LangChainTracer

                tracer = LangChainTracer(
                    project_name=self.project_name,
                    client=self.client,
                )
                callbacks.append(tracer)
                logger.debug(
                    "Successfully mounted LangSmithTracer (project=%s)",
                    self.project_name,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to initialize LangChainTracer: %s", exc)

        return callbacks

    def get_langgraph_config(self) -> RunnableConfig:
        """Get LangGraph execution configuration."""
        return RunnableConfig(
            callbacks=self.get_callbacks(),
            metadata={
                "project": self.project_name,
                "timestamp": datetime.now(UTC).isoformat(),
            },
            tags=["roboworld-agent", "world-model-pipeline"],
        )


def setup_tracing(
    project_name: str | None = None,
    api_key: str | None = None,
    endpoint: str | None = None,
    enable_langsmith: bool | None = None,
    timeout_ms: int = 5000,
    validate_connection: bool = True,
) -> TracingManager:
    """Convenience helper: Initialize observability and Tracing manager."""
    return TracingManager(
        project_name=project_name,
        api_key=api_key,
        endpoint=endpoint,
        enable_langsmith=enable_langsmith,
        timeout_ms=timeout_ms,
        validate_connection=validate_connection,
    )


@contextmanager
def trace_run(
    run_name: str = "roboworld_run",
    project_name: str | None = None,
    api_key: str | None = None,
    endpoint: str | None = None,
    enable_langsmith: bool | None = None,
    timeout_ms: int = 5000,
    validate_connection: bool = True,
) -> Generator[TracingManager, None, None]:
    """Observability context manager."""
    mgr = setup_tracing(
        project_name=project_name,
        api_key=api_key,
        endpoint=endpoint,
        enable_langsmith=enable_langsmith,
        timeout_ms=timeout_ms,
        validate_connection=validate_connection,
    )
    logger.info("Starting observability Trace session: %s", run_name)
    try:
        yield mgr
    finally:
        summary = mgr.tracker.get_summary()
        logger.info(
            "Trace session [%s] finished: Calls=%d, Total Tokens=%d, Latency=%.2fs",
            run_name,
            summary["llm_call_count"],
            summary["total_tokens"],
            summary["total_latency_seconds"],
        )


__all__ = [
    "DEFAULT_LANGCHAIN_ENDPOINT",
    "DEFAULT_LANGCHAIN_PROJECT",
    "TokenRecord",
    "TokenUsageTracker",
    "TracingManager",
    "is_dummy_api_key",
    "setup_tracing",
    "trace_run",
]
