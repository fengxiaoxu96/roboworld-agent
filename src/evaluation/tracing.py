"""可观测性追踪与 Token 消耗监控模块 (Task 5.3).

主要功能：
1. 全流程接入 LangSmith（与可选 Phoenix），监控所有 LLM / Agent 调用拓扑与节点状态；
2. TokenUsageTracker 回调处理器：精确记录 Prompt Tokens、Completion Tokens、总耗时与调用历史；
3. 支持离线/单测环境下的本地 Token 估算与拓扑记录，确保无 API Key 场景下的全功能可观测性；
4. 提供上下文管理器与配置注入接口，无缝挂载至 LangGraph 与工作流流水线。
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

# 优先定位项目根目录下的 .env 文件进行加载
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH)
else:
    load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_LANGCHAIN_PROJECT = "roboworld-agent"
DEFAULT_LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"


def is_dummy_api_key(key: str | None) -> bool:
    """检查 API Key 是否为空或占位符。"""
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
    """单次 LLM 调用的 Token 与耗时记录。"""

    model_name: str = "unknown"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    prompt_snippet: str = ""
    completion_snippet: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class TokenUsageTracker(BaseCallbackHandler):
    """基于 LangChain 回调协议的 Token 消耗与拓扑统计跟踪器。"""

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
        """针对离线/Mock 环境的简易 Token 估算（约 4 字符/Token）。"""
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
        """捕获 LLM 调用发起。"""
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
        """捕获 LLM 调用完成并计算 Token 与延迟。"""
        key = str(run_id)
        start_t = self._llm_start_times.pop(key, time.perf_counter())
        latency = max(0.0, time.perf_counter() - start_t)
        self.total_latency += latency

        prompt_snippet = self._llm_prompts.pop(key, "")

        # 尝试从 llm_output 获取官方 Token 统计
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
            # 离线环境无 token_usage 字典时，进行高质量估算
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
        """捕获 LLM 调用异常。"""
        self.error_count += 1
        key = str(run_id)
        self._llm_start_times.pop(key, None)
        self._llm_prompts.pop(key, None)
        logger.warning("Tracing 捕获 LLM 调用异常: %s", error)

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
        """获取当前 Tracker 的完整统计摘要。"""
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
        """重置所有统计指标。"""
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
    """可观测性与 Tracing 管理器。"""

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
            # 同步配置到全局环境变量，确保第三方模块与底层子进程能读取一致配置
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
                    # 探测连接与凭据有效性 (若 403 Forbidden / 401 / 网络不可达将抛出异常)
                    client_obj.has_project(self.project_name)

                self.client = client_obj
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGSMITH_TRACING"] = "true"
                logger.info(
                    "LangSmith 追踪已就绪 (project=%s, endpoint=%s)",
                    self.project_name,
                    self.endpoint,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "LangSmith API 连接或鉴权失败 (project=%s, endpoint=%s): %s。"
                    "已自动降级为本地 Tracing 模式，防止后台上传重试阻塞及 SSL 报错。",
                    self.project_name,
                    self.endpoint,
                    exc,
                )
                self.langsmith_enabled = False
                self.client = None
                # 关键：显式关闭全局环境变量，防止 LangChain 内部后台批处理线程抛出 SSLEOFError
                os.environ["LANGCHAIN_TRACING_V2"] = "false"
                os.environ["LANGSMITH_TRACING"] = "false"
        else:
            # 未开启 LangSmith 时显式置为 false，防止残留环境变量触发上传报错
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            os.environ["LANGSMITH_TRACING"] = "false"

    def get_callbacks(self) -> list[BaseCallbackHandler]:
        """获取需要挂载的回调列表。包含本地 TokenTracker 及可选的 LangSmithTracer。"""
        callbacks: list[BaseCallbackHandler] = [self.tracker]

        if self.langsmith_enabled and self.client is not None:
            try:
                from langchain_core.tracers import LangChainTracer

                tracer = LangChainTracer(
                    project_name=self.project_name,
                    client=self.client,
                )
                callbacks.append(tracer)
                logger.debug("成功挂载 LangSmithTracer (project=%s)", self.project_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("初始化 LangChainTracer 失败: %s", exc)

        return callbacks

    def get_langgraph_config(self) -> RunnableConfig:
        """获取 LangGraph 执行配置。"""
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
    """便捷函数：初始化可观测性与 Tracing 管理器。"""
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
    """可观测性上下文管理器。"""
    mgr = setup_tracing(
        project_name=project_name,
        api_key=api_key,
        endpoint=endpoint,
        enable_langsmith=enable_langsmith,
        timeout_ms=timeout_ms,
        validate_connection=validate_connection,
    )
    logger.info("启动可观测性 Trace 会话: %s", run_name)
    try:
        yield mgr
    finally:
        summary = mgr.tracker.get_summary()
        logger.info(
            "Trace 会话 [%s] 结束: Calls=%d, Total Tokens=%d, Latency=%.2fs",
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
