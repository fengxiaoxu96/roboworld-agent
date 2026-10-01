"""全局测试配置与环境变量注入。"""

import os

import pytest

# 在模块导入阶段直接关闭 LangSmith 追踪，避免测试过程向假 API 发送心跳请求
os.environ["LANGCHAIN_TRACING_V2"] = "false"


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
