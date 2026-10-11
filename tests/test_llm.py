"""单测：通用 LLM 接口规范、提供商适配与 Classifier/RAG 接入 (feature/add-llm)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_openai import ChatOpenAI

from src.agents.classifier_agent import ClassifierAgent, FilterDecision
from src.app import RAGSynthesizer
from src.core.llm import (
    check_api_credential_match,
    get_llm,
    get_llm_info,
    is_truthy,
    is_valid_api_key,
    resolve_llm_config,
)
from src.core.schema import PaperInsight, RawPaper
from src.storage.vector_store import VectorSearchResult, VectorStore
from src.workflows.graph import WorkflowEngine, build_workflow_graph


def _make_paper(
    arxiv_id: str = "2609.00001",
    title: str = "Action-conditioned Flow Matching World Model for Robotics",
    abstract: str = "We present an action-conditioned continuous flow matching world model for robot manipulation.",
) -> RawPaper:
    return RawPaper(
        arxiv_id=arxiv_id,
        title=title,
        authors=["Alice", "Bob"],
        abstract=abstract,
        categories=["cs.RO", "cs.AI"],
        published_date=datetime(2026, 9, 20, tzinfo=UTC),
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
    )


# =========================================================================
# 1. API Key 有效性与配置解析测试
# =========================================================================


def test_is_truthy() -> None:
    """测试布尔环境变量解析。"""
    assert is_truthy("true")
    assert is_truthy("TRUE")
    assert is_truthy("1")
    assert is_truthy("yes")
    assert is_truthy("on")
    assert is_truthy("enable")

    assert not is_truthy("false")
    assert not is_truthy("0")
    assert not is_truthy("no")
    assert not is_truthy("")
    assert not is_truthy(None)


def test_is_valid_api_key() -> None:
    """测试 API Key 占位符识别与有效性过滤。"""
    assert not is_valid_api_key(None)
    assert not is_valid_api_key("")
    assert not is_valid_api_key("   ")
    assert not is_valid_api_key("your_openai_api_key_here")
    assert not is_valid_api_key("sk-placeholder")
    assert not is_valid_api_key("xxx")
    assert not is_valid_api_key("none")

    assert is_valid_api_key("sk-proj-1234567890abcdef")
    assert is_valid_api_key("AIzaSyDummyKey123")
    assert is_valid_api_key("ollama")
    assert is_valid_api_key("any-custom-key-format")


def test_check_api_credential_match() -> None:
    """测试 API Key 非空/非占位符校验，以及服务商 provider 与端点 URL 的匹配校验。"""
    # 1. 正常匹配用例（不限制 API Key 格式，只要非空非占位符即可）
    ok, _ = check_api_credential_match(
        "sk-proj-openai-test-key-12345678",
        provider="openai",
        base_url="https://api.openai.com/v1",
    )
    assert ok is True

    # 允许不同厂商格式的 Key（例如 Gemini 传入任意非空非占位符 Key）
    ok, _ = check_api_credential_match(
        "custom-key-format-123",
        provider="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    assert ok is True

    ok, _ = check_api_credential_match(
        "any-valid-key-format-456",
        provider="deepseek",
        base_url="https://api.deepseek.com/v1",
    )
    assert ok is True

    ok, _ = check_api_credential_match(
        "ollama",
        provider="ollama",
        base_url="http://localhost:11434/v1",
    )
    assert ok is True

    # 自定义反代端点（不限制自定义域名，无冲突即匹配）
    ok, _ = check_api_credential_match(
        "any-token",
        provider="openai",
        base_url="https://my-custom-proxy.internal/v1",
    )
    assert ok is True

    # 2. 空值与占位符过滤
    ok_empty, reason = check_api_credential_match("", provider="openai")
    assert ok_empty is False
    assert "为空" in reason

    ok_none, reason = check_api_credential_match(None, provider="openai")
    assert ok_none is False
    assert "为空" in reason

    ok_placeholder, reason = check_api_credential_match(
        "your_openai_api_key_here", provider="openai"
    )
    assert ok_placeholder is False
    assert "占位符" in reason

    # 3. 服务商与 URL 冲突校验（如 DeepSeek 搭配 OpenAI URL，或 Gemini 搭配 OpenAI URL）
    ok_conflict_ds, reason = check_api_credential_match(
        "any-valid-key",
        provider="deepseek",
        base_url="https://api.openai.com/v1",
    )
    assert ok_conflict_ds is False
    assert "冲突" in reason

    ok_conflict_gemini, reason = check_api_credential_match(
        "any-valid-key",
        provider="gemini",
        base_url="https://api.openai.com/v1",
    )
    assert ok_conflict_gemini is False
    assert "冲突" in reason

    ok_conflict_openai, reason = check_api_credential_match(
        "any-valid-key",
        provider="openai",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    assert ok_conflict_openai is False
    assert "冲突" in reason

    # 4. is_valid_api_key 校验
    assert is_valid_api_key(
        "any-valid-key",
        provider="openai",
        base_url="https://api.openai.com/v1",
    )
    assert not is_valid_api_key(
        "your_openai_api_key_here",
        provider="openai",
        base_url="https://api.openai.com/v1",
    )
    assert not is_valid_api_key(
        "any-valid-key",
        provider="deepseek",
        base_url="https://api.openai.com/v1",
    )


def test_resolve_llm_config_enabled_switch() -> None:
    """测试 LLM_ENABLED 总开关在不同场景下的解析。"""
    # 场景 1：云端部署 / GitHub Actions 默认环境（无 LLM 变量，默认关闭纯启发式）
    with patch.dict("os.environ", {}, clear=True):
        cfg_deploy = resolve_llm_config()
        assert cfg_deploy.enabled is False

    # 场景 2：显式配置 LLM_ENABLED=false
    with patch.dict("os.environ", {"LLM_ENABLED": "false"}, clear=True):
        cfg_disabled = resolve_llm_config()
        assert cfg_disabled.enabled is False

    # 场景 3：显式配置 LLM_ENABLED=true
    with patch.dict("os.environ", {"LLM_ENABLED": "true"}, clear=True):
        cfg_enabled = resolve_llm_config()
        assert cfg_enabled.enabled is True

    # 场景 4：显式参数覆盖环境变量
    with patch.dict("os.environ", {"LLM_ENABLED": "false"}, clear=True):
        cfg_override = resolve_llm_config(enabled=True)
        assert cfg_override.enabled is True


def test_resolve_llm_config_presets() -> None:
    """测试常见提供商预设（DeepSeek, Gemini, Qwen, Ollama, SiliconFlow）。"""
    # 1. DeepSeek 预设
    cfg_ds = resolve_llm_config(provider="deepseek")
    assert cfg_ds.base_url == "https://api.deepseek.com/v1"
    assert cfg_ds.model == "deepseek-chat"

    # 2. Gemini 预设 (OpenAI 兼容端点)
    cfg_gemini = resolve_llm_config(provider="gemini")
    assert "generativelanguage.googleapis.com" in cfg_gemini.base_url
    assert cfg_gemini.model == "gemini-2.0-flash"

    # 3. Qwen 通义千问预设
    cfg_qwen = resolve_llm_config(provider="qwen")
    assert "dashscope.aliyuncs.com" in cfg_qwen.base_url
    assert cfg_qwen.model == "qwen-plus"

    # 4. 本地 Ollama 预设 (自动为本地无密钥环境填充占位密钥)
    cfg_ollama = resolve_llm_config(provider="ollama")
    assert cfg_ollama.base_url == "http://localhost:11434/v1"
    assert cfg_ollama.model == "qwen2.5:7b"
    assert cfg_ollama.api_key == "ollama"

    # 5. SiliconFlow 预设
    cfg_sf = resolve_llm_config(provider="siliconflow")
    assert cfg_sf.base_url == "https://api.siliconflow.cn/v1"
    assert "DeepSeek-V3" in cfg_sf.model


def test_resolve_llm_config_custom_override() -> None:
    """测试显式参数优先覆盖预设与默认值。"""
    cfg = resolve_llm_config(
        provider="deepseek",
        model="custom-deepseek-r1",
        base_url="https://custom-proxy.internal/v1",
        api_key="sk-custom-secret-key-123",
        temperature=0.7,
        timeout=15.0,
        max_retries=5,
    )
    assert cfg.model == "custom-deepseek-r1"
    assert cfg.base_url == "https://custom-proxy.internal/v1"
    assert cfg.api_key == "sk-custom-secret-key-123"
    assert cfg.temperature == 0.7
    assert cfg.timeout == 15.0
    assert cfg.max_retries == 5
    assert cfg.enabled is True  # 显式传入 api_key 自动激活


def test_resolve_llm_config_from_env() -> None:
    """测试从自定义环境变量提取通用配置。"""
    env_mock = {
        "LLM_ENABLED": "true",
        "LLM_API_KEY": "sk-env-key-99999",
        "LLM_BASE_URL": "https://api.custom-llm.com/v1",
        "LLM_MODEL": "custom-v1",
        "LLM_TEMPERATURE": "0.3",
    }
    with patch.dict("os.environ", env_mock, clear=True):
        cfg = resolve_llm_config()
        assert cfg.enabled is True
        assert cfg.api_key == "sk-env-key-99999"
        assert cfg.base_url == "https://api.custom-llm.com/v1"
        assert cfg.model == "custom-v1"
        assert cfg.temperature == 0.3


def test_resolve_llm_config_env_custom_url_with_provider() -> None:
    """测试当环境变量指定 provider 但同时指定了自定义 base_url 与 model 时，优先使用环境指定的自定义值。"""
    env_mock = {
        "LLM_ENABLED": "true",
        "LLM_PROVIDER": "openai",
        "LLM_BASE_URL": "https://my-custom-proxy.internal/v1",
        "LLM_MODEL": "gpt-4o",
        "LLM_API_KEY": "sk-proj-proxy-key-12345678",
    }
    with patch.dict("os.environ", env_mock, clear=True):
        cfg = resolve_llm_config()
        assert cfg.enabled is True
        assert cfg.provider == "openai"
        assert cfg.base_url == "https://my-custom-proxy.internal/v1"
        assert cfg.model == "gpt-4o"
        assert cfg.api_key == "sk-proj-proxy-key-12345678"


# =========================================================================
# 2. get_llm 与 get_llm_info 测试
# =========================================================================


def test_get_llm_offline_fallback() -> None:
    """在未启用或无有效 API Key 时，get_llm 优雅返回 None。"""
    # 场景 A: LLM_ENABLED=false (云端部署场景)
    with patch.dict("os.environ", {"LLM_ENABLED": "false"}, clear=True):
        llm = get_llm()
        assert llm is None

        with pytest.raises(ValueError, match="LLM 未启用"):
            get_llm(strict=True)

    # 场景 B: LLM_ENABLED=true 但未配置 API Key (本地自动降级场景)
    with patch.dict("os.environ", {"LLM_ENABLED": "true"}, clear=True):
        llm = get_llm()
        assert llm is None

        with pytest.raises(ValueError, match="校验未通过"):
            get_llm(strict=True)

    # 场景 C: LLM_ENABLED=true 且配了无效占位符
    with patch.dict(
        "os.environ",
        {"LLM_ENABLED": "true", "LLM_API_KEY": "your_openai_api_key_here"},
        clear=True,
    ):
        llm = get_llm()
        assert llm is None

    # 场景 D: LLM_ENABLED=true 但 Provider 与 URL 冲突导致校验失败
    with patch.dict(
        "os.environ",
        {
            "LLM_ENABLED": "true",
            "LLM_PROVIDER": "deepseek",
            "LLM_API_KEY": "valid-api-key-123456",
            "LLM_BASE_URL": "https://api.openai.com/v1",
        },
        clear=True,
    ):
        llm = get_llm()
        assert llm is None

        with pytest.raises(ValueError, match="冲突"):
            get_llm(strict=True)


def test_get_llm_success() -> None:
    """提供有效 Key 且启用时成功构建 ChatOpenAI 实例。"""
    llm = get_llm(
        enabled=True,
        api_key="sk-test-valid-key-123456",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        temperature=0.1,
    )
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "deepseek-chat"
    assert str(llm.openai_api_base).rstrip("/") == "https://api.deepseek.com/v1"
    assert llm.temperature == 0.1


def test_get_llm_info() -> None:
    """测试提取 LLM 运行状态元信息。"""
    # 1. 场景 A: LLM 未启用 (DISABLED)
    with patch.dict("os.environ", {"LLM_ENABLED": "false"}, clear=True):
        info_disabled = get_llm_info(None)
        assert not info_disabled["is_active"]
        assert info_disabled["status_code"] == "DISABLED"
        assert info_disabled["provider"] == "Rule-Based"

    # 2. 场景 B: LLM 开启但 Key 无效导致自动降级 (FALLBACK)
    with patch.dict("os.environ", {"LLM_ENABLED": "true"}, clear=True):
        info_fallback = get_llm_info(None)
        assert not info_fallback["is_active"]
        assert info_fallback["status_code"] == "FALLBACK"
        assert info_fallback["provider"] == "Rule-Based"

    # 3. 场景 C: 已成功接入实例 (ACTIVE)
    llm = get_llm(
        enabled=True,
        api_key="sk-test-valid-key-123456",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
    )
    info_active = get_llm_info(llm)
    assert info_active["is_active"]
    assert info_active["status_code"] == "ACTIVE"
    assert info_active["provider"] == "DEEPSEEK"
    assert info_active["model"] == "deepseek-chat"
    assert "api.deepseek.com" in info_active["base_url"]


# =========================================================================
# 3. ClassifierAgent 接入 LLM 与降级测试
# =========================================================================


def test_classifier_agent_auto_init() -> None:
    """测试 ClassifierAgent 默认自动调用 get_llm()。"""
    # 环境变量未激活 -> llm 为 None，使用规则
    with patch.dict("os.environ", {"LLM_ENABLED": "false"}, clear=True):
        agent = ClassifierAgent()
        assert agent.llm is None

        paper = _make_paper()
        decision = agent.classify(paper)
        assert decision.is_world_model is True
        assert decision.confidence == "high"


def test_classifier_agent_with_mock_llm() -> None:
    """测试 ClassifierAgent 接入真实 LLM 接口时的结构化调用。"""
    mock_llm = MagicMock()

    # 模拟 with_structured_output(FilterDecision)
    mock_filter_seq = MagicMock()
    mock_filter_seq.invoke.return_value = FilterDecision(
        is_world_model=True,
        confidence="high",
        needs_full_text=False,
        reasoning="【LLM 研判】该论文明确提出基于动作条件化的流匹配潜空间动力学模型。",
    )

    # 模拟 with_structured_output(PaperInsight)
    mock_insight_seq = MagicMock()
    mock_insight_seq.invoke.return_value = PaperInsight(
        is_world_model=True,
        model_paradigm="Flow-Matching",
        action_conditioned=True,
        eval_environments=["CALVIN", "Real Robot"],
        core_innovation="【LLM 提取】连续流匹配世界模型实现实时闭环动作规划。",
    )

    def side_effect(schema: Any) -> Any:
        if schema == FilterDecision:
            return mock_filter_seq
        if schema == PaperInsight:
            return mock_insight_seq
        return MagicMock()

    mock_llm.with_structured_output.side_effect = side_effect

    agent = ClassifierAgent(llm=mock_llm)
    paper = _make_paper()

    # 1. 执行分类
    decision = agent.classify(paper)
    mock_filter_seq.invoke.assert_called_once()
    assert decision.is_world_model is True
    assert "【LLM 研判】" in decision.reasoning

    # 2. 执行结构化洞察提取
    insight = agent.extract_insight(paper)
    mock_insight_seq.invoke.assert_called_once()
    assert insight.model_paradigm == "Flow-Matching"
    assert "【LLM 提取】" in insight.core_innovation


def test_classifier_agent_llm_exception_fallback() -> None:
    """当 LLM 调用发生网络或解析异常时，平滑降级到启发式规则引擎。"""
    mock_llm = MagicMock()
    mock_failing_seq = MagicMock()
    mock_failing_seq.invoke.side_effect = RuntimeError("LLM API 连接超时或配额超限")
    mock_llm.with_structured_output.return_value = mock_failing_seq

    agent = ClassifierAgent(llm=mock_llm)
    paper = _make_paper()

    # 分类调用失败回退
    decision = agent.classify(paper)
    assert decision.is_world_model is True
    assert decision.confidence == "high"

    # 提取调用失败回退
    insight = agent.extract_insight(paper)
    assert insight.is_world_model is True
    assert insight.model_paradigm == "Flow-Matching"


# =========================================================================
# 4. RAGSynthesizer 接入 LLM 与对比问答测试
# =========================================================================


def test_rag_synthesizer_auto_init(tmp_path: Any) -> None:
    """测试 RAGSynthesizer 在无配置时自动回退启发式，有配置时接入 LLM。"""
    vector_store = VectorStore(storage_path=tmp_path / "qdrant_test")

    with patch.dict("os.environ", {"LLM_ENABLED": "false"}, clear=True):
        synth = RAGSynthesizer(vector_store=vector_store)
        assert synth.llm is None


def test_rag_synthesizer_with_universal_llm(tmp_path: Any) -> None:
    """测试 RAGSynthesizer 接入通用 LLM 时的对比问答生成。"""
    vector_store = VectorStore(storage_path=tmp_path / "qdrant_test")
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = (
        "### 🤖 LLM 针对世界模型的横向深度对比\n\n"
        "| 模型范式 | 代表作 | 动作闭环 |\n"
        "|---|---|---|\n"
        "| Flow-Matching | 2609.00001 | 是 |\n\n"
        "### 📚 参考文献\n"
        "- [2609.00001] Action-conditioned Flow Matching World Model"
    )
    mock_llm.invoke.return_value = mock_response

    synth = RAGSynthesizer(vector_store=vector_store, llm=mock_llm)

    sample_results = [
        VectorSearchResult(
            arxiv_id="2609.00001",
            title="Action-conditioned Flow Matching World Model",
            chunk_index=0,
            score=0.92,
            chunk_text="Action conditioning details in CALVIN benchmark.",
            model_paradigm="Flow-Matching",
            action_conditioned=True,
            is_world_model=True,
            eval_environments=["CALVIN"],
            core_innovation="Flow Matching 动作闭环世界模型",
        )
    ]

    answer = synth.generate_answer(
        query="对比近期 Flow Matching 与扩散世界模型",
        search_results=sample_results,
        lang="zh",
    )

    mock_llm.invoke.assert_called_once()
    assert "LLM 针对世界模型的横向深度对比" in answer
    assert "2609.00001" in answer


# =========================================================================
# 5. 工作流集成注入测试 (WorkflowEngine / build_workflow_graph)
# =========================================================================


def test_workflow_engine_llm_injection(tmp_path: Any) -> None:
    """测试 WorkflowEngine 与 build_workflow_graph 支持通用 llm 依赖注入。"""
    mock_llm = MagicMock()
    mock_vec = MagicMock(spec=VectorStore)
    mock_db = MagicMock()

    engine = WorkflowEngine(db_manager=mock_db, vector_store=mock_vec, llm=mock_llm)
    assert engine.classifier.llm is mock_llm

    graph = build_workflow_graph(
        db_manager=mock_db, vector_store=mock_vec, llm=mock_llm
    )
    assert graph is not None
