"""Unit tests: Universal LLM interface specification, provider adaptation, and Classifier/RAG integration."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_openai import ChatOpenAI

from src.agents.classifier_agent import ClassifierAgent, FilterDecision
from src.app import RAGSynthesizer
from src.core.llm import (
    PROVIDER_PRESETS,
    LLMConfig,
    check_api_credential_match,
    detect_provider_from_url,
    get_llm,
    get_llm_info,
    is_truthy,
    is_valid_api_key,
    resolve_llm_config,
)
from src.core.schema import PaperInsight, RawPaper
from src.storage import VectorSearchResult, VectorStore
from src.workflows.graph import WorkflowEngine, build_workflow_graph


def _make_paper() -> RawPaper:
    return RawPaper(
        arxiv_id="2609.12345",
        title="Flow Matching for Robotic World Models",
        authors=["Alice", "Bob"],
        abstract="We propose an action-conditioned continuous normalizing flow world model on CALVIN.",
        categories=["cs.RO", "cs.AI"],
        published_date=datetime(2026, 9, 30, tzinfo=UTC),
        pdf_url="https://arxiv.org/pdf/2609.12345.pdf",
    )


# =========================================================================
# 1. API Key Validity and Configuration Resolution Tests
# =========================================================================


def test_is_truthy() -> None:
    """Test boolean environment variable resolution."""
    assert is_truthy("true") is True
    assert is_truthy("True") is True
    assert is_truthy("1") is True
    assert is_truthy("yes") is True
    assert is_truthy("YES") is True
    assert is_truthy("on") is True
    assert is_truthy("enable") is True
    assert is_truthy("enabled") is True

    assert is_truthy("false") is False
    assert is_truthy("0") is False
    assert is_truthy("no") is False
    assert is_truthy("") is False
    assert is_truthy(None) is False
    assert is_truthy("other") is False


def test_api_key_placeholder_and_validity() -> None:
    """Test API key placeholder identification and validity filtering."""
    assert is_valid_api_key(None) is False
    assert is_valid_api_key("") is False
    assert is_valid_api_key("   ") is False
    assert is_valid_api_key("your_openai_api_key_here") is False
    assert is_valid_api_key("your_deepseek_api_key_here") is False
    assert is_valid_api_key("your_gemini_api_key_here") is False
    assert is_valid_api_key("your_llm_api_key_here") is False
    assert is_valid_api_key("sk-placeholder") is False
    assert is_valid_api_key("xxx") is False
    assert is_valid_api_key("none") is False

    assert is_valid_api_key("sk-real-valid-api-key-1234567890") is True
    assert is_valid_api_key("ollama") is True


def test_api_key_and_url_validation() -> None:
    """Test API key non-empty/non-placeholder check, and provider vs. endpoint URL matching."""
    # 1. Standard matching cases (no restriction on key format as long as non-empty and non-placeholder)
    ok, _ = check_api_credential_match(
        "sk-proj-test123456789",
        provider="openai",
        base_url="https://api.openai.com/v1",
    )
    assert ok is True

    # Allow different vendor formats (e.g., Gemini arbitrary non-empty non-placeholder key)
    ok, _ = check_api_credential_match(
        "AIzaSyD-dummy-gemini-key",
        provider="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    assert ok is True

    ok, _ = check_api_credential_match(
        "sk-deepseek-123456",
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

    # Custom proxy endpoint (no domain restriction, matches if no conflict)
    ok, _ = check_api_credential_match(
        "any-token",
        provider="openai",
        base_url="https://my-custom-proxy.internal/v1",
    )
    assert ok is True

    # 2. Empty values and placeholder filtering
    ok_empty, reason = check_api_credential_match("", provider="openai")
    assert ok_empty is False
    assert "empty" in reason.lower()

    ok_none, reason = check_api_credential_match(None, provider="openai")
    assert ok_none is False
    assert "empty" in reason.lower()

    ok_placeholder, reason = check_api_credential_match(
        "your_openai_api_key_here", provider="openai"
    )
    assert ok_placeholder is False
    assert "placeholder" in reason.lower()

    # 3. Provider and URL conflict validation (e.g., DeepSeek with OpenAI URL, or Gemini with OpenAI URL)
    ok_conflict_ds, reason = check_api_credential_match(
        "any-valid-key",
        provider="deepseek",
        base_url="https://api.openai.com/v1",
    )
    assert ok_conflict_ds is False
    assert "conflict" in reason.lower()

    ok_conflict_gemini, reason = check_api_credential_match(
        "any-valid-key",
        provider="gemini",
        base_url="https://api.openai.com/v1",
    )
    assert ok_conflict_gemini is False
    assert "conflict" in reason.lower()

    ok_conflict_openai, reason = check_api_credential_match(
        "any-valid-key",
        provider="openai",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    assert ok_conflict_openai is False
    assert "conflict" in reason.lower()

    # 4. is_valid_api_key validation
    assert is_valid_api_key(
        "any-valid-key",
        provider="openai",
        base_url="https://api.openai.com/v1",
    )
    assert not is_valid_api_key(
        "any-valid-key",
        provider="deepseek",
        base_url="https://api.openai.com/v1",
    )


def test_detect_provider_from_url() -> None:
    assert detect_provider_from_url("https://api.openai.com/v1") == "openai"
    assert (
        detect_provider_from_url(
            "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        == "gemini"
    )
    assert detect_provider_from_url("https://api.deepseek.com/v1") == "deepseek"
    assert (
        detect_provider_from_url("https://dashscope.aliyuncs.com/compatible-mode/v1")
        == "qwen"
    )
    assert detect_provider_from_url("http://localhost:11434/v1") == "ollama"
    assert detect_provider_from_url("https://api.siliconflow.cn/v1") == "siliconflow"
    assert detect_provider_from_url("https://my-custom-proxy.internal/v1") is None
    assert detect_provider_from_url(None) is None


def test_resolve_llm_config_scenarios() -> None:
    """Test LLM_ENABLED toggle resolution across different scenarios."""
    # Scenario 1: Cloud deployment / GitHub Actions default (no LLM variables, defaults to heuristics)
    with patch.dict("os.environ", {}, clear=True):
        cfg = resolve_llm_config()
        assert cfg.enabled is False

    # Scenario 2: Explicitly configured LLM_ENABLED=false
    with patch.dict("os.environ", {"LLM_ENABLED": "false"}, clear=True):
        cfg = resolve_llm_config()
        assert cfg.enabled is False

    # Scenario 3: Explicitly configured LLM_ENABLED=true
    with patch.dict("os.environ", {"LLM_ENABLED": "true"}, clear=True):
        cfg = resolve_llm_config()
        assert cfg.enabled is True

    # Scenario 4: Explicit parameter overrides environment variable
    with patch.dict("os.environ", {"LLM_ENABLED": "false"}, clear=True):
        cfg = resolve_llm_config(enabled=True)
        assert cfg.enabled is True


def test_resolve_llm_config_providers() -> None:
    """Test common provider presets (DeepSeek, Gemini, Qwen, Ollama, SiliconFlow)."""
    # 1. DeepSeek preset
    cfg_ds = resolve_llm_config(provider="deepseek")
    assert cfg_ds.base_url == PROVIDER_PRESETS["deepseek"]["base_url"]
    assert cfg_ds.model == PROVIDER_PRESETS["deepseek"]["default_model"]

    # 2. Gemini preset (OpenAI-compatible endpoint)
    cfg_gemini = resolve_llm_config(provider="gemini")
    assert cfg_gemini.base_url == PROVIDER_PRESETS["gemini"]["base_url"]
    assert cfg_gemini.model == PROVIDER_PRESETS["gemini"]["default_model"]

    # 3. Qwen DashScope preset
    cfg_qwen = resolve_llm_config(provider="qwen")
    assert cfg_qwen.base_url == PROVIDER_PRESETS["qwen"]["base_url"]
    assert cfg_qwen.model == PROVIDER_PRESETS["qwen"]["default_model"]

    # 4. Local Ollama preset (automatically populates dummy key for local keyless environment)
    cfg_ollama = resolve_llm_config(provider="ollama")
    assert cfg_ollama.base_url == PROVIDER_PRESETS["ollama"]["base_url"]
    assert cfg_ollama.model == PROVIDER_PRESETS["ollama"]["default_model"]
    assert cfg_ollama.api_key == "ollama"

    # 5. SiliconFlow preset
    cfg_sf = resolve_llm_config(provider="siliconflow")
    assert cfg_sf.base_url == PROVIDER_PRESETS["siliconflow"]["base_url"]
    assert cfg_sf.model == PROVIDER_PRESETS["siliconflow"]["default_model"]


def test_resolve_llm_config_parameter_priority() -> None:
    """Test explicit arguments overriding presets and defaults."""
    cfg = resolve_llm_config(
        provider="openai",
        base_url="https://custom.endpoint.com/v1",
        model="custom-model-x",
        temperature=0.7,
        timeout=15.0,
        max_retries=5,
        api_key="sk-explicit-token-123456",
    )
    assert cfg.base_url == "https://custom.endpoint.com/v1"
    assert cfg.model == "custom-model-x"
    assert cfg.temperature == 0.7
    assert cfg.timeout == 15.0
    assert cfg.max_retries == 5
    assert cfg.api_key == "sk-explicit-token-123456"
    assert cfg.enabled is True  # Explicit api_key activates automatically


def test_resolve_llm_config_from_custom_env() -> None:
    """Test resolving configuration from custom environment variables."""
    custom_env = {
        "LLM_ENABLED": "true",
        "LLM_PROVIDER": "deepseek",
        "LLM_API_KEY": "sk-deepseek-env-token",
        "LLM_BASE_URL": "https://api.deepseek.com/v1",
        "LLM_MODEL": "deepseek-coder",
        "LLM_TEMPERATURE": "0.3",
        "LLM_TIMEOUT": "45.0",
        "LLM_MAX_RETRIES": "4",
    }
    with patch.dict("os.environ", custom_env, clear=True):
        cfg = resolve_llm_config()
        assert cfg.enabled is True
        assert cfg.provider == "deepseek"
        assert cfg.api_key == "sk-deepseek-env-token"
        assert cfg.base_url == "https://api.deepseek.com/v1"
        assert cfg.model == "deepseek-coder"
        assert cfg.temperature == 0.3
        assert cfg.timeout == 45.0
        assert cfg.max_retries == 4


def test_resolve_llm_config_provider_custom_url_override() -> None:
    """Test that when environment specifies provider but custom base_url and model are set, custom values take precedence."""
    env = {
        "LLM_PROVIDER": "openai",
        "LLM_BASE_URL": "https://openai-compatible.azure.com/v1",
        "LLM_MODEL": "azure-gpt-4o",
    }
    with patch.dict("os.environ", env, clear=True):
        cfg = resolve_llm_config()
        assert cfg.base_url == "https://openai-compatible.azure.com/v1"
        assert cfg.model == "azure-gpt-4o"


# =========================================================================
# 2. get_llm and get_llm_info Tests
# =========================================================================


def test_get_llm_offline_fallback() -> None:
    """get_llm gracefully returns None when disabled or without a valid API key."""
    # Scenario A: LLM_ENABLED=false (Cloud deployment scenario)
    with patch.dict("os.environ", {"LLM_ENABLED": "false"}, clear=True):
        llm = get_llm()
        assert llm is None

        with pytest.raises(ValueError, match="LLM is disabled"):
            get_llm(strict=True)

    # Scenario B: LLM_ENABLED=true but no API key configured (Local automatic fallback scenario)
    with patch.dict("os.environ", {"LLM_ENABLED": "true"}, clear=True):
        llm = get_llm()
        assert llm is None

        with pytest.raises(ValueError, match="validation failed"):
            get_llm(strict=True)

    # Scenario C: LLM_ENABLED=true with invalid placeholder
    with patch.dict(
        "os.environ",
        {"LLM_ENABLED": "true", "LLM_API_KEY": "your_openai_api_key_here"},
        clear=True,
    ):
        llm = get_llm()
        assert llm is None

    # Scenario D: LLM_ENABLED=true but Provider and URL conflict causes validation failure
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

        with pytest.raises(ValueError, match="conflict"):
            get_llm(strict=True)


def test_get_llm_success() -> None:
    """Builds ChatOpenAI instance successfully when valid key is provided and enabled."""
    llm = get_llm(
        enabled=True,
        api_key="sk-test-valid-key-123456",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        temperature=0.1,
    )
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "deepseek-chat"
    assert llm.temperature == 0.1
    assert str(llm.openai_api_base).rstrip("/") == "https://api.deepseek.com/v1"


def test_get_llm_info() -> None:
    """Test extracting LLM runtime status metadata."""
    # 1. Scenario A: LLM disabled (DISABLED)
    info_disabled = get_llm_info(llm=None, config=LLMConfig(enabled=False))
    assert info_disabled["is_active"] is False
    assert info_disabled["status_code"] == "DISABLED"
    assert info_disabled["provider"] == "Rule-Based"

    # 2. Scenario B: LLM enabled but key invalid causes fallback (FALLBACK)
    info_fallback = get_llm_info(
        llm=None,
        config=LLMConfig(enabled=True, api_key="your_placeholder", provider="openai"),
    )
    assert info_fallback["is_active"] is False
    assert info_fallback["status_code"] == "FALLBACK"

    # 3. Scenario C: Active instance connected (ACTIVE)
    mock_llm = MagicMock()
    mock_llm.model_name = "gemini-2.0-flash"
    mock_llm.openai_api_base = (
        "https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    info_active = get_llm_info(llm=mock_llm)
    assert info_active["is_active"] is True
    assert info_active["status_code"] == "ACTIVE"
    assert info_active["provider"] == "GEMINI"
    assert info_active["model"] == "gemini-2.0-flash"


# =========================================================================
# 3. ClassifierAgent LLM Integration and Fallback Tests
# =========================================================================


def test_classifier_agent_auto_init_llm() -> None:
    """Test ClassifierAgent calling get_llm() by default."""
    # Environment variable not active -> llm is None, uses rules
    with patch.dict("os.environ", {"LLM_ENABLED": "false"}, clear=True):
        agent = ClassifierAgent()
        assert agent.llm is None
        paper = _make_paper()
        decision = agent.classify(paper)
        assert decision.is_world_model is True
        assert decision.confidence == "high"


def test_classifier_agent_with_mock_llm() -> None:
    """Test ClassifierAgent structured invocation when integrated with LLM."""
    mock_llm = MagicMock()

    # Mock with_structured_output(FilterDecision)
    mock_filter_seq = MagicMock()
    mock_filter_seq.invoke.return_value = FilterDecision(
        is_world_model=True,
        confidence="high",
        needs_full_text=False,
        reasoning="[LLM Analysis] The paper explicitly proposes an action-conditioned flow matching latent space dynamics model.",
    )

    # Mock with_structured_output(PaperInsight)
    mock_insight_seq = MagicMock()
    mock_insight_seq.invoke.return_value = PaperInsight(
        is_world_model=True,
        model_paradigm="Flow-Matching",
        action_conditioned=True,
        eval_environments=["CALVIN", "Real Robot"],
        core_innovation="[LLM Extraction] Continuous flow matching world model enables real-time closed-loop action planning.",
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

    # 1. Execute classification
    decision = agent.classify(paper)
    mock_filter_seq.invoke.assert_called_once()
    assert decision.is_world_model is True
    assert "[LLM Analysis]" in decision.reasoning

    # 2. Execute structured insight extraction
    insight = agent.extract_insight(paper)
    mock_insight_seq.invoke.assert_called_once()
    assert insight.model_paradigm == "Flow-Matching"
    assert "[LLM Extraction]" in insight.core_innovation


def test_classifier_agent_llm_exception_fallback() -> None:
    """Gracefully falls back to heuristic rule engine when LLM call encounters network or parsing error."""
    mock_llm = MagicMock()
    mock_failing_seq = MagicMock()
    mock_failing_seq.invoke.side_effect = RuntimeError(
        "LLM API connection timeout or quota exceeded"
    )
    mock_llm.with_structured_output.return_value = mock_failing_seq

    agent = ClassifierAgent(llm=mock_llm)
    paper = _make_paper()

    # Classification failure fallback
    decision = agent.classify(paper)
    assert decision.is_world_model is True
    assert decision.confidence == "high"

    # Extraction failure fallback
    insight = agent.extract_insight(paper)
    assert insight.is_world_model is True
    assert insight.model_paradigm == "Flow-Matching"


# =========================================================================
# 4. RAGSynthesizer LLM Integration and Comparative Q&A Tests
# =========================================================================


def test_rag_synthesizer_auto_init(tmp_path: Any) -> None:
    """Test RAGSynthesizer falling back to heuristics without config, using LLM with config."""
    vector_store = VectorStore(storage_path=tmp_path / "qdrant_test")

    with patch.dict("os.environ", {"LLM_ENABLED": "false"}, clear=True):
        synth = RAGSynthesizer(vector_store=vector_store)
        assert synth.llm is None


def test_rag_synthesizer_with_universal_llm(tmp_path: Any) -> None:
    """Test RAGSynthesizer comparative Q&A generation when connected to LLM."""
    vector_store = VectorStore(storage_path=tmp_path / "qdrant_test")
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = (
        "### 🤖 LLM Comparative Analysis of World Models\n\n"
        "| Model Paradigm | Representative Paper | Action-Conditioned |\n"
        "|---|---|---|\n"
        "| Flow-Matching | 2609.00001 | Yes |\n\n"
        "### 📚 References\n"
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
            core_innovation="Flow Matching action-conditioned world model",
        )
    ]

    answer = synth.generate_answer(
        query="Compare recent Flow Matching and diffusion world models",
        search_results=sample_results,
        lang="en",
    )

    mock_llm.invoke.assert_called_once()
    assert "LLM Comparative Analysis of World Models" in answer
    assert "2609.00001" in answer


# =========================================================================
# 5. Workflow Integration Dependency Injection Tests
# =========================================================================


def test_workflow_engine_llm_injection(tmp_path: Any) -> None:
    """Test WorkflowEngine and build_workflow_graph supporting llm dependency injection."""
    mock_llm = MagicMock()
    mock_vec = MagicMock(spec=VectorStore)
    mock_db = MagicMock()

    engine = WorkflowEngine(db_manager=mock_db, vector_store=mock_vec, llm=mock_llm)
    assert engine.classifier.llm is mock_llm

    graph = build_workflow_graph(
        db_manager=mock_db, vector_store=mock_vec, llm=mock_llm
    )
    assert graph is not None
