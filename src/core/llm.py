"""Universal Large Language Model (LLM) abstraction and factory module.

Provides a unified, universal LLM access interface for ClassifierAgent and RAGSynthesizer.

Key Features:
1. Universal standard protocol: Built on the OpenAI-compatible API protocol,
   supporting OpenAI, Google Gemini, DeepSeek, Qwen (DashScope), Claude (via proxy),
   Ollama (local models), vLLM, SiliconFlow, and other providers;
2. Scenario separation & dual-mode control (LLM_ENABLED toggle):
   - Scenario A (Cloud deployment / GitHub Actions CI / Streamlit Cloud): LLM_ENABLED=false (or unset),
     forces the local heuristic rule engine: 0 token cost, 0 network requests, 0 failure points;
   - Scenario B (Local run / deep analysis): LLM_ENABLED=true, activates the configured LLM;
     if misconfigured or during network issues, gracefully falls back to the heuristic rule engine;
3. Provider & endpoint URL matching and API key validity checks:
   Ensures API key is non-empty and non-placeholder, and validates provider and endpoint URL compatibility;
4. Zero-friction configuration: Reads environment variables (LLM_ENABLED, LLM_API_KEY / OPENAI_API_KEY, etc.)
   with support for explicit parameter overrides;
5. Fault tolerance & graceful fallback: Automatically degrades to local heuristic engine when credentials fail.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Ensure .env is loaded
load_dotenv()

logger = logging.getLogger(__name__)

# Predefined base URLs and representative models for common LLM providers
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.0-flash",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "default_model": "qwen2.5:7b",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "default_model": "deepseek-ai/DeepSeek-V3",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "default_model": "glm-4-flash",
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "default_model": "moonshot-v1-8k",
    },
}

# Provider to characteristic domain keyword mapping
PROVIDER_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "openai": ["api.openai.com"],
    "gemini": ["generativelanguage.googleapis.com", "googleapis.com"],
    "deepseek": ["api.deepseek.com", "deepseek.com"],
    "qwen": ["dashscope.aliyuncs.com", "aliyuncs.com"],
    "ollama": ["11434", "localhost:11434", "127.0.0.1:11434", "ollama"],
    "siliconflow": ["api.siliconflow.cn", "siliconflow.cn"],
    "zhipu": ["open.bigmodel.cn", "bigmodel.cn"],
    "moonshot": ["api.moonshot.cn", "moonshot.cn"],
}

# Common placeholder keys
PLACEHOLDER_KEYS: set[str] = {
    "your_openai_api_key_here",
    "your_api_key_here",
    "your_key_here",
    "your_deepseek_api_key_here",
    "your_gemini_api_key_here",
    "your_qwen_api_key_here",
    "your_llm_api_key_here",
    "sk-placeholder",
    "xxx",
    "none",
}


def is_truthy(val: str | None) -> bool:
    """Determine whether a string represents boolean True (e.g., 'true', '1', 'yes', 'on', 'enable', 'enabled')."""
    if not val:
        return False
    return val.strip().lower() in ("true", "1", "yes", "on", "enable", "enabled")


class LLMConfig(BaseModel):
    """Universal LLM configuration parameters."""

    enabled: bool = Field(default=False, description="Whether LLM is enabled")
    api_key: str | None = Field(default=None, description="API secret key")
    base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI-compatible API Base URL",
    )
    model: str = Field(default="gpt-4o-mini", description="Model name")
    temperature: float = Field(
        default=0.0, description="Sampling temperature (0.0 for high determinism)"
    )
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    max_retries: int = Field(default=2, description="Maximum retry attempts")
    provider: str | None = Field(default=None, description="Provider identifier")


def detect_provider_from_url(url: str | None) -> str | None:
    """Infer the LLM provider from domain keywords in the URL."""
    if not url:
        return None
    url_lower = url.lower()
    for prov, kws in PROVIDER_DOMAIN_KEYWORDS.items():
        if any(kw in url_lower for kw in kws):
            return prov
    return None


def check_api_credential_match(
    key: str | None,
    provider: str | None = None,
    base_url: str | None = None,
) -> tuple[bool, str]:
    """Validate API key and check compatibility between provider and endpoint URL (base_url).

    Validation rules:
    1. Basic API Key validity: Must be non-empty, non-whitespace, and not a template placeholder;
    2. Provider and URL matching: If provider is explicitly specified, it must not conflict with
       well-known third-party domains (e.g., provider='deepseek' with OpenAI base URL);
    3. Custom proxy/generic: Self-hosted proxies or local services match as long as they do not
       conflict with known provider domains.
    """
    if not key:
        return False, "API Key is empty"

    clean_key = key.strip()
    if not clean_key:
        return False, "API Key is empty"

    if clean_key.lower() in PLACEHOLDER_KEYS:
        return False, f"API Key is a template placeholder: {clean_key}"

    req_prov = provider.strip().lower() if provider else None
    url_str = base_url.strip() if base_url else ""
    url_prov = detect_provider_from_url(url_str)

    # Provider and URL consistency check
    if req_prov and url_prov and req_prov != url_prov:
        return (
            False,
            f"Provider '{req_prov}' conflicts with base URL '{url_str}' owned by provider '{url_prov}'",
        )

    return True, "OK"


def is_valid_api_key(
    key: str | None,
    provider: str | None = None,
    base_url: str | None = None,
) -> bool:
    """Check whether the API Key is valid and the provider matches the base URL."""
    is_valid, reason = check_api_credential_match(
        key, provider=provider, base_url=base_url
    )
    if not is_valid and key and key.strip().lower() not in PLACEHOLDER_KEYS:
        logger.warning("LLM API credential validation failed: %s", reason)
    return is_valid


def resolve_llm_config(
    model: str | None = None,
    temperature: float | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    provider: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    enabled: bool | None = None,
) -> LLMConfig:
    """Resolve the final LLM configuration from arguments, presets, and environment variables.

    Precedence: Explicit argument > Provider preset > Environment variable > Default fallback.
    """
    # Resolve provider
    if provider:
        resolved_provider = provider.strip().lower()
    elif base_url and detect_provider_from_url(base_url):
        resolved_provider = detect_provider_from_url(base_url)
    else:
        resolved_provider = (os.getenv("LLM_PROVIDER") or "").strip().lower() or None

    preset = PROVIDER_PRESETS.get(resolved_provider) if resolved_provider else None

    # 1. Determine if LLM is enabled (LLM_ENABLED)
    env_enabled_str = os.getenv("LLM_ENABLED") or os.getenv("ENABLE_LLM")
    if enabled is not None:
        resolved_enabled = enabled
    elif api_key is not None:
        resolved_enabled = True
    elif env_enabled_str is not None:
        resolved_enabled = is_truthy(env_enabled_str)
    else:
        resolved_enabled = False

    # 2. Resolve Base URL
    if base_url is not None:
        resolved_base_url = base_url.strip()
    elif provider is not None and preset and preset.get("base_url"):
        resolved_base_url = preset["base_url"].strip()
    else:
        resolved_base_url = (
            os.getenv("LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or (preset["base_url"] if preset else None)
            or "https://api.openai.com/v1"
        ).strip()

    # 3. Resolve API Key
    if (
        resolved_provider == "ollama" or "11434" in resolved_base_url
    ) and api_key is None:
        resolved_key = "ollama"
    else:
        resolved_key = (
            api_key
            or os.getenv("LLM_API_KEY")
            or (os.getenv("GEMINI_API_KEY") if resolved_provider == "gemini" else None)
            or (
                os.getenv("DEEPSEEK_API_KEY")
                if resolved_provider == "deepseek"
                else None
            )
            or (os.getenv("DASHSCOPE_API_KEY") if resolved_provider == "qwen" else None)
            or os.getenv("OPENAI_API_KEY")
        )
    if resolved_key:
        resolved_key = resolved_key.strip()
        if resolved_key.lower() in PLACEHOLDER_KEYS:
            resolved_key = None

    if (
        resolved_provider == "ollama" or "11434" in resolved_base_url
    ) and not resolved_key:
        resolved_key = "ollama"

    # 4. Resolve Model Name
    if model is not None:
        resolved_model = model.strip()
    elif provider is not None and preset and preset.get("default_model"):
        resolved_model = preset["default_model"].strip()
    else:
        resolved_model = (
            os.getenv("LLM_MODEL")
            or os.getenv("OPENAI_MODEL_NAME")
            or (preset["default_model"] if preset else None)
            or "gpt-4o-mini"
        ).strip()

    # 5. Temperature, Timeout, Retries
    resolved_temp = (
        temperature
        if temperature is not None
        else float(os.getenv("LLM_TEMPERATURE", "0.0"))
    )
    resolved_timeout = (
        timeout if timeout is not None else float(os.getenv("LLM_TIMEOUT", "30.0"))
    )
    resolved_retries = (
        max_retries
        if max_retries is not None
        else int(os.getenv("LLM_MAX_RETRIES", "2"))
    )

    return LLMConfig(
        enabled=resolved_enabled,
        api_key=resolved_key,
        base_url=resolved_base_url,
        model=resolved_model,
        temperature=resolved_temp,
        timeout=resolved_timeout,
        max_retries=resolved_retries,
        provider=resolved_provider,
    )


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    provider: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    enabled: bool | None = None,
    strict: bool = False,
    **extra_kwargs: Any,
) -> BaseChatModel | None:
    """Build a universal LangChain ChatModel instance.

    Uses standard OpenAI-compatible format (`ChatOpenAI`), applicable to
    OpenAI, Gemini, DeepSeek, Qwen, Ollama, and other providers.

    Args:
        model: Model name (e.g., 'gpt-4o-mini', 'gemini-2.0-flash', 'deepseek-chat', 'qwen-plus').
        temperature: Sampling temperature (0.0 ~ 1.0).
        base_url: Compatible API endpoint URL (e.g., 'https://api.deepseek.com/v1').
        api_key: API key. Resolved from environment variables if not provided.
        provider: Provider shorthand (e.g., 'openai', 'gemini', 'deepseek', 'qwen', 'ollama').
        timeout: Request timeout in seconds.
        max_retries: Request retry count.
        enabled: Explicitly specify whether LLM is enabled.
        strict: If True, raises ValueError on invalid credentials or when disabled; returns None if False.
        **extra_kwargs: Extra keyword arguments passed to underlying ChatOpenAI.

    Returns:
        BaseChatModel instance, or None if disabled or credential validation fails with strict=False.
    """
    config = resolve_llm_config(
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        provider=provider,
        timeout=timeout,
        max_retries=max_retries,
        enabled=enabled,
    )

    # 1. Check master toggle: return None when LLM is disabled (fallback to heuristic rule engine)
    if not config.enabled:
        logger.info(
            "LLM is disabled (LLM_ENABLED=false), running in local heuristic rule engine mode."
        )
        if strict:
            raise ValueError("LLM is disabled (LLM_ENABLED=false)")
        return None

    # 2. LLM is enabled (LLM_ENABLED=true): validate credentials and provider/URL matching
    is_valid, reason = check_api_credential_match(
        config.api_key,
        provider=config.provider,
        base_url=config.base_url,
    )
    if not is_valid:
        msg = (
            f"LLM activation configured (LLM_ENABLED=true), but API Key or provider/URL validation failed: {reason}. "
            "Automatically degrading to local heuristic rule engine to maintain service availability."
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
        return None

    # 3. Build standard ChatOpenAI client
    try:
        chat_model = ChatOpenAI(
            model=config.model,
            temperature=config.temperature,
            api_key=config.api_key,  # type: ignore[arg-type]
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
            **extra_kwargs,
        )
        logger.info(
            "Successfully initialized universal LLM instance: provider=%s, model=%s, base_url=%s",
            config.provider or "generic/openai-compatible",
            config.model,
            config.base_url,
        )
        return chat_model
    except Exception as exc:
        logger.error(
            "Failed to initialize universal LLM instance: %s. Falling back to local heuristic rule engine.",
            exc,
        )
        if strict:
            raise
        return None


def get_llm_info(
    llm: Any | None = None,
    config: LLMConfig | None = None,
) -> dict[str, Any]:
    """Get the current LLM runtime status and display information (for Web UI or logging)."""
    cfg = config or resolve_llm_config()
    if llm is None:
        if not cfg.enabled:
            return {
                "is_active": False,
                "status_code": "DISABLED",
                "provider": "Rule-Based",
                "model": "Local Heuristic Rule Engine (Offline/Cloud Deployment Mode)",
                "base_url": "N/A",
                "hint": "LLM is disabled (LLM_ENABLED=false). Cloud deployments default to heuristics; set LLM_ENABLED=true in .env for local deep analysis.",
            }

        is_match, reason = check_api_credential_match(
            cfg.api_key, provider=cfg.provider, base_url=cfg.base_url
        )
        hint_msg = (
            f"LLM enabled but validation failed ({reason}); gracefully degraded to heuristic engine."
            if not is_match
            else "LLM enabled but service not ready; gracefully degraded to heuristic engine."
        )
        return {
            "is_active": False,
            "status_code": "FALLBACK",
            "provider": "Rule-Based",
            "model": "Local Heuristic Rule Engine (Graceful Fallback Mode)",
            "base_url": "N/A",
            "hint": hint_msg,
        }

    model_name = getattr(llm, "model_name", getattr(llm, "model", "Unknown Model"))
    base_url = str(getattr(llm, "openai_api_base", getattr(llm, "base_url", "")))

    provider_name = "OpenAI Compatible"
    for p_name, p_info in PROVIDER_PRESETS.items():
        if p_info["base_url"].rstrip("/") in base_url.rstrip("/"):
            provider_name = p_name.upper()
            break

    return {
        "is_active": True,
        "status_code": "ACTIVE",
        "provider": provider_name,
        "model": str(model_name),
        "base_url": base_url or "https://api.openai.com/v1",
        "hint": f"Successfully connected to {provider_name} model: {model_name}",
    }


__all__ = [
    "PROVIDER_PRESETS",
    "LLMConfig",
    "check_api_credential_match",
    "detect_provider_from_url",
    "get_llm",
    "get_llm_info",
    "is_truthy",
    "is_valid_api_key",
    "resolve_llm_config",
]
