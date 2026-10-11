"""通用大语言模型 (LLM) 抽象与接入工厂模块。

为智能体 (ClassifierAgent) 和问答系统 (RAGSynthesizer) 提供统一、通用的 LLM 接入定义。
主要特性：
1. 通用标准协议：基于 OpenAI 兼容协议 (OpenAI-compatible API)，
   天然支持 OpenAI、Google Gemini、DeepSeek、Qwen (通义千问)、Claude (通过中转/代理)、
   Ollama (本地模型)、vLLM、SiliconFlow 等绝大多数大模型服务商；
2. 运行场景分离与双模控制 (LLM_ENABLED 开关)：
   - 场景 A (云端部署 / GitHub Actions CI / Streamlit Cloud)：LLM_ENABLED=false (或未配置)，
     强制使用本地启发式规则引擎，0 Token 成本、0 网络请求、0 故障点；
   - 场景 B (本地运行 / 自主研判)：LLM_ENABLED=true，正常配置 Key 时激活对应大模型；
     若未正常配置或网络异常，系统自动优雅降级为启发式规则引擎，保证服务 100% 可用；
3. 严格的凭证三元组匹配校验 (API Key 格式 + 服务商 Provider + 端点 URL)：
   不仅过滤空值和模板占位符，且严格校验 Key 的特征前缀与长度格式、指定的 Provider 与 URL 域名三者是否彼此匹配；
4. 零侵入配置：优先读取环境变量 (LLM_ENABLED, LLM_API_KEY / OPENAI_API_KEY 等)，亦支持代码显式传参；
5. 容错与平滑回退：当凭证不匹配或未配置有效 Key 时，自动降级为本地启发式规则引擎。
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 确保读取 .env 配置
load_dotenv()

logger = logging.getLogger(__name__)

# 常见 LLM 服务商预置 Base URL 与代表性模型
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

# 常见服务商与特征域名映射关系
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

# 常见的未填写的无效占位符
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
    """判断字符串是否表示布尔 True (如 'true', '1', 'yes', 'on', 'enable', 'enabled')。"""
    if not val:
        return False
    return val.strip().lower() in ("true", "1", "yes", "on", "enable", "enabled")


class LLMConfig(BaseModel):
    """通用 LLM 配置参数模型。"""

    enabled: bool = Field(default=False, description="是否激活 LLM")
    api_key: str | None = Field(default=None, description="API 密钥")
    base_url: str = Field(
        default="https://api.openai.com/v1", description="OpenAI 兼容 API Base URL"
    )
    model: str = Field(default="gpt-4o-mini", description="模型名称")
    temperature: float = Field(default=0.0, description="采样温度 (0.0 表示高确定性)")
    timeout: float = Field(default=30.0, description="请求超时时间（秒）")
    max_retries: int = Field(default=2, description="失败重试次数")
    provider: str | None = Field(default=None, description="服务商标识（如有）")


def detect_provider_from_url(url: str | None) -> str | None:
    """根据 URL 中的特征域名推断大模型服务商。"""
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
    """严谨校验 API Key 格式、服务商 (provider) 与对应的端点 URL (base_url) 是否匹配。

    校验维度：
    1. 基础有效性：非空、非空白字符、长度充足、非模板占位符；
    2. 服务商与 URL 匹配：若显式指定了 provider，不能与 base_url 指向的第三方知名服务商冲突；
    3. API Key 格式与服务商/URL 匹配：
       - Google Gemini: 必须以 'AIzaSy' 开头，URL 须指向 googleapis 或自建代理；
       - OpenAI: 必须以 'sk-' 开头 (如 sk-proj-, sk- 等)，URL 须指向 openai 或兼容端点；
       - DeepSeek: 必须以 'sk-' 开头；
       - Qwen (通义千问): 必须以 'sk-' 开头；
       - Ollama: 端口通常为 11434，密钥支持 'ollama' 或任意非空串；
       - SiliconFlow / Moonshot: 必须以 'sk-' 开头；
       - Zhipu: 包含 '.' 分隔符或标准 Token 格式；
       - 自定义代理/通用: URL 为合法 HTTP/HTTPS 地址，Key 满足通用非空规范。
    """
    if not key:
        return False, "API Key 为空"

    clean_key = key.strip()
    if not clean_key:
        return False, "API Key 为空"

    if clean_key.lower() in PLACEHOLDER_KEYS:
        return False, f"API Key 为模板占位符: {clean_key}"

    if len(clean_key) < 4:
        return False, "API Key 长度过短 (少于 4 位)"

    req_prov = provider.strip().lower() if provider else None
    url_str = base_url.strip() if base_url else ""
    url_prov = detect_provider_from_url(url_str)

    # 1. 服务商与 URL 一致性校验
    if req_prov and url_prov and req_prov != url_prov:
        return (
            False,
            f"服务商 '{req_prov}' 与 Base URL '{url_str}' 所属服务商 '{url_prov}' 冲突",
        )

    # 确定生效的目标服务商 (显式指定优先，其次根据 URL 识别)
    effective_prov = req_prov or url_prov

    # 2. 各服务商特定的 Key 格式与 URL 匹配规则
    if effective_prov == "gemini":
        if clean_key.startswith("sk-"):
            return (
                False,
                "OpenAI 格式密钥 ('sk-...') 与 Google Gemini 供应商/端点不匹配 (必须以 'AIzaSy' 开头)",
            )
        if not clean_key.startswith("AIzaSy"):
            return False, "Google Gemini API Key 格式错误：必须以 'AIzaSy' 开头"
        if len(clean_key) < 16:
            return False, "Google Gemini API Key 长度不足"
        if url_str and not any(
            kw in url_str.lower()
            for kw in ["googleapis.com", "localhost", "127.0.0.1", "gemini", "proxy"]
        ):
            return False, f"Google Gemini 密钥与端点 URL 不匹配: {url_str}"
        return True, "OK"

    if effective_prov == "ollama":
        if url_str and not any(
            kw in url_str.lower()
            for kw in ["11434", "ollama", "localhost", "127.0.0.1"]
        ):
            return False, f"Ollama 密钥与非本地服务端点不匹配: {url_str}"
        return True, "OK"

    if effective_prov == "openai":
        if clean_key.startswith("AIzaSy"):
            return (
                False,
                "Gemini 格式密钥 ('AIzaSy...') 与 OpenAI 供应商/端点不匹配 (必须以 'sk-' 开头)",
            )
        if not clean_key.startswith("sk-"):
            return False, "OpenAI API Key 格式错误：必须以 'sk-' (或 'sk-proj-') 开头"
        if len(clean_key) < 15:
            return False, "OpenAI API Key 长度不足"
        return True, "OK"

    if effective_prov == "deepseek":
        if clean_key.startswith("AIzaSy"):
            return (
                False,
                "Gemini 格式密钥 ('AIzaSy...') 与 DeepSeek 供应商/端点不匹配 (必须以 'sk-' 开头)",
            )
        if not clean_key.startswith("sk-"):
            return False, "DeepSeek API Key 格式错误：必须以 'sk-' 开头"
        if len(clean_key) < 15:
            return False, "DeepSeek API Key 长度不足"
        return True, "OK"

    if effective_prov == "qwen":
        if clean_key.startswith("AIzaSy"):
            return (
                False,
                "Gemini 格式密钥 ('AIzaSy...') 与通义千问供应商/端点不匹配 (必须以 'sk-' 开头)",
            )
        if not clean_key.startswith("sk-"):
            return False, "阿里通义千问 (DashScope) API Key 格式错误：必须以 'sk-' 开头"
        if len(clean_key) < 15:
            return False, "通义千问 API Key 长度不足"
        return True, "OK"

    if effective_prov == "siliconflow":
        if clean_key.startswith("AIzaSy"):
            return (
                False,
                "Gemini 格式密钥 ('AIzaSy...') 与 SiliconFlow 供应商/端点不匹配 (必须以 'sk-' 开头)",
            )
        if not clean_key.startswith("sk-"):
            return False, "硅基流动 (SiliconFlow) API Key 格式错误：必须以 'sk-' 开头"
        if len(clean_key) < 15:
            return False, "SiliconFlow API Key 长度不足"
        return True, "OK"

    if effective_prov == "moonshot":
        if clean_key.startswith("AIzaSy"):
            return (
                False,
                "Gemini 格式密钥 ('AIzaSy...') 与 Moonshot 供应商/端点不匹配 (必须以 'sk-' 开头)",
            )
        if not clean_key.startswith("sk-"):
            return False, "月之暗面 (Moonshot) API Key 格式错误：必须以 'sk-' 开头"
        if len(clean_key) < 15:
            return False, "Moonshot API Key 长度不足"
        return True, "OK"

    if effective_prov == "zhipu":
        if "." not in clean_key and not clean_key.startswith("sk-"):
            return (
                False,
                "智谱 GLM API Key 格式错误：标准 Key 应包含 '.' 或以 'sk-' 开头",
            )
        return True, "OK"

    # 3. 未指定服务商且 URL 为通用/自定义代理端点
    if clean_key.startswith("AIzaSy"):
        if url_str and any(
            kw in url_str.lower()
            for kw in ["api.openai.com", "api.deepseek.com", "aliyuncs.com"]
        ):
            return False, f"Gemini 格式 Key ('AIzaSy...') 与端点 '{url_str}' 不匹配"
        return True, "OK"

    if clean_key.startswith("sk-") and len(clean_key) >= 15:
        if url_str and "googleapis.com" in url_str.lower():
            return (
                False,
                f"OpenAI 格式 Key ('sk-...') 与 Google Gemini 端点 '{url_str}' 不匹配",
            )
        return True, "OK"

    if clean_key.lower() == "ollama":
        return True, "OK"

    if len(clean_key) >= 6:
        return True, "OK"

    return False, "API Key 格式未知且不符合通用标准"


def is_valid_api_key(
    key: str | None,
    provider: str | None = None,
    base_url: str | None = None,
) -> bool:
    """检查 API Key 格式、供应商 (provider) 以及对应 URL (base_url) 是否有效且互相匹配。"""
    is_valid, reason = check_api_credential_match(
        key, provider=provider, base_url=base_url
    )
    if not is_valid and key and key.strip().lower() not in PLACEHOLDER_KEYS:
        logger.warning("LLM API 凭证校验不通过: %s", reason)
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
    """根据传入参数、服务商预设及环境变量解析出最终的 LLM 配置。

    配置优先级：显式入参 > 服务商预设 > 环境变量 > 默认兜底。
    """
    resolved_provider = (
        provider or os.getenv("LLM_PROVIDER") or ""
    ).strip().lower() or None
    preset = PROVIDER_PRESETS.get(resolved_provider) if resolved_provider else None

    # 1. 判定是否激活 LLM (LLM_ENABLED)
    # 优先级：显式入参 enabled > 显式入参 api_key > 环境变量 LLM_ENABLED / ENABLE_LLM > 默认 False
    env_enabled_str = os.getenv("LLM_ENABLED") or os.getenv("ENABLE_LLM")
    if enabled is not None:
        resolved_enabled = enabled
    elif api_key is not None:
        resolved_enabled = True
    elif env_enabled_str is not None:
        resolved_enabled = is_truthy(env_enabled_str)
    else:
        # 当未配置 LLM_ENABLED 且未显式传入 api_key 时，默认 False（云端部署 / GitHub Actions 默认纯启发式运行）
        resolved_enabled = False

    # 2. 提取 Base URL
    # 优先级：
    # a. 显式入参 base_url
    # b. 若显式指定了 provider 入参，优先采用该 provider 的预设 base_url
    # c. 环境变量 LLM_BASE_URL / OPENAI_BASE_URL
    # d. 环境变量推断的 provider 预设 base_url
    # e. 兜底默认 https://api.openai.com/v1
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

    # 3. 提取 API Key
    resolved_key = (
        api_key
        or os.getenv("LLM_API_KEY")
        or (os.getenv("GEMINI_API_KEY") if resolved_provider == "gemini" else None)
        or (os.getenv("DEEPSEEK_API_KEY") if resolved_provider == "deepseek" else None)
        or (os.getenv("DASHSCOPE_API_KEY") if resolved_provider == "qwen" else None)
        or os.getenv("OPENAI_API_KEY")
    )
    if resolved_key:
        resolved_key = resolved_key.strip()
        # 模板占位符统一视为未提供
        if resolved_key.lower() in PLACEHOLDER_KEYS:
            resolved_key = None

    # 特殊规则：本地 Ollama 默认无需真实密钥，填入占位串以满足 OpenAI SDK 要求
    if (
        resolved_provider == "ollama" or "11434" in resolved_base_url
    ) and not resolved_key:
        resolved_key = "ollama"

    # 4. 提取模型名称
    # 优先级与 Base URL 一致：显式入参 > 显式 provider 预设 > 环境变量 > 环境 provider 预设 > 兜底
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

    # 5. 温度、超时与重试
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
    """构建通用的 LangChain ChatModel 实例。

    本函数采用标准通用的 OpenAI 兼容格式 (`ChatOpenAI`)。
    适用于包括 OpenAI (GPT 系列)、Gemini (经由 Google OpenAI 兼容端点)、
    DeepSeek、Qwen、Ollama 本地服务等在内的所有提供商。

    Args:
        model: 模型名称（如 'gpt-4o-mini', 'gemini-2.0-flash', 'deepseek-chat', 'qwen-plus' 等）。
        temperature: 采样温度（0.0 ~ 1.0）。
        base_url: 兼容的 API 端点 URL（例如 'https://api.deepseek.com/v1'）。
        api_key: API 密钥。若不传则从环境变量解析。
        provider: 简写服务商名（如 'openai', 'gemini', 'deepseek', 'qwen', 'ollama'）。
        timeout: 请求超时时间。
        max_retries: 请求失败重试次数。
        enabled: 显式指定是否激活 LLM（若为 False 则直接返回 None 回退启发式引擎）。
        strict: 若为 True 且凭证无效或未启用时抛出 ValueError；若为 False 则返回 None。
        **extra_kwargs: 传递给底层 ChatOpenAI 的额外参数。

    Returns:
        BaseChatModel 实例，若未启用或凭证校验失败且 strict=False 则返回 None。
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

    # 1. 检查总开关：未启用 LLM 时直接返回 None（运行于启发式规则引擎）
    if not config.enabled:
        logger.info(
            "LLM 处于未启用状态 (LLM_ENABLED=false)，系统运行于纯本地启发式规则引擎模式。"
        )
        if strict:
            raise ValueError("LLM 未启用 (LLM_ENABLED=false)")
        return None

    # 2. 已激活 LLM (LLM_ENABLED=true)，严格校验 API Key 格式、服务商与端点 URL 匹配
    is_valid, reason = check_api_credential_match(
        config.api_key,
        provider=config.provider,
        base_url=config.base_url,
    )
    if not is_valid:
        msg = (
            f"已配置激活 LLM (LLM_ENABLED=true)，但 API Key、服务商与端点 URL 校验未通过: {reason}。"
            "系统将自动平滑降级为本地启发式规则引擎以保证服务连续可用。"
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
        return None

    # 3. 正常构建通用 ChatOpenAI 客户端
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
            "成功初始化通用 LLM 实例: provider=%s, model=%s, base_url=%s",
            config.provider or "generic/openai-compatible",
            config.model,
            config.base_url,
        )
        return chat_model
    except Exception as exc:
        logger.error(
            "初始化通用 LLM 实例失败: %s。系统自动回退至本地启发式规则引擎。", exc
        )
        if strict:
            raise
        return None


def get_llm_info(
    llm: Any | None = None,
    config: LLMConfig | None = None,
) -> dict[str, Any]:
    """获取当前 LLM 的运行状态与展示信息（用于 Web UI 或日志呈现）。"""
    cfg = config or resolve_llm_config()
    if llm is None:
        if not cfg.enabled:
            return {
                "is_active": False,
                "status_code": "DISABLED",
                "provider": "Rule-Based",
                "model": "本地启发式规则引擎 (离线/云端部署模式)",
                "base_url": "N/A",
                "hint": "LLM 未启用 (LLM_ENABLED=false)。云端部署默认纯启发式，本地可在 .env 中设置 LLM_ENABLED=true 开启大模型。",
            }

        is_match, reason = check_api_credential_match(
            cfg.api_key, provider=cfg.provider, base_url=cfg.base_url
        )
        hint_msg = (
            f"已激活 LLM 但凭证校验未通过 ({reason})，已自动降级为启发式规则引擎。"
            if not is_match
            else "已激活 LLM 但服务未就绪，已自动降级为启发式规则引擎。"
        )
        return {
            "is_active": False,
            "status_code": "FALLBACK",
            "provider": "Rule-Based",
            "model": "本地启发式规则引擎 (自动降级模式)",
            "base_url": "N/A",
            "hint": hint_msg,
        }

    model_name = getattr(llm, "model_name", getattr(llm, "model", "Unknown Model"))
    base_url = str(getattr(llm, "openai_api_base", getattr(llm, "base_url", "")))

    # 推断服务商展示名
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
        "hint": f"已成功接入 {provider_name} 模型: {model_name}",
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
