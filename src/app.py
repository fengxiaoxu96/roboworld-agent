"""Streamlit 交互式问答界面与文献库可视化系统 (Task 5.2).

提供：
1. 知识库文献的可视化检索与多维标签筛选（模型范式、Action-conditioned、时间等）；
2. 基于 Qdrant 向量检索的 RAG 智能问答对话框，支持多篇世界模型文献横向对比；
3. 输出答案具备严谨学术论述与完整的原始文献引用（arXiv ID、标题、PDF 链接）。
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st

# 将项目根目录 (src 的上一级) 加入模块搜索路径
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.core.llm import get_llm, get_llm_info
from src.core.schema import ModelParadigm, PaperInsight, RawPaper
from src.storage.db_manager import DEFAULT_SQLITE_PATH, DBManager
from src.storage.vector_store import (
    DEFAULT_QDRANT_STORAGE_PATH,
    VectorSearchResult,
    VectorStore,
)

logger = logging.getLogger(__name__)

# 支持的模型范式候选
AVAILABLE_PARADIGMS: list[ModelParadigm] = [
    "Diffusion-based",
    "Transformer/AutoRegressive",
    "Flow-Matching",
    "State-Space/Mamba",
    "Other",
]

# 多语言界面文本映射表 (i18n)
TRANSLATIONS: dict[str, dict[str, str]] = {
    "zh": {
        "page_title": "RoboWorld Agent - 具身机器人世界模型文献知识库",
        "app_title": "🤖 RoboWorld Agent - 具身世界模型前沿学术平台",
        "app_desc": "专为具身智能与机器人世界模型领域打造的自动化文献追踪、结构化研判与交互式对比问答系统。",
        "storage_error": "连接存储底座失败: {error}",
        "sidebar_caption": "具身机器人世界模型文献知识库与 RAG 问答平台",
        "stats_title": "### 📊 知识库统计",
        "metric_total": "总文献数",
        "metric_wm": "世界模型",
        "metric_action": "🎯 动作闭环 (Action-cond): **{count}** 篇",
        "filter_title": "### 🔍 文献筛选与检索",
        "search_label": "🔎 关键词检索 (标题 / arXiv ID)",
        "search_placeholder": "例如 Flow Matching 或 2609...",
        "paradigm_label": "🏷️ 模型范式筛选",
        "paradigm_help": "选择一个或多个模型范式进行过滤，留空显示全部",
        "action_only_label": "🎮 仅显示动作条件 (Action-conditioned)",
        "action_only_help": "仅保留显式输入机器人动作条件的具身世界模型文献",
        "wm_only_label": "🌐 仅展示确认世界模型 (WM Only)",
        "wm_only_help": "过滤掉初筛未通过或非世界模型论文",
        "sample_title": "### 💡 快捷提问示例",
        "clear_chat": "🗑️ 清空问答历史",
        "mgmt_title": "### ⚙️ 文献管理",
        "delete_input_label": "🗑️ 输入 arXiv ID 快速删除",
        "delete_input_placeholder": "例如 2405.00001",
        "delete_btn": "🗑️ 确认删除",
        "delete_success": "✅ 已成功从知识库删除文献 {arxiv_id}",
        "delete_not_found": "⚠️ 未在数据库或向量库中找到文献 {arxiv_id}",
        "delete_card_btn": "🗑️ 删除",
        "delete_card_help": "从知识库和向量库中同步移除此文献",
        "select_all": "全选",
        "deselect_all": "清空选择",
        "batch_delete_btn": "🗑️ 批量删除 ({count})",
        "batch_delete_btn_empty": "🗑️ 批量删除",
        "batch_delete_help": "批量删除所有勾选的文献",
        "batch_delete_success": "✅ 已成功批量删除 {count} 篇文献",
        "batch_delete_warning": "⚠️ 请先勾选需要删除的文献",
        "selected_info": "已选择 {selected} / {total} 篇",
        "select_paper_help": "勾选以进行批量操作",
        "tab_chat": "💬 交互式学术问答 (RAG)",
        "tab_library": "📚 知识库文献库",
        "confirm_delete_dialog_title": "🗑️ 确认删除文献",
        "confirm_delete_single_prompt": "确定要删除文献 **{title}** (`arXiv:{arxiv_id}`) 吗？",
        "confirm_delete_batch_prompt": "确定要批量删除当前选中的 **{count}** 篇文献吗？",
        "confirm_delete_warning": "⚠️ 此操作将从 SQLite 数据库和 Qdrant 向量库中永久移除相关文献与切片索引，不可撤销。",
        "confirm_delete_btn": "🗑️ 确认删除",
        "cancel_delete_btn": "❌ 取消",
        "pending_delete_list": "查看待删除文献清单",
        "library_header": "📚 已收录文献列表（当前筛选展示 {count} 篇）",
        "library_empty": "⚠️ 当前筛选条件下暂未找到匹配的文献，请调整侧边栏的过滤参数。",
        "badge_wm": "🌐 World Model",
        "badge_general": "📄 General",
        "badge_action": "✅ Action-conditioned",
        "badge_no_action": "❌ No Action",
        "eval_envs": "评测环境",
        "core_innovation": "核心创新",
        "no_innovation": "暂无结构化洞察",
        "view_meta": "🔍 查看完整元数据与摘要",
        "chat_header": "💬 机器人世界模型知识库交互式问答 (RAG)",
        "chat_caption": "依托 Qdrant 向量语义检索与论文切片底座，支持针对世界模型方法的多篇文献横向对比、机制剖析与溯源引用。",
        "chat_welcome": (
            "你好！我是 **RoboWorld Agent** 学术助理。\n\n"
            "我可以基于知识库中收录的机器人世界模型论文为您提供深度问答与横向对比分析，例如：\n"
            "- *“对比基于 Flow Matching 与扩散架构的世界模型方法”*\n"
            "- *“总结显式接收 Action 输入并完成闭环规划的代表性模型”*\n"
            "- *“CALVIN 环境下目前主流的世界模型有哪些？”*\n\n"
            "请在下方输入您的问题开始探索！"
        ),
        "chat_input_placeholder": "提出关于世界模型的问题，例如：对比上周入库的使用 Flow Matching 的世界模型方法...",
        "chat_spinner": "🔍 正在检索向量知识库并进行横向对比分析...",
        "llm_status_title": "### 🧠 LLM 引擎状态",
        "llm_active_badge": "🟢 当前模型: **{model}** ({provider})",
        "llm_inactive_badge": "⚪ 启发式规则引擎 (离线/云端部署模式)",
        "llm_fallback_badge": "🟡 启发式规则引擎 (自动降级模式)",
        "llm_endpoint_caption": "端点: `{base_url}`",
        "llm_hint_caption": "当前未激活 LLM (LLM_ENABLED=false)。云端部署默认纯启发式；本地可在 `.env` 中设置 `LLM_ENABLED=true` 并配置 Key 开启。",
        "llm_fallback_hint": "已配置 `LLM_ENABLED=true` 但未检测到有效 Key，已自动降级为启发式规则引擎。请检查 `.env` 配置。",
    },
    "en": {
        "page_title": "RoboWorld Agent - Embodied World Model Literature Base",
        "app_title": "🤖 RoboWorld Agent - Embodied World Models Academic Platform",
        "app_desc": "Automated literature tracking, structured analysis, and interactive comparative Q&A system for embodied intelligence and robotics world models.",
        "storage_error": "Failed to connect to storage backends: {error}",
        "sidebar_caption": "Embodied Robotics World Model Knowledge Base & RAG Platform",
        "stats_title": "### 📊 Knowledge Base Statistics",
        "metric_total": "Total Papers",
        "metric_wm": "World Models",
        "metric_action": "🎯 Action-conditioned: **{count}** papers",
        "filter_title": "### 🔍 Paper Filtering & Search",
        "search_label": "🔎 Keyword Search (Title / arXiv ID)",
        "search_placeholder": "e.g. Flow Matching or 2609...",
        "paradigm_label": "🏷️ Model Paradigm Filter",
        "paradigm_help": "Select one or more model paradigms to filter, leave empty for all",
        "action_only_label": "🎮 Action-conditioned Only",
        "action_only_help": "Only retain embodied world model papers with explicit robot action inputs",
        "wm_only_label": "🌐 World Models Only (WM Only)",
        "wm_only_help": "Filter out papers that did not pass screening or are non-world-models",
        "sample_title": "### 💡 Sample Questions",
        "clear_chat": "🗑️ Clear Chat History",
        "mgmt_title": "### ⚙️ Literature Management",
        "delete_input_label": "🗑️ Enter arXiv ID to Delete",
        "delete_input_placeholder": "e.g. 2405.00001",
        "delete_btn": "🗑️ Confirm Delete",
        "delete_success": "✅ Successfully deleted paper {arxiv_id} from knowledge base",
        "delete_not_found": "⚠️ Paper {arxiv_id} not found in database or vector store",
        "delete_card_btn": "🗑️ Delete",
        "delete_card_help": "Remove this paper from knowledge base and vector store",
        "select_all": "Select All",
        "deselect_all": "Deselect All",
        "batch_delete_btn": "🗑️ Batch Delete ({count})",
        "batch_delete_btn_empty": "🗑️ Batch Delete",
        "batch_delete_help": "Batch delete all selected papers",
        "batch_delete_success": "✅ Successfully batch deleted {count} paper(s)",
        "batch_delete_warning": "⚠️ Please select at least one paper to delete",
        "selected_info": "Selected {selected} / {total} paper(s)",
        "select_paper_help": "Select for batch action",
        "tab_chat": "💬 Interactive Academic Q&A (RAG)",
        "tab_library": "📚 Literature Library",
        "confirm_delete_dialog_title": "🗑️ Confirm Literature Deletion",
        "confirm_delete_single_prompt": "Are you sure you want to delete paper **{title}** (`arXiv:{arxiv_id}`)?",
        "confirm_delete_batch_prompt": "Are you sure you want to delete the selected **{count}** papers?",
        "confirm_delete_warning": "⚠️ This operation will permanently remove the paper(s) and their chunks from SQLite and Qdrant.",
        "confirm_delete_btn": "🗑️ Confirm Delete",
        "cancel_delete_btn": "❌ Cancel",
        "pending_delete_list": "View papers to be deleted",
        "library_header": "📚 Literature Library (Displaying {count} papers matching filter)",
        "library_empty": "⚠️ No matching papers found under current filter criteria. Please adjust sidebar filters.",
        "badge_wm": "🌐 World Model",
        "badge_general": "📄 General",
        "badge_action": "✅ Action-conditioned",
        "badge_no_action": "❌ No Action",
        "eval_envs": "Evaluation Environments",
        "core_innovation": "Core Innovation",
        "no_innovation": "No structured insight available",
        "view_meta": "🔍 View Full Metadata & Abstract",
        "chat_header": "💬 Robotics World Model Interactive Q&A (RAG)",
        "chat_caption": "Powered by Qdrant vector semantic search, supporting comparative analysis, mechanistic dissection, and literature citations.",
        "chat_welcome": (
            "Hello! I am the **RoboWorld Agent** Academic Assistant.\n\n"
            "I can provide deep Q&A and cross-paper comparative analysis based on robotics world model literature in the knowledge base, for example:\n"
            "- *“Compare Flow Matching and diffusion-based world model methods”*\n"
            "- *“Summarize representative models that explicitly receive Action inputs for closed-loop planning”*\n"
            "- *“What are the mainstream world models evaluated on the CALVIN benchmark?”*\n\n"
            "Please enter your question below to start exploring!"
        ),
        "chat_input_placeholder": "Ask a question about world models, e.g.: Compare recently added Flow Matching world models...",
        "chat_spinner": "🔍 Searching vector knowledge base and performing comparative analysis...",
        "llm_status_title": "### 🧠 LLM Engine Status",
        "llm_active_badge": "🟢 Active Model: **{model}** ({provider})",
        "llm_inactive_badge": "⚪ Heuristic Engine (Offline / Cloud Deploy Mode)",
        "llm_fallback_badge": "🟡 Heuristic Engine (Auto Fallback Mode)",
        "llm_endpoint_caption": "Endpoint: `{base_url}`",
        "llm_hint_caption": "LLM is disabled (LLM_ENABLED=false). Cloud deploy defaults to heuristic mode; set `LLM_ENABLED=true` in local `.env` to enable.",
        "llm_fallback_hint": "`LLM_ENABLED=true` was set but no valid API Key was detected. Automatically fell back to heuristic engine.",
    },
}

SAMPLE_QUERIES: dict[str, list[str]] = {
    "zh": [
        "对比近期基于 Flow Matching 的世界模型方法",
        "有哪些显式支持 Action-conditioned 的世界模型？",
        "列出在 CALVIN 或 RLBench 基准上评测的代表性工作",
        "扩散架构在机器人世界模型中的主要应用机制是什么？",
    ],
    "en": [
        "Compare recent Flow Matching based world model methods",
        "Which world models explicitly support Action-conditioned inputs?",
        "List representative works evaluated on CALVIN or RLBench benchmarks",
        "What is the primary mechanism of diffusion architectures in robotics world models?",
    ],
}


def get_text(key: str, lang: str = "zh", **kwargs: Any) -> str:
    """获取多语言对应界面的文本，支持占位符格式化。"""
    texts = TRANSLATIONS.get(lang, TRANSLATIONS["zh"])
    template = texts.get(key, TRANSLATIONS["zh"].get(key, key))
    if kwargs:
        return template.format(**kwargs)
    return template


class RAGSynthesizer:
    """RAG 问答与文献对比合成器。"""

    def __init__(
        self,
        vector_store: VectorStore,
        llm: Any | None = None,
        auto_init_llm: bool = True,
    ) -> None:
        """初始化 RAG 合成器。

        Args:
            vector_store: 向量库实例。
            llm: 显式传入的大语言模型实例。若为 None 且 auto_init_llm=True，
                 将调用 get_llm(temperature=0.2) 自动根据环境配置初始化通用 LLM。
            auto_init_llm: 是否自动尝试根据环境配置初始化通用 LLM。
        """
        self.vector_store = vector_store
        if llm is not None:
            self.llm = llm
        elif auto_init_llm:
            self.llm = get_llm(temperature=0.2)
        else:
            self.llm = None

    def search(
        self,
        query: str,
        limit: int = 5,
        paradigm_filter: str | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        """从向量库中检索相关文献片段。"""
        filter_criteria: dict[str, Any] | None = None
        if paradigm_filter and paradigm_filter not in ("全部", "All"):
            filter_criteria = {"model_paradigm": paradigm_filter}

        return self.vector_store.search(
            query=query,
            limit=limit,
            score_threshold=score_threshold,
            filter_criteria=filter_criteria,
        )

    def generate_answer(
        self,
        query: str,
        search_results: list[VectorSearchResult],
        lang: str = "zh",
    ) -> str:
        """根据检索结果生成回答与引用文献。"""
        if not search_results:
            if lang == "en":
                return (
                    "**No highly relevant literature found in the vector knowledge base.**\n\n"
                    "💡 **Suggestions**:\n"
                    "- Adjust query keywords (e.g. try `Flow Matching`, `Diffusion`, `CALVIN`, `Action-conditioned`);\n"
                    "- Relax model paradigm and action conditioning filter criteria in the sidebar;\n"
                    "- Ensure the background pipeline has run and indexed papers into the database."
                )
            return (
                "**未在向量知识库中检索到高相关性的文献。**\n\n"
                "💡 **建议尝试**：\n"
                "- 调整查询关键词（如更换为 `Flow Matching`、`扩散模型`、`CALVIN`、`动作条件` 等）；\n"
                "- 在左侧侧边栏放宽范式与动作条件过滤条件；\n"
                "- 确认后台流水线已运行并完成文献入库。"
            )

        # 若存在配置好的 LLM，则调用 LLM 生成
        if self.llm is not None:
            try:
                return self._generate_with_llm(query, search_results, lang=lang)
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM 生成回答失败，回退到启发式合成引擎: %s", exc)

        # 默认高质量结构化合成引擎（用于离线、免 Key 或单测环境）
        return self._generate_with_heuristics(query, search_results, lang=lang)

    def _generate_with_llm(
        self,
        query: str,
        search_results: list[VectorSearchResult],
        lang: str = "zh",
    ) -> str:
        """调用外部 LLM 完成 RAG 合成与多篇对比。"""
        context_blocks: list[str] = []
        for idx, res in enumerate(search_results, start=1):
            if lang == "en":
                context_blocks.append(
                    f"[Paper {idx}]\n"
                    f"- Title: {res.title}\n"
                    f"- arXiv ID: {res.arxiv_id}\n"
                    f"- Paradigm: {res.model_paradigm}\n"
                    f"- World Model: {'Yes' if res.is_world_model else 'No'}\n"
                    f"- Action-conditioned: {'Yes' if res.action_conditioned else 'No'}\n"
                    f"- Evaluation Envs: {', '.join(res.eval_environments) if res.eval_environments else 'None'}\n"
                    f"- Core Innovation: {res.core_innovation}\n"
                    f"- Context Chunk:\n{res.chunk_text}\n"
                )
            else:
                context_blocks.append(
                    f"【文献 {idx}】\n"
                    f"- 标题: {res.title}\n"
                    f"- arXiv ID: {res.arxiv_id}\n"
                    f"- 模型范式: {res.model_paradigm}\n"
                    f"- 世界模型 (World Model): {'是' if res.is_world_model else '否'}\n"
                    f"- 动作条件 (Action-conditioned): {'是' if res.action_conditioned else '否'}\n"
                    f"- 评测环境: {', '.join(res.eval_environments) if res.eval_environments else '未提及'}\n"
                    f"- 核心创新: {res.core_innovation}\n"
                    f"- 相关文本切片:\n{res.chunk_text}\n"
                )

        context_str = "\n".join(context_blocks)
        if lang == "en":
            prompt = (
                "You are an academic researcher specializing in robotics and embodied world models.\n"
                "Based on the following retrieved literature snippets, provide a professional, comprehensive, and in-depth answer to the user query.\n"
                "If the query involves comparison or contrasting, conduct a rigorous comparative analysis across Model Paradigms, World Model Attributes, "
                "Action Conditioning, Evaluation Benchmarks, and Core Innovations, and summarize them in a markdown table.\n"
                "You must append a standard [References] citation list at the end with paper titles, arXiv IDs, and URLs.\n\n"
                f"[Retrieved Context]:\n{context_str}\n\n"
                f"[User Query]: {query}\n"
            )
        else:
            prompt = (
                "你是一名资深的具身智能与机器人学术前沿领域专家。\n"
                "请基于以下提供的检索文献片段，针对用户的问题给出专业、全面、深度的分析。\n"
                "如果用户提到了对比/比较，请从【模型范式】、【世界模型属性】、【动作闭环控制 (Action-conditioned)】、"
                "【评测基准环境】及【核心架构与创新机制】等维度进行严谨横向对比，并整理对比表格。\n"
                "回答最后必须附带标准的【📚 参考文献引用】列表（包含论文题目、arXiv ID 及链接）。\n\n"
                f"【检索上下文】:\n{context_str}\n\n"
                f"【用户问题】: {query}\n"
            )

        assert self.llm is not None
        response = self.llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        if isinstance(content, list):
            content = "\n".join(str(c) for c in content)
        return str(content)

    def _generate_with_heuristics(
        self,
        query: str,
        search_results: list[VectorSearchResult],
        lang: str = "zh",
    ) -> str:
        """基于结构化领域知识的本地启发式 RAG 合成引擎。"""
        # 聚合并去重文献
        papers_dict: dict[str, dict[str, Any]] = {}
        for res in search_results:
            if res.arxiv_id not in papers_dict:
                papers_dict[res.arxiv_id] = {
                    "arxiv_id": res.arxiv_id,
                    "title": res.title,
                    "model_paradigm": res.model_paradigm,
                    "action_conditioned": res.action_conditioned,
                    "is_world_model": res.is_world_model,
                    "eval_environments": res.eval_environments,
                    "core_innovation": res.core_innovation,
                    "payload": res.payload,
                    "chunks": [res.chunk_text],
                }
            else:
                papers_dict[res.arxiv_id]["chunks"].append(res.chunk_text)

        unique_papers = list(papers_dict.values())
        is_comparison = any(
            kw in query.lower()
            for kw in [
                "对比",
                "比较",
                "区别",
                "差异",
                "横向",
                "compare",
                "vs",
                "comparison",
                "differences",
                "versus",
            ]
        )

        lines: list[str] = []

        if lang == "en":
            if is_comparison:
                lines.append(
                    f"### 📊 Literature Retrieval & Comparative Analysis (Retrieved {len(unique_papers)} representative papers)\n"
                )
                lines.append(
                    f'Regarding your query **"{query}"**, based on vector semantic retrieval and knowledge base metadata, '
                    "here is a systematic comparative breakdown across architecture paradigms, action conditioning mechanisms, and experimental benchmarks:\n"
                )

                # 横向对比表格
                lines.append(
                    "| Paper ID | Paper Title | Paradigm | Action-conditioned | World Model | Benchmarks | Core Technical Innovation |"
                )
                lines.append("| :--- | :--- | :--- | :---: | :---: | :--- | :--- |")
                for p in unique_papers:
                    action_sym = "✅ Yes" if p["action_conditioned"] else "❌ No"
                    wm_sym = "🌐 Yes" if p.get("is_world_model", True) else "📄 No"
                    envs_str = (
                        ", ".join(p["eval_environments"])
                        if p["eval_environments"]
                        else "None"
                    )
                    title_short = (
                        p["title"][:45] + "..." if len(p["title"]) > 45 else p["title"]
                    )
                    innov_short = (
                        p["core_innovation"][:60] + "..."
                        if len(p["core_innovation"]) > 60
                        else (p["core_innovation"] or "N/A")
                    )
                    lines.append(
                        f"| [`{p['arxiv_id']}`](https://arxiv.org/abs/{p['arxiv_id']}) "
                        f"| {title_short} "
                        f"| `{p['model_paradigm']}` "
                        f"| {action_sym} "
                        f"| {wm_sym} "
                        f"| {envs_str} "
                        f"| {innov_short} |"
                    )
                lines.append("\n#### 🔬 Core Technical Dimension Insights:")

                # 范式分布剖析
                paradigms_found = {p["model_paradigm"] for p in unique_papers}
                lines.append(
                    f"1. **Dynamics Modeling Architecture**: Retrieved papers span {', '.join(f'`{pm}`' for pm in paradigms_found)} architectural paradigms."
                )
                if "Flow-Matching" in paradigms_found:
                    lines.append(
                        "   - *Flow Matching Family*: Utilizes Continuous Normalizing Flows (CNF) to optimize ODE trajectories, demonstrating high physical prediction fidelity and reduced inference steps."
                    )
                if "Diffusion-based" in paradigms_found:
                    lines.append(
                        "   - *Diffusion-based Family*: Models multimodal future state distributions via denoising diffusion, effectively handling contact-rich and physically uncertain environments."
                    )
                if "Transformer/AutoRegressive" in paradigms_found:
                    lines.append(
                        "   - *Transformer/AutoRegressive Family*: Discretizes spatiotemporal states into tokens and predicts next tokens via causal attention, providing strong sequential context modeling."
                    )

                # 动作条件剖析
                action_papers = [p for p in unique_papers if p["action_conditioned"]]
                lines.append(
                    f"2. **Action Conditioning Support**: {len(action_papers)}/{len(unique_papers)} papers explicitly accept robot action inputs, "
                    "directly applicable to imagination-based policy optimization or model predictive control (MPC)."
                )

                # 评测环境剖析
                all_envs = sorted(
                    {env for p in unique_papers for env in p["eval_environments"]}
                )
                if all_envs:
                    lines.append(
                        f"3. **Primary Evaluation Benchmarks**: Evaluated on benchmark environments including {', '.join(f'`{e}`' for e in all_envs)}."
                    )
            else:
                lines.append(
                    f"### 💡 Literature Analysis & Paper Summary ({len(unique_papers)} relevant papers)\n"
                )
                lines.append(
                    f'Regarding your question **"{query}"**, based on knowledge base literature analysis, here are the key highlights:\n'
                )
                for idx, p in enumerate(unique_papers, start=1):
                    action_desc = (
                        "Explicit robot action input (Action-conditioned)"
                        if p["action_conditioned"]
                        else "Non-action-conditioned"
                    )
                    envs_desc = (
                        f", evaluated on {', '.join(p['eval_environments'])}"
                        if p["eval_environments"]
                        else ""
                    )
                    wm_desc = (
                        "World Model"
                        if p.get("is_world_model", True)
                        else "General Literature"
                    )
                    innov = p["core_innovation"] or "Not explicitly specified"
                    lines.append(
                        f"**{idx}. [{p['title']}](https://arxiv.org/abs/{p['arxiv_id']})**\n"
                        f"- **Paradigm & Category**: `{p['model_paradigm']}` ({wm_desc}), {action_desc}{envs_desc}.\n"
                        f"- **Core Innovation**: {innov}\n"
                    )

            # References
            lines.append("\n---\n### 📚 Original Literature Citations")
            for idx, p in enumerate(unique_papers, start=1):
                pdf_url = p["payload"].get(
                    "pdf_url", f"https://arxiv.org/pdf/{p['arxiv_id']}"
                )
                wm_tag = "Yes" if p.get("is_world_model", True) else "No"
                lines.append(
                    f"{idx}. **[{p['title']}](https://arxiv.org/abs/{p['arxiv_id']})**  \n"
                    f"   - **arXiv ID**: `{p['arxiv_id']}` | **Paradigm**: `{p['model_paradigm']}` | "
                    f"**World Model**: `{wm_tag}` | **Action-conditioned**: {'✅ Yes' if p['action_conditioned'] else '❌ No'}  \n"
                    f"   - **Evaluation Environments**: {', '.join(p['eval_environments']) if p['eval_environments'] else 'None'}  \n"
                    f"   - **Full PDF**: [{pdf_url}]({pdf_url})"
                )
            return "\n".join(lines)

        # Chinese (default)
        if is_comparison:
            lines.append(
                f"### 📊 检索与横向对比分析（共召回 {len(unique_papers)} 篇代表性文献）\n"
            )
            lines.append(
                f"针对您提出的问题 **“{query}”**，基于向量语义检索与知识库元数据，"
                "我们对相关代表性文献的架构范式、动作调节机制与实验评测展开系统性横向对比：\n"
            )

            # 横向对比表格
            lines.append(
                "| 论文编号 | 论文标题 | 模型范式 | 动作闭环 (Action-cond) | 世界模型 (WM) | 评测基准环境 | 核心技术创新点 |"
            )
            lines.append("| :--- | :--- | :--- | :---: | :---: | :--- | :--- |")
            for p in unique_papers:
                action_sym = "✅ 显式支持" if p["action_conditioned"] else "❌ 否"
                wm_sym = "🌐 是" if p.get("is_world_model", True) else "📄 否"
                envs_str = (
                    ", ".join(p["eval_environments"])
                    if p["eval_environments"]
                    else "None"
                )
                title_short = (
                    p["title"][:45] + "..." if len(p["title"]) > 45 else p["title"]
                )
                innov_short = (
                    p["core_innovation"][:60] + "..."
                    if len(p["core_innovation"]) > 60
                    else (p["core_innovation"] or "未提供")
                )
                lines.append(
                    f"| [`{p['arxiv_id']}`](https://arxiv.org/abs/{p['arxiv_id']}) "
                    f"| {title_short} "
                    f"| `{p['model_paradigm']}` "
                    f"| {action_sym} "
                    f"| {wm_sym} "
                    f"| {envs_str} "
                    f"| {innov_short} |"
                )
            lines.append("\n#### 🔬 核心技术维度对比维度洞察：")

            # 范式分布剖析
            paradigms_found = {p["model_paradigm"] for p in unique_papers}
            lines.append(
                f"1. **模型动力学建模架构**：本次检索文献涵盖了 {', '.join(f'`{pm}`' for pm in paradigms_found)} 架构范式。"
            )
            if "Flow-Matching" in paradigms_found:
                lines.append(
                    "   - *Flow Matching 系列*：采用连续归一化流 (CNF) 优化常微分方程 (ODE) 轨迹，在物理预测保真度与推断步数压缩上表现突出。"
                )
            if "Diffusion-based" in paradigms_found:
                lines.append(
                    "   - *Diffusion-based 系列*：借助去噪扩散概率机制刻画未来状态的多模态分布，能较好地应对接触丰富与物理不确定性场景。"
                )
            if "Transformer/AutoRegressive" in paradigms_found:
                lines.append(
                    "   - *Transformer/自回归系列*：将时空状态标记离散化后通过因果注意力预测下一个 Token，具备出色的序列上下文建模能力。"
                )

            # 动作条件剖析
            action_papers = [p for p in unique_papers if p["action_conditioned"]]
            lines.append(
                f"2. **动作闭环支持 (Action-conditioned)**：共有 {len(action_papers)}/{len(unique_papers)} 篇文献显式接收机器人动作输入，"
                "可直接部署于基于想象的策略优化 (Imagination-based Policy Optimization) 或模型预测控制 (MPC)。"
            )

            # 评测环境剖析
            all_envs = sorted(
                {env for p in unique_papers for env in p["eval_environments"]}
            )
            if all_envs:
                lines.append(
                    f"3. **主要验证基准**：主要涵盖了 {', '.join(f'`{e}`' for e in all_envs)} 等具身智能标杆环境。"
                )

        else:
            lines.append(
                f"### 💡 学术问答分析与文献总结（相关文献 {len(unique_papers)} 篇）\n"
            )
            lines.append(
                f"针对您关心的 **“{query}”**，基于知识库文献切片分析，核心要点如下：\n"
            )
            for idx, p in enumerate(unique_papers, start=1):
                action_desc = (
                    "具备显式动作输入能力（Action-conditioned）"
                    if p["action_conditioned"]
                    else "非动作显式条件"
                )
                envs_desc = (
                    f"，验证基准包括 {', '.join(p['eval_environments'])}"
                    if p["eval_environments"]
                    else ""
                )
                wm_desc = "世界模型" if p.get("is_world_model", True) else "通用文献"
                innov = p["core_innovation"] or "未明确注明"
                lines.append(
                    f"**{idx}. [{p['title']}](https://arxiv.org/abs/{p['arxiv_id']})**\n"
                    f"- **范式与类别**：采用 `{p['model_paradigm']}` ({wm_desc}), {action_desc}{envs_desc}。\n"
                    f"- **核心创新机制**：{innov}\n"
                )

        # 规范的引用文献清单
        lines.append("\n---\n### 📚 原始文献引用列表")
        for idx, p in enumerate(unique_papers, start=1):
            pdf_url = p["payload"].get(
                "pdf_url", f"https://arxiv.org/pdf/{p['arxiv_id']}"
            )
            wm_tag = "是" if p.get("is_world_model", True) else "否"
            lines.append(
                f"{idx}. **[{p['title']}](https://arxiv.org/abs/{p['arxiv_id']})**  \n"
                f"   - **arXiv 编号**: `{p['arxiv_id']}` | **模型范式**: `{p['model_paradigm']}` | "
                f"**世界模型**: `{wm_tag}` | **动作条件**: {'✅ 是' if p['action_conditioned'] else '❌ 否'}  \n"
                f"   - **评测环境**: {', '.join(p['eval_environments']) if p['eval_environments'] else '无'}  \n"
                f"   - **PDF 全文**: [{pdf_url}]({pdf_url})"
            )

        return "\n".join(lines)


def sync_db_to_vector_store(db: DBManager, vector_store: VectorStore) -> int:
    """确保 SQLite 数据库中所有已落库的论文均在 Qdrant 向量库中建立切片索引。

    对于已有切片的论文将自动幂等跳过，确保 RAG 检索能覆盖数据库中所有已收录文献。
    """
    papers = db.list_papers(limit=1000)
    synced_count = 0
    for p_record in papers:
        arxiv_id = str(p_record["arxiv_id"])
        if not vector_store.paper_exists(arxiv_id):
            pub_date_raw = str(p_record.get("published_date", ""))
            try:
                pub_date = datetime.fromisoformat(pub_date_raw)
            except Exception:  # noqa: BLE001
                pub_date = datetime.now(UTC)

            paper = RawPaper(
                arxiv_id=arxiv_id,
                title=str(p_record.get("title", "")),
                authors=[],
                abstract=str(p_record.get("title", "")),
                categories=[],
                published_date=pub_date,
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            )
            insight = db.get_paper_insight(arxiv_id)
            if insight is None:
                insight = PaperInsight(
                    is_world_model=bool(p_record.get("is_wm", 0)),
                    model_paradigm="Other",
                    action_conditioned=False,
                    eval_environments=[],
                    core_innovation="",
                )
            if vector_store.add_paper(paper=paper, insight=insight):
                synced_count += 1
    return synced_count


@st.cache_resource
def get_storage_managers(
    sqlite_path: str | None = None,
    qdrant_path: str | None = None,
) -> tuple[DBManager, VectorStore]:
    """初始化并缓存 SQLite 与 Qdrant 存储管理器，并同步全库文献至向量库。"""
    actual_sqlite = sqlite_path or os.getenv("SQLITE_DB_PATH") or DEFAULT_SQLITE_PATH
    actual_qdrant = (
        qdrant_path or os.getenv("QDRANT_STORAGE_PATH") or DEFAULT_QDRANT_STORAGE_PATH
    )
    db_manager = DBManager(db_path=actual_sqlite)
    vector_store = VectorStore(storage_path=actual_qdrant)
    sync_db_to_vector_store(db_manager, vector_store)
    return db_manager, vector_store


def delete_paper_record(
    arxiv_id: str,
    db_manager: DBManager,
    vector_store: VectorStore,
) -> bool:
    """从 SQLite 元数据库与 Qdrant 向量库中同步删除指定论文及其切片索引。

    Returns:
        bool: 若在任意一个存储底座中成功删除记录则返回 True。
    """
    clean_id = arxiv_id.strip()
    db_ok = db_manager.delete_paper(clean_id)
    vs_ok = vector_store.delete_paper(clean_id)
    logger.info("同步删除论文 %s 结果: SQLite=%s, Qdrant=%s", clean_id, db_ok, vs_ok)
    return db_ok or vs_ok


def batch_delete_paper_records(
    arxiv_ids: Sequence[str],
    db_manager: DBManager,
    vector_store: VectorStore,
) -> int:
    """从 SQLite 元数据库与 Qdrant 向量库中批量同步删除指定论文列表及其切片索引。

    Returns:
        int: 实际成功删除的记录条数。
    """
    if not arxiv_ids:
        return 0
    clean_ids = [aid.strip() for aid in arxiv_ids if aid.strip()]
    if not clean_ids:
        return 0
    db_count = db_manager.batch_delete_papers(clean_ids)
    vs_count = vector_store.batch_delete_papers(clean_ids)
    logger.info(
        "批量删除论文 %s 结果: SQLite=%d, Qdrant=%d",
        clean_ids,
        db_count,
        vs_count,
    )
    return max(db_count, vs_count)


def parse_insight_payload(insight_json_str: str | None) -> dict[str, Any]:
    """安全解析 SQLite 中存储的 insight_json。"""
    if not insight_json_str:
        return {}
    try:
        return json.loads(insight_json_str)  # type: ignore[no-any-return]
    except Exception:  # noqa: BLE001
        return {}


def filter_papers_data(
    papers: list[dict[str, Any]],
    search_keyword: str = "",
    selected_paradigms: list[str] | None = None,
    action_conditioned_only: bool = False,
    world_model_only: bool = True,
) -> list[dict[str, Any]]:
    """根据多维筛选项过滤已加载的论文列表。"""
    filtered: list[dict[str, Any]] = []
    keyword_clean = search_keyword.strip().lower()

    for item in papers:
        # 世界模型过滤
        is_wm = bool(item.get("is_wm", 0))
        if world_model_only and not is_wm:
            continue

        insight = parse_insight_payload(item.get("insight_json"))

        # 范式过滤
        if selected_paradigms:
            paper_paradigm = insight.get("model_paradigm", "Other")
            if paper_paradigm not in selected_paradigms:
                continue

        # 动作条件过滤
        if action_conditioned_only and not insight.get("action_conditioned", False):
            continue

        # 关键词匹配 (匹配 title 或 arxiv_id)
        if keyword_clean:
            title_text = str(item.get("title", "")).lower()
            arxiv_text = str(item.get("arxiv_id", "")).lower()
            if keyword_clean not in title_text and keyword_clean not in arxiv_text:
                continue

        filtered.append({**item, "parsed_insight": insight})

    return filtered


def render_sidebar(
    all_papers: list[dict[str, Any]],
    lang: str = "zh",
    llm: Any | None = None,
) -> dict[str, Any]:
    """渲染侧边栏统计信息与交互式多维过滤组件。"""
    # 顶部多语言切换下拉框
    lang_options = ["中文", "English"]
    selected_lang_str = st.sidebar.selectbox(
        "🌐 语言 / Language",
        options=lang_options,
        index=0 if lang == "zh" else 1,
        key="language_select",
    )
    current_lang = "en" if selected_lang_str == "English" else "zh"

    st.sidebar.title("🤖 RoboWorld Agent")
    st.sidebar.caption(get_text("sidebar_caption", current_lang))

    # 1. 知识库全局统计指标
    total_count = len(all_papers)
    wm_count = sum(1 for p in all_papers if p.get("is_wm", 0))
    action_cond_count = 0
    for p in all_papers:
        ins = parse_insight_payload(p.get("insight_json"))
        if ins.get("action_conditioned", False):
            action_cond_count += 1

    st.sidebar.markdown(get_text("stats_title", current_lang))
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric(get_text("metric_total", current_lang), total_count)
    with col2:
        st.metric(get_text("metric_wm", current_lang), wm_count)
    # st.sidebar.caption(get_text("metric_action", current_lang, count=action_cond_count))
    st.sidebar.markdown("---")

    # LLM 模型状态展示
    st.sidebar.markdown(get_text("llm_status_title", current_lang))
    llm_info = get_llm_info(llm)
    if llm_info["is_active"]:
        st.sidebar.success(
            get_text(
                "llm_active_badge",
                current_lang,
                model=llm_info["model"],
                provider=llm_info["provider"],
            )
        )
        st.sidebar.caption(
            get_text(
                "llm_endpoint_caption",
                current_lang,
                base_url=llm_info["base_url"],
            )
        )
    elif llm_info.get("status_code") == "FALLBACK":
        st.sidebar.warning(get_text("llm_fallback_badge", current_lang))
        st.sidebar.caption(get_text("llm_fallback_hint", current_lang))
    else:
        st.sidebar.info(get_text("llm_inactive_badge", current_lang))
        st.sidebar.caption(get_text("llm_hint_caption", current_lang))
    st.sidebar.markdown("---")

    # 2. 多维分类标签筛选
    st.sidebar.markdown(get_text("filter_title", current_lang))

    search_keyword = st.sidebar.text_input(
        get_text("search_label", current_lang),
        value="",
        placeholder=get_text("search_placeholder", current_lang),
    )

    selected_paradigms = st.sidebar.multiselect(
        get_text("paradigm_label", current_lang),
        options=AVAILABLE_PARADIGMS,
        default=[],
        help=get_text("paradigm_help", current_lang),
    )

    action_conditioned_only = st.sidebar.checkbox(
        get_text("action_only_label", current_lang),
        value=False,
        help=get_text("action_only_help", current_lang),
    )

    world_model_only = st.sidebar.checkbox(
        get_text("wm_only_label", current_lang),
        value=False,
        help=get_text("wm_only_help", current_lang),
    )

    # st.sidebar.markdown("---")

    # # 3. 示例快捷问答
    # st.sidebar.markdown(get_text("sample_title", current_lang))
    # sample_queries = SAMPLE_QUERIES.get(current_lang, SAMPLE_QUERIES["zh"])
    # selected_sample: str | None = None
    # for idx, sq in enumerate(sample_queries):
    #     if st.sidebar.button(f"📌 {sq}", key=f"sq_btn_{idx}"):
    #         selected_sample = sq

    st.sidebar.markdown("---")
    if st.sidebar.button(
        get_text("clear_chat", current_lang), use_container_width=True
    ):
        st.session_state["messages"] = []
        st.rerun()

    return {
        "search_keyword": search_keyword,
        "selected_paradigms": selected_paradigms,
        "action_conditioned_only": action_conditioned_only,
        "world_model_only": world_model_only,
        # "selected_sample": selected_sample,
        "language": current_lang,
    }


@st.dialog("🗑️ 确认删除文献 / Confirm Deletion")
def render_confirm_delete_dialog(
    arxiv_ids: list[str],
    titles: list[str],
    db_manager: DBManager,
    vector_store: VectorStore,
    lang: str = "zh",
) -> None:
    """弹出删除确认对话框，提供确认与取消选项。"""
    st.subheader(get_text("confirm_delete_dialog_title", lang))
    count = len(arxiv_ids)
    if count == 1:
        st.markdown(
            get_text(
                "confirm_delete_single_prompt",
                lang,
                title=titles[0] if titles else arxiv_ids[0],
                arxiv_id=arxiv_ids[0],
            )
        )
    else:
        st.markdown(get_text("confirm_delete_batch_prompt", lang, count=count))
        with st.expander(get_text("pending_delete_list", lang), expanded=False):
            for aid, t in zip(arxiv_ids, titles, strict=False):
                st.markdown(f"- **{t}** (`{aid}`)")

    st.warning(get_text("confirm_delete_warning", lang))

    col_confirm, col_cancel = st.columns([1, 1])
    with col_confirm:
        if st.button(
            get_text("confirm_delete_btn", lang),
            key="btn_dialog_confirm_delete",
            type="primary",
            use_container_width=True,
        ):
            deleted_cnt = batch_delete_paper_records(
                arxiv_ids, db_manager, vector_store
            )
            for aid in arxiv_ids:
                st.session_state.pop(f"paper_select_{aid}", None)
            st.session_state.pop("delete_dialog_payload", None)
            st.session_state["active_tab_id"] = "library"
            st.toast(get_text("batch_delete_success", lang, count=deleted_cnt))
            st.rerun()

    with col_cancel:
        if st.button(
            get_text("cancel_delete_btn", lang),
            key="btn_dialog_cancel_delete",
            use_container_width=True,
        ):
            st.session_state.pop("delete_dialog_payload", None)
            st.session_state["active_tab_id"] = "library"
            st.rerun()


def render_paper_library_tab(
    filtered_papers: list[dict[str, Any]],
    db_manager: DBManager | None = None,
    vector_store: VectorStore | None = None,
    lang: str = "zh",
) -> None:
    """渲染已收录文献库面板。"""
    st.subheader(get_text("library_header", lang, count=len(filtered_papers)))

    if not filtered_papers:
        st.info(get_text("library_empty", lang))
        return

    # 若未显式传入存储管理器，尝试从全局缓存获取
    if db_manager is None or vector_store is None:
        try:
            cached_db, cached_vs = get_storage_managers()
            db_manager = db_manager or cached_db
            vector_store = vector_store or cached_vs
        except Exception as exc:  # noqa: BLE001
            logger.warning("获取存储管理器失败: %s", exc)

    # 提取当前可见列表中的所有 arXiv ID 与被选中的 ID 列表
    current_paper_ids = [
        item.get("arxiv_id", "") for item in filtered_papers if item.get("arxiv_id")
    ]
    selected_ids = [
        aid
        for aid in current_paper_ids
        if st.session_state.get(f"paper_select_{aid}", False)
    ]

    # 顶部批量操作栏
    col_sel_all, col_desel_all, col_batch_del, col_info = st.columns(
        [1.0, 1.0, 1.8, 2.5]
    )

    with col_sel_all:
        if st.button(
            get_text("select_all", lang),
            key="btn_select_all",
            use_container_width=True,
        ):
            for aid in current_paper_ids:
                st.session_state[f"paper_select_{aid}"] = True
            st.rerun()

    with col_desel_all:
        if st.button(
            get_text("deselect_all", lang),
            key="btn_deselect_all",
            use_container_width=True,
        ):
            for aid in current_paper_ids:
                st.session_state[f"paper_select_{aid}"] = False
            st.rerun()

    with col_batch_del:
        batch_btn_label = (
            get_text("batch_delete_btn", lang, count=len(selected_ids))
            if selected_ids
            else get_text("batch_delete_btn_empty", lang)
        )
        if st.button(
            batch_btn_label,
            key="btn_batch_delete",
            type="primary" if selected_ids else "secondary",
            use_container_width=True,
            help=get_text("batch_delete_help", lang),
        ):
            if not selected_ids:
                st.warning(get_text("batch_delete_warning", lang))
            elif db_manager and vector_store:
                st.session_state["active_tab_id"] = "library"
                target_titles = [
                    p.get("title", p.get("arxiv_id", ""))
                    for p in filtered_papers
                    if p.get("arxiv_id") in selected_ids
                ]
                st.session_state["delete_dialog_payload"] = {
                    "arxiv_ids": selected_ids,
                    "titles": target_titles,
                }

    with col_info:
        st.markdown(
            f"<div style='padding-top: 8px; color: gray;'>"
            f"{get_text('selected_info', lang, selected=len(selected_ids), total=len(filtered_papers))}"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    for item in filtered_papers:
        arxiv_id = item.get("arxiv_id", "")
        title = item.get("title", "")
        pub_date = str(item.get("published_date", ""))[:10]
        insight = item.get("parsed_insight", {})
        is_wm = bool(item.get("is_wm", 0))

        paradigm = insight.get("model_paradigm", "Other")
        action_cond = insight.get("action_conditioned", False)
        envs = insight.get("eval_environments", [])
        innov = insight.get("core_innovation") or get_text("no_innovation", lang)

        action_badge = (
            get_text("badge_action", lang)
            if action_cond
            else get_text("badge_no_action", lang)
        )
        wm_badge = (
            get_text("badge_wm", lang) if is_wm else get_text("badge_general", lang)
        )

        with st.container():
            col_chk, col_card = st.columns([0.05, 0.95])
            with col_chk:
                st.checkbox(
                    "选择",
                    key=f"paper_select_{arxiv_id}",
                    label_visibility="collapsed",
                    help=get_text("select_paper_help", lang),
                )
            with col_card:
                st.markdown(
                    f"#### [{title}](https://arxiv.org/abs/{arxiv_id}) "
                    f"<span style='font-size: 0.8em; color: gray;'>(`arXiv:{arxiv_id}`)</span>",
                    unsafe_allow_html=True,
                )
                col_badges, col_links, col_del = st.columns([2.6, 1.0, 0.6])
                with col_badges:
                    st.markdown(
                        f"`{paradigm}` | `{action_badge}` | `{wm_badge}` | 📅 `{pub_date}`"
                    )
                with col_links:
                    st.markdown(
                        f"[📖 arXiv](https://arxiv.org/abs/{arxiv_id}) &nbsp;|&nbsp; "
                        f"[📥 PDF](https://arxiv.org/pdf/{arxiv_id})"
                    )
                with col_del:
                    if (
                        st.button(
                            get_text("delete_card_btn", lang),
                            key=f"del_btn_{arxiv_id}",
                            help=get_text("delete_card_help", lang),
                        )
                        and db_manager
                        and vector_store
                    ):
                        st.session_state["active_tab_id"] = "library"
                        st.session_state["delete_dialog_payload"] = {
                            "arxiv_ids": [arxiv_id],
                            "titles": [title],
                        }

                if envs:
                    st.markdown(
                        f"**{get_text('eval_envs', lang)}**: "
                        + " ".join([f"`{e}`" for e in envs])
                    )
                st.markdown(f"**{get_text('core_innovation', lang)}**: {innov}")

                with st.expander(get_text("view_meta", lang)):
                    st.json(
                        {
                            "arxiv_id": arxiv_id,
                            "title": title,
                            "published_date": pub_date,
                            "insight": insight,
                        }
                    )
            st.divider()

    # 响应弹出的删除确认对话框
    if "delete_dialog_payload" in st.session_state and db_manager and vector_store:
        payload = st.session_state["delete_dialog_payload"]
        render_confirm_delete_dialog(
            payload["arxiv_ids"],
            payload["titles"],
            db_manager,
            vector_store,
            lang=lang,
        )


def render_chat_tab(
    synthesizer: RAGSynthesizer,
    default_prompt: str | None = None,
    lang: str = "zh",
) -> None:
    """渲染交互式 RAG 问答对话框。"""
    st.subheader(get_text("chat_header", lang))
    st.caption(get_text("chat_caption", lang))

    # 初始化会话状态
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": get_text("chat_welcome", lang),
            }
        ]
    elif (
        len(st.session_state["messages"]) == 1
        and st.session_state["messages"][0]["role"] == "assistant"
    ):
        # 若仅有初始欢迎语，切换语言时同步更新欢迎语语言
        st.session_state["messages"][0]["content"] = get_text("chat_welcome", lang)

    # 渲染历史消息
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 检查是否有侧边栏点击的快捷提问
    active_prompt: str | None = None
    if default_prompt:
        active_prompt = default_prompt

    # 对话输入框
    user_input = st.chat_input(get_text("chat_input_placeholder", lang))
    if user_input:
        active_prompt = user_input

    if active_prompt:
        # 1. 记录并展示用户问题
        st.session_state["messages"].append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        # 2. 检索并生成回答
        with (
            st.chat_message("assistant"),
            st.spinner(get_text("chat_spinner", lang)),
        ):
            search_results = synthesizer.search(query=active_prompt, limit=5)
            answer = synthesizer.generate_answer(
                query=active_prompt,
                search_results=search_results,
                lang=lang,
            )
            st.markdown(answer)

        # 3. 记录助手回答
        st.session_state["messages"].append({"role": "assistant", "content": answer})


def main() -> None:
    """Streamlit 主应用程序入口。"""
    # 读取当前选择的语言偏好 (默认为中文)
    lang_pref = st.session_state.get("language_select", "中文")
    lang = "en" if lang_pref == "English" else "zh"

    st.set_page_config(
        page_title=get_text("page_title", lang),
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 标题与介绍横幅
    st.title(get_text("app_title", lang))
    st.markdown(get_text("app_desc", lang))

    # 获取存储管理器与合成器
    try:
        db_manager, vector_store = get_storage_managers()
        synthesizer = RAGSynthesizer(vector_store=vector_store)
        all_papers = db_manager.list_papers(limit=500)
    except Exception as exc:  # noqa: BLE001
        st.error(get_text("storage_error", lang, error=exc))
        return

    # 渲染侧边栏并获取筛选状态与当前语言
    sidebar_state = render_sidebar(all_papers, lang=lang, llm=synthesizer.llm)
    lang = sidebar_state.get("language", lang)

    # 根据侧边栏选项过滤文献
    filtered_papers = filter_papers_data(
        papers=all_papers,
        search_keyword=sidebar_state["search_keyword"],
        selected_paradigms=sidebar_state["selected_paradigms"],
        action_conditioned_only=sidebar_state["action_conditioned_only"],
        world_model_only=sidebar_state["world_model_only"],
    )

    # 主界面选项卡
    tab_chat_title = get_text("tab_chat", lang)
    tab_library_title = get_text("tab_library", lang)
    tab_titles = [tab_chat_title, tab_library_title]

    if "active_tab_id" not in st.session_state:
        st.session_state["active_tab_id"] = "chat"

    default_tab_title = (
        tab_library_title
        if st.session_state.get("active_tab_id") == "library"
        else tab_chat_title
    )

    def on_main_tab_change() -> None:
        selected_title = st.session_state.get("main_tabs")
        if selected_title in [
            get_text("tab_library", "zh"),
            get_text("tab_library", "en"),
        ]:
            st.session_state["active_tab_id"] = "library"
        else:
            st.session_state["active_tab_id"] = "chat"

    # 若切换语言，确保 session_state 中的 main_tabs 与当前语言对齐
    if st.session_state.get("main_tabs") not in tab_titles:
        st.session_state["main_tabs"] = default_tab_title

    tab_chat, tab_library = st.tabs(
        tab_titles,
        default=default_tab_title,
        key="main_tabs",
        on_change=on_main_tab_change,
    )

    with tab_chat:
        render_chat_tab(
            synthesizer=synthesizer,
            default_prompt=sidebar_state.get("selected_sample"),
            lang=lang,
        )

    with tab_library:
        render_paper_library_tab(
            filtered_papers=filtered_papers,
            db_manager=db_manager,
            vector_store=vector_store,
            lang=lang,
        )


if __name__ == "__main__":
    main()
