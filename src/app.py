"""Streamlit 交互式问答界面与文献库可视化系统 (Task 5.2).

提供：
1. 知识库文献的可视化检索与多维标签筛选（模型范式、Action-conditioned、时间等）；
2. 基于 Qdrant 向量检索的 RAG 智能问答对话框，支持多篇世界模型文献横向对比；
3. 输出答案具备严谨学术论述与完整的原始文献引用（arXiv ID、标题、PDF 链接）。
"""

from __future__ import annotations

import json
import logging
from typing import Any

import streamlit as st

from src.core.schema import ModelParadigm
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


class RAGSynthesizer:
    """RAG 问答与文献对比合成器。"""

    def __init__(self, vector_store: VectorStore, llm: Any | None = None) -> None:
        self.vector_store = vector_store
        self.llm = llm

    def search(
        self,
        query: str,
        limit: int = 5,
        paradigm_filter: str | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        """从向量库中检索相关文献片段。"""
        filter_criteria: dict[str, Any] | None = None
        if paradigm_filter and paradigm_filter != "全部":
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
    ) -> str:
        """根据检索结果生成回答与引用文献。"""
        if not search_results:
            return (
                "**未在向量知识库中检索到高相关性的世界模型文献。**\n\n"
                "💡 **建议尝试**：\n"
                "- 调整查询关键词（如更换为 `Flow Matching`、`扩散模型`、`CALVIN`、`动作条件` 等）；\n"
                "- 在左侧侧边栏放宽范式与动作条件过滤条件；\n"
                "- 确认后台流水线已运行并完成文献入库。"
            )

        # 若存在配置好的 LLM，则调用 LLM 生成
        if self.llm is not None:
            try:
                return self._generate_with_llm(query, search_results)
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM 生成回答失败，回退到启发式合成引擎: %s", exc)

        # 默认高质量结构化合成引擎（用于离线、免 Key 或单测环境）
        return self._generate_with_heuristics(query, search_results)

    def _generate_with_llm(
        self, query: str, search_results: list[VectorSearchResult]
    ) -> str:
        """调用外部 LLM 完成 RAG 合成与多篇对比。"""
        context_blocks: list[str] = []
        for idx, res in enumerate(search_results, start=1):
            context_blocks.append(
                f"【文献 {idx}】\n"
                f"- 标题: {res.title}\n"
                f"- arXiv ID: {res.arxiv_id}\n"
                f"- 模型范式: {res.model_paradigm}\n"
                f"- 动作条件 (Action-conditioned): {'是' if res.action_conditioned else '否'}\n"
                f"- 评测环境: {', '.join(res.eval_environments) if res.eval_environments else '未提及'}\n"
                f"- 核心创新: {res.core_innovation}\n"
                f"- 相关文本切片:\n{res.chunk_text}\n"
            )

        context_str = "\n".join(context_blocks)
        prompt = (
            "你是一名资深的具身智能与机器人世界模型领域专家。\n"
            "请基于以下提供的检索文献片段，针对用户的问题给出专业、全面、深度的分析。\n"
            "如果用户提到了对比/比较，请从【模型范式】、【动作闭环控制 (Action-conditioned)】、"
            "【评测基准环境】及【核心架构与创新机制】等维度进行严谨横向对比，并整理对比表格。\n"
            "回答最后必须附带标准的【📚 参考文献引用】列表（包含论文题目、arXiv ID 及链接）。\n\n"
            f"【检索上下文】:\n{context_str}\n\n"
            f"【用户问题】: {query}\n"
        )

        assert self.llm is not None
        response = self.llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        return str(content)

    def _generate_with_heuristics(
        self, query: str, search_results: list[VectorSearchResult]
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
            for kw in ["对比", "比较", "compare", "vs", "区别", "差异", "横向"]
        )

        lines: list[str] = []

        if is_comparison:
            lines.append(
                f"### 📊 检索与横向对比分析（共召回 {len(unique_papers)} 篇代表性世界模型文献）\n"
            )
            lines.append(
                f"针对您提出的问题 **“{query}”**，基于向量语义检索与知识库元数据，"
                "我们对相关代表性文献的架构范式、动作调节机制与实验评测展开系统性横向对比：\n"
            )

            # 横向对比表格
            lines.append(
                "| 论文编号 | 论文标题 | 模型范式 | 动作闭环 (Action-cond) | 评测基准环境 | 核心技术创新点 |"
            )
            lines.append("| :--- | :--- | :--- | :---: | :--- | :--- |")
            for p in unique_papers:
                action_sym = "✅ 显式支持" if p["action_conditioned"] else "❌ 否"
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
                innov = p["core_innovation"] or "未明确注明"
                lines.append(
                    f"**{idx}. [{p['title']}](https://arxiv.org/abs/{p['arxiv_id']})**\n"
                    f"- **范式与控制**：采用 `{p['model_paradigm']}`，{action_desc}{envs_desc}。\n"
                    f"- **核心创新机制**：{innov}\n"
                )

        # 规范的引用文献清单
        lines.append("\n---\n### 📚 原始文献引用列表")
        for idx, p in enumerate(unique_papers, start=1):
            pdf_url = p["payload"].get(
                "pdf_url", f"https://arxiv.org/pdf/{p['arxiv_id']}"
            )
            lines.append(
                f"{idx}. **[{p['title']}](https://arxiv.org/abs/{p['arxiv_id']})**  \n"
                f"   - **arXiv 编号**: `{p['arxiv_id']}` | **模型范式**: `{p['model_paradigm']}` | "
                f"**动作条件**: {'✅ 是' if p['action_conditioned'] else '❌ 否'}  \n"
                f"   - **评测环境**: {', '.join(p['eval_environments']) if p['eval_environments'] else '无'}  \n"
                f"   - **PDF 全文**: [{pdf_url}]({pdf_url})"
            )

        return "\n".join(lines)


@st.cache_resource
def get_storage_managers(
    sqlite_path: str | None = None,
    qdrant_path: str | None = None,
) -> tuple[DBManager, VectorStore]:
    """初始化并缓存 SQLite 与 Qdrant 存储管理器。"""
    db_manager = DBManager(db_path=sqlite_path or DEFAULT_SQLITE_PATH)
    vector_store = VectorStore(storage_path=qdrant_path or DEFAULT_QDRANT_STORAGE_PATH)
    return db_manager, vector_store


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
) -> dict[str, Any]:
    """渲染侧边栏统计信息与交互式多维过滤组件。"""
    st.sidebar.title("🤖 RoboWorld Agent")
    st.sidebar.caption("具身机器人世界模型文献知识库与 RAG 问答平台")

    # 1. 知识库全局统计指标
    total_count = len(all_papers)
    wm_count = sum(1 for p in all_papers if p.get("is_wm", 0))
    action_cond_count = 0
    for p in all_papers:
        ins = parse_insight_payload(p.get("insight_json"))
        if ins.get("action_conditioned", False):
            action_cond_count += 1

    st.sidebar.markdown("### 📊 知识库统计")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("总文献数", total_count)
    with col2:
        st.metric("世界模型", wm_count)
    st.sidebar.caption(f"🎯 动作闭环 (Action-cond): **{action_cond_count}** 篇")
    st.sidebar.markdown("---")

    # 2. 多维分类标签筛选
    st.sidebar.markdown("### 🔍 文献筛选与检索")

    search_keyword = st.sidebar.text_input(
        "🔎 关键词检索 (标题 / arXiv ID)",
        value="",
        placeholder="例如 Flow Matching 或 2609...",
    )

    selected_paradigms = st.sidebar.multiselect(
        "🏷️ 模型范式筛选",
        options=AVAILABLE_PARADIGMS,
        default=[],
        help="选择一个或多个模型范式进行过滤，留空显示全部",
    )

    action_conditioned_only = st.sidebar.checkbox(
        "🎮 仅显示动作条件 (Action-conditioned)",
        value=False,
        help="仅保留显式输入机器人动作条件的具身世界模型文献",
    )

    world_model_only = st.sidebar.checkbox(
        "🌐 仅展示确认世界模型 (WM Only)",
        value=True,
        help="过滤掉初筛未通过或非世界模型论文",
    )

    st.sidebar.markdown("---")

    # 3. 示例快捷问答
    st.sidebar.markdown("### 💡 快捷提问示例")
    sample_queries = [
        "对比近期基于 Flow Matching 的世界模型方法",
        "有哪些显式支持 Action-conditioned 的世界模型？",
        "列出在 CALVIN 或 RLBench 基准上评测的代表性工作",
        "扩散架构在机器人世界模型中的主要应用机制是什么？",
    ]
    selected_sample: str | None = None
    for idx, sq in enumerate(sample_queries):
        if st.sidebar.button(f"📌 {sq}", key=f"sq_btn_{idx}"):
            selected_sample = sq

    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ 清空问答历史", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

    return {
        "search_keyword": search_keyword,
        "selected_paradigms": selected_paradigms,
        "action_conditioned_only": action_conditioned_only,
        "world_model_only": world_model_only,
        "selected_sample": selected_sample,
    }


def render_paper_library_tab(filtered_papers: list[dict[str, Any]]) -> None:
    """渲染已收录文献库面板。"""
    st.subheader(f"📚 已收录文献列表（当前筛选展示 {len(filtered_papers)} 篇）")

    if not filtered_papers:
        st.info("⚠️ 当前筛选条件下暂未找到匹配的文献，请调整侧边栏的过滤参数。")
        return

    for item in filtered_papers:
        arxiv_id = item.get("arxiv_id", "")
        title = item.get("title", "")
        pub_date = str(item.get("published_date", ""))[:10]
        insight = item.get("parsed_insight", {})
        is_wm = bool(item.get("is_wm", 0))

        paradigm = insight.get("model_paradigm", "Other")
        action_cond = insight.get("action_conditioned", False)
        envs = insight.get("eval_environments", [])
        innov = insight.get("core_innovation", "暂无结构化洞察")

        action_badge = "✅ Action-conditioned" if action_cond else "❌ No Action"
        wm_badge = "🌐 World Model" if is_wm else "📄 General"

        with st.container():
            st.markdown(
                f"#### [{title}](https://arxiv.org/abs/{arxiv_id}) "
                f"<span style='font-size: 0.8em; color: gray;'>(`arXiv:{arxiv_id}`)</span>",
                unsafe_allow_html=True,
            )
            col_badges, col_links = st.columns([3, 1])
            with col_badges:
                st.markdown(
                    f"`{paradigm}` | `{action_badge}` | `{wm_badge}` | 📅 `{pub_date}`"
                )
            with col_links:
                st.markdown(
                    f"[📖 arXiv](https://arxiv.org/abs/{arxiv_id}) &nbsp;|&nbsp; "
                    f"[📥 PDF](https://arxiv.org/pdf/{arxiv_id})"
                )

            if envs:
                st.markdown("**评测环境**: " + " ".join([f"`{e}`" for e in envs]))
            st.markdown(f"**核心创新**: {innov}")

            with st.expander("🔍 查看完整元数据与摘要"):
                st.json(
                    {
                        "arxiv_id": arxiv_id,
                        "title": title,
                        "published_date": pub_date,
                        "insight": insight,
                    }
                )
            st.divider()


def render_chat_tab(
    synthesizer: RAGSynthesizer,
    default_prompt: str | None = None,
) -> None:
    """渲染交互式 RAG 问答对话框。"""
    st.subheader("💬 机器人世界模型知识库交互式问答 (RAG)")
    st.caption(
        "依托 Qdrant 向量语义检索与论文切片底座，支持针对世界模型方法的多篇文献横向对比、机制剖析与溯源引用。"
    )

    # 初始化会话状态
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": (
                    "你好！我是 **RoboWorld Agent** 学术助理。\n\n"
                    "我可以基于知识库中收录的机器人世界模型论文为您提供深度问答与横向对比分析，例如：\n"
                    "- *“对比基于 Flow Matching 与扩散架构的世界模型方法”*\n"
                    "- *“总结显式接收 Action 输入并完成闭环规划的代表性模型”*\n"
                    "- *“CALVIN 环境下目前主流的世界模型有哪些？”*\n\n"
                    "请在下方输入您的问题开始探索！"
                ),
            }
        ]

    # 渲染历史消息
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 检查是否有侧边栏点击的快捷提问
    active_prompt: str | None = None
    if default_prompt:
        active_prompt = default_prompt

    # 对话输入框
    user_input = st.chat_input(
        "提出关于世界模型的问题，例如：对比上周入库的使用 Flow Matching 的世界模型方法..."
    )
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
            st.spinner("🔍 正在检索向量知识库并进行横向对比分析..."),
        ):
            search_results = synthesizer.search(query=active_prompt, limit=5)
            answer = synthesizer.generate_answer(
                query=active_prompt,
                search_results=search_results,
            )
            st.markdown(answer)

        # 3. 记录助手回答
        st.session_state["messages"].append({"role": "assistant", "content": answer})


def main() -> None:
    """Streamlit 主应用程序入口。"""
    st.set_page_config(
        page_title="RoboWorld Agent - 具身机器人世界模型文献知识库",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 标题与介绍横幅
    st.title("🤖 RoboWorld Agent - 具身世界模型前沿学术平台")
    st.markdown(
        "专为具身智能与机器人世界模型领域打造的自动化文献追踪、结构化研判与交互式对比问答系统。"
    )

    # 获取存储管理器与合成器
    try:
        db_manager, vector_store = get_storage_managers()
        synthesizer = RAGSynthesizer(vector_store=vector_store)
        all_papers = db_manager.list_papers(limit=500)
    except Exception as exc:  # noqa: BLE001
        st.error(f"连接存储底座失败: {exc}")
        return

    # 渲染侧边栏并获取筛选状态
    sidebar_state = render_sidebar(all_papers)

    # 根据侧边栏选项过滤文献
    filtered_papers = filter_papers_data(
        papers=all_papers,
        search_keyword=sidebar_state["search_keyword"],
        selected_paradigms=sidebar_state["selected_paradigms"],
        action_conditioned_only=sidebar_state["action_conditioned_only"],
        world_model_only=sidebar_state["world_model_only"],
    )

    # 主界面选项卡
    tab_chat, tab_library = st.tabs(
        ["💬 交互式学术问答 (RAG)", f"📚 知识库文献库 ({len(filtered_papers)})"]
    )

    with tab_chat:
        render_chat_tab(
            synthesizer=synthesizer,
            default_prompt=sidebar_state.get("selected_sample"),
        )

    with tab_library:
        render_paper_library_tab(filtered_papers)


if __name__ == "__main__":
    main()
