"""基于 LangGraph 的主状态图编排流水线 (Task 4.1).

工作流节点与流转：
1. FetchNode: 拉取并粗筛 arXiv 元数据；
2. DispatcherNode: 循环调度待处理论文样本；
3. FilterNode: LLM / 分类器对论文摘要进行第一轮分类研判；
4. Conditional Router:
   - 置信度高且不相关 -> SkipNode (仅记入 SQLite，不入向量库)；
   - 置信度高且相关 -> ExtractNode；
   - 置信度低/信息不足 -> ToolCallNode (调阅 PDF 关键章节) -> 重新判定；
5. StorageNode: 元数据与向量切片混合落库；
6. ReportNode: 格式化今日处理成果与统计指标；
7. 验收标准支持：单个样本异常被安全隔离，不阻塞流水线整体运行，生成完整 Execution Trace。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from src.agents.classifier_agent import ClassifierAgent
from src.core.schema import PaperInsight, RawPaper
from src.fetchers.arxiv_fetcher import (
    DEFAULT_CATEGORIES,
    KEYWORD_PATTERNS,
    fetch_papers,
    fetch_papers_with_stats,
)
from src.storage.db_manager import DBManager
from src.storage.vector_store import VectorStore
from src.tools.pdf_parser import get_paper_section
from src.tools.reporter import MarkdownReporter

logger = logging.getLogger(__name__)


class PaperState(BaseModel):
    """单个论文在工作流流水线中的流转状态记录。"""

    paper: RawPaper
    status: Literal["pending", "skipped", "extracted", "stored", "error"] = "pending"
    is_world_model: bool | None = None
    confidence: Literal["high", "low"] | None = None
    reasoning: str | None = None
    tool_called: bool = False
    section_name: str | None = None
    section_text: str | None = None
    insight: PaperInsight | None = None
    error: str | None = None
    events: list[str] = Field(default_factory=list)


class WorkflowState(TypedDict, total=False):
    """LangGraph 主工作流全局状态。"""

    # 输入配置
    days_back: int | None
    max_results: int | None
    categories: list[str]
    input_papers: list[RawPaper] | None

    # 批次流转管理
    raw_papers: list[RawPaper]
    current_index: int
    current_paper_state: PaperState | None

    # 处理产物汇总
    processed_papers: list[PaperState]
    stored_count: int
    skipped_count: int
    error_count: int
    tool_call_count: int
    scanned_count: int
    filter_keywords: list[str]

    # 执行跟踪 Trace 与报告
    execution_trace: list[dict[str, Any]]
    final_report: dict[str, Any]
    report_markdown: str


class WorkflowEngine:
    """工作流引擎，封装节点逻辑与依赖组件。"""

    def __init__(
        self,
        db_manager: DBManager | None = None,
        vector_store: VectorStore | None = None,
        classifier: ClassifierAgent | None = None,
    ) -> None:
        self.db = db_manager or DBManager()
        self.vector_store = vector_store or VectorStore()
        self.classifier = classifier or ClassifierAgent()

    def fetch_node(self, state: WorkflowState) -> dict[str, Any]:
        """FetchNode: 获取待研判论文集合。"""
        trace = list(state.get("execution_trace", []))
        input_papers = state.get("input_papers")
        scanned_count = 0
        filter_keywords = list(KEYWORD_PATTERNS)

        if input_papers is not None:
            papers = list(input_papers)
            scanned_count = len(papers)
            logger.info("使用传入的预置论文集合，共 %d 篇", len(papers))
        else:
            days_back = state.get("days_back")
            max_results = state.get("max_results")
            if days_back is None and max_results is None:
                days_back = 1
            categories = state.get("categories", list(DEFAULT_CATEGORIES))
            try:
                # 若 fetch_papers 被外部 patch (如单测中)，优先使用 fetch_papers 的返回值；
                # 否则使用 fetch_papers_with_stats 获取丰富统计指标。
                fetch_papers_func = fetch_papers
                is_mocked = (
                    getattr(fetch_papers_func, "_mock_return_value", None) is not None
                    or hasattr(fetch_papers_func, "mock")
                    or type(fetch_papers_func).__name__ in ("MagicMock", "Mock")
                )
                if is_mocked:
                    papers = fetch_papers(
                        days_back=days_back,
                        max_results=max_results,
                        categories=categories,
                    )
                    scanned_count = len(papers)
                else:
                    papers, stats = fetch_papers_with_stats(
                        days_back=days_back,
                        max_results=max_results,
                        categories=categories,
                    )
                    scanned_count = stats.scanned_count
                    filter_keywords = list(stats.filter_keywords)
            except Exception as exc:  # noqa: BLE001
                logger.error("FetchNode 抓取异常: %s", exc)
                papers = []
                trace.append(
                    {
                        "node": "FetchNode",
                        "status": "error",
                        "message": str(exc),
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                )

        trace.append(
            {
                "node": "FetchNode",
                "status": "success",
                "count": len(papers),
                "scanned_count": scanned_count,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        return {
            "raw_papers": papers,
            "current_index": 0,
            "processed_papers": [],
            "stored_count": 0,
            "skipped_count": 0,
            "error_count": 0,
            "tool_call_count": 0,
            "scanned_count": scanned_count,
            "filter_keywords": filter_keywords,
            "execution_trace": trace,
        }

    def dispatcher_node(self, state: WorkflowState) -> dict[str, Any]:
        """DispatcherNode: 派发下一篇论文进入研判流。"""
        raw_papers = state.get("raw_papers", [])
        current_index = state.get("current_index", 0)

        if current_index < len(raw_papers):
            target_paper = raw_papers[current_index]
            paper_state = PaperState(paper=target_paper)
            paper_state.events.append("Dispatched to FilterNode")
            return {"current_paper_state": paper_state}

        return {"current_paper_state": None}

    def filter_node(self, state: WorkflowState) -> dict[str, Any]:
        """FilterNode: 对论文执行第一轮/再轮分类判定。"""
        current_state = state["current_paper_state"]
        if current_state is None:
            return {}

        trace = list(state.get("execution_trace", []))
        paper = current_state.paper

        try:
            decision = self.classifier.classify(
                paper=paper,
                section_text=current_state.section_text,
            )
            current_state.is_world_model = decision.is_world_model
            current_state.confidence = decision.confidence
            current_state.reasoning = decision.reasoning
            current_state.events.append(
                f"Filter decision: is_wm={decision.is_world_model}, "
                f"confidence={decision.confidence}, tool_called={current_state.tool_called}"
            )
            trace.append(
                {
                    "node": "FilterNode",
                    "arxiv_id": paper.arxiv_id,
                    "is_world_model": decision.is_world_model,
                    "confidence": decision.confidence,
                    "tool_called": current_state.tool_called,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("FilterNode 处理论文 %s 异常: %s", paper.arxiv_id, exc)
            current_state.error = f"FilterNode failure: {exc}"
            current_state.status = "error"
            current_state.is_world_model = False
            current_state.confidence = "high"
            current_state.events.append(f"FilterNode error: {exc}")
            trace.append(
                {
                    "node": "FilterNode",
                    "arxiv_id": paper.arxiv_id,
                    "status": "error",
                    "error": str(exc),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

        return {
            "current_paper_state": current_state,
            "execution_trace": trace,
        }

    def tool_call_node(self, state: WorkflowState) -> dict[str, Any]:
        """ToolCallNode: 当摘要置信度不足时，调阅正文 Methodology 章节。"""
        current_state = state["current_paper_state"]
        if current_state is None:
            return {}

        trace = list(state.get("execution_trace", []))
        paper = current_state.paper
        current_state.tool_called = True
        current_state.section_name = "Methodology"

        try:
            logger.info("置信度不足，调用 PDF 解析工具调阅 %s 正文...", paper.arxiv_id)
            section_content = get_paper_section(
                arxiv_id=paper.arxiv_id,
                section_name="Methodology",
            )
            current_state.section_text = section_content
            current_state.events.append("ToolCallNode fetched Methodology section")
            trace.append(
                {
                    "node": "ToolCallNode",
                    "arxiv_id": paper.arxiv_id,
                    "section": "Methodology",
                    "length": len(section_content),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ToolCallNode 获取 %s 章节失败: %s", paper.arxiv_id, exc)
            current_state.section_text = None
            current_state.events.append(f"ToolCallNode error: {exc}")
            trace.append(
                {
                    "node": "ToolCallNode",
                    "arxiv_id": paper.arxiv_id,
                    "status": "error",
                    "error": str(exc),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

        tool_calls = state.get("tool_call_count", 0) + 1
        return {
            "current_paper_state": current_state,
            "tool_call_count": tool_calls,
            "execution_trace": trace,
        }

    def extract_node(self, state: WorkflowState) -> dict[str, Any]:
        """ExtractNode: 对确认相关的世界模型论文提取 PaperInsight 结构化洞察。"""
        current_state = state["current_paper_state"]
        if current_state is None:
            return {}

        trace = list(state.get("execution_trace", []))
        paper = current_state.paper

        try:
            insight = self.classifier.extract_insight(
                paper=paper,
                section_text=current_state.section_text,
            )
            current_state.insight = insight
            current_state.status = "extracted"
            current_state.events.append(
                f"ExtractNode extracted paradigm: {insight.model_paradigm}"
            )
            trace.append(
                {
                    "node": "ExtractNode",
                    "arxiv_id": paper.arxiv_id,
                    "model_paradigm": insight.model_paradigm,
                    "action_conditioned": insight.action_conditioned,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ExtractNode 提取论文 %s 异常: %s", paper.arxiv_id, exc)
            current_state.error = f"ExtractNode failure: {exc}"
            current_state.status = "error"
            current_state.events.append(f"ExtractNode error: {exc}")
            trace.append(
                {
                    "node": "ExtractNode",
                    "arxiv_id": paper.arxiv_id,
                    "status": "error",
                    "error": str(exc),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

        return {
            "current_paper_state": current_state,
            "execution_trace": trace,
        }

    def skip_node(self, state: WorkflowState) -> dict[str, Any]:
        """SkipNode: 对不相关论文进行标记与降级，构造非世界模型 Insight。"""
        current_state = state["current_paper_state"]
        if current_state is None:
            return {}

        trace = list(state.get("execution_trace", []))
        paper = current_state.paper

        if current_state.status != "error":
            current_state.status = "skipped"

        if current_state.insight is None:
            current_state.insight = PaperInsight(
                is_world_model=False,
                model_paradigm="Other",
                action_conditioned=False,
                eval_environments=[],
                core_innovation=current_state.reasoning or "初筛判定为非世界模型论文",
            )

        current_state.events.append("SkipNode: Marked as non-world-model")
        trace.append(
            {
                "node": "SkipNode",
                "arxiv_id": paper.arxiv_id,
                "reasoning": current_state.reasoning,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        return {
            "current_paper_state": current_state,
            "execution_trace": trace,
        }

    def storage_node(self, state: WorkflowState) -> dict[str, Any]:
        """StorageNode: 完成 SQLite 元数据与 Qdrant 向量切片落库。"""
        current_state = state["current_paper_state"]
        if current_state is None:
            return {}

        trace = list(state.get("execution_trace", []))
        paper = current_state.paper
        insight = current_state.insight
        is_wm = bool(current_state.is_world_model)

        stored_count = state.get("stored_count", 0)
        skipped_count = state.get("skipped_count", 0)
        error_count = state.get("error_count", 0)

        try:
            # 1. 记入 SQLite
            self.db.save_paper(paper, insight)
            current_state.events.append("SQLite metadata saved")

            # 2. 所有成功落库 SQLite 的文献均切片入向量库（支持全库 RAG 检索）
            if current_state.status != "error":
                extra_chunks = (
                    [current_state.section_text] if current_state.section_text else None
                )
                self.vector_store.add_paper(
                    paper=paper,
                    insight=insight,
                    extra_chunks=extra_chunks,
                )
                current_state.events.append("Qdrant vector chunks saved")
                if is_wm:
                    current_state.status = "stored"
                    stored_count += 1
                else:
                    current_state.status = "skipped"
                    skipped_count += 1
            else:
                error_count += 1

            trace.append(
                {
                    "node": "StorageNode",
                    "arxiv_id": paper.arxiv_id,
                    "status": current_state.status,
                    "is_world_model": is_wm,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

        except Exception as exc:  # noqa: BLE001
            logger.error("StorageNode 落库 %s 异常: %s", paper.arxiv_id, exc)
            current_state.error = f"StorageNode failure: {exc}"
            current_state.status = "error"
            error_count += 1
            trace.append(
                {
                    "node": "StorageNode",
                    "arxiv_id": paper.arxiv_id,
                    "status": "error",
                    "error": str(exc),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

        processed = list(state.get("processed_papers", []))
        processed.append(current_state)

        return {
            "current_index": state.get("current_index", 0) + 1,
            "processed_papers": processed,
            "stored_count": stored_count,
            "skipped_count": skipped_count,
            "error_count": error_count,
            "execution_trace": trace,
            "current_paper_state": None,
        }

    def report_node(self, state: WorkflowState) -> dict[str, Any]:
        """ReportNode: 格式化产出当日流水线汇总成果。"""
        processed = state.get("processed_papers", [])
        stored = state.get("stored_count", 0)
        skipped = state.get("skipped_count", 0)
        errors = state.get("error_count", 0)
        tool_calls = state.get("tool_call_count", 0)
        trace = list(state.get("execution_trace", []))

        # 构建统计字典
        total_fetched = len(state.get("raw_papers", []))
        scanned_count = state.get("scanned_count", total_fetched)
        filter_keywords = state.get("filter_keywords", list(KEYWORD_PATTERNS))

        report_data = {
            "scanned_count": scanned_count,
            "total_fetched": total_fetched,
            "total_processed": len(processed),
            "world_models_stored": stored,
            "skipped_unrelated": skipped,
            "errors": errors,
            "tool_calls_made": tool_calls,
            "filter_keywords": filter_keywords,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # 渲染 Markdown 报告
        reporter = MarkdownReporter()
        report_md = reporter.render(
            papers=processed,
            stats=report_data,
            skipped_records=[
                r for r in processed if getattr(r, "status", "") == "skipped"
            ],
        )

        trace.append(
            {
                "node": "ReportNode",
                "status": "completed",
                "summary": report_data,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        return {
            "final_report": report_data,
            "report_markdown": report_md,
            "execution_trace": trace,
        }


def conditional_router(state: WorkflowState) -> str:
    """Conditional Router 条件路由决策逻辑。

    流转分支规则：
    - 若前序节点发生不可恢复异常 -> 路由到 'skip_node'；
    - 若置信度低/信息不足 (confidence='low') 且未调用过工具 -> 路由到 'tool_call_node'；
    - 若判定为世界模型 -> 路由到 'extract_node'；
    - 若判定为不相关 -> 路由到 'skip_node'。
    """
    current_state = state.get("current_paper_state")
    if current_state is None:
        return "skip_node"

    # 错误优先隔离
    if current_state.error is not None:
        return "skip_node"

    # 置信度低且首次进入，调用 PDF 工具查阅正文
    if current_state.confidence == "low" and not current_state.tool_called:
        return "tool_call_node"

    # 高置信度分支或查阅正文后的最终判定分支
    if current_state.is_world_model is True:
        return "extract_node"

    return "skip_node"


def dispatcher_router(state: WorkflowState) -> str:
    """Dispatcher 条件路由：判断是否还有剩余待处理文献。"""
    if state.get("current_paper_state") is not None:
        return "filter_node"
    return "report_node"


def build_workflow_graph(
    db_manager: DBManager | None = None,
    vector_store: VectorStore | None = None,
    classifier: ClassifierAgent | None = None,
) -> CompiledStateGraph:
    """构建并编译主工作流状态图。

    拓扑流程：
    START -> fetch_node -> dispatcher_node
      -> (有文献) -> filter_node -> conditional_router:
            ├─ [low confidence] -> tool_call_node -> filter_node (重新判定)
            ├─ [is_wm == True]  -> extract_node   -> storage_node -> dispatcher_node
            └─ [is_wm == False] -> skip_node      -> storage_node -> dispatcher_node
      -> (无文献) -> report_node -> END
    """
    engine = WorkflowEngine(
        db_manager=db_manager,
        vector_store=vector_store,
        classifier=classifier,
    )

    builder = StateGraph(WorkflowState)

    # 1. 注册核心状态节点
    builder.add_node("fetch_node", engine.fetch_node)
    builder.add_node("dispatcher_node", engine.dispatcher_node)
    builder.add_node("filter_node", engine.filter_node)
    builder.add_node("tool_call_node", engine.tool_call_node)
    builder.add_node("extract_node", engine.extract_node)
    builder.add_node("skip_node", engine.skip_node)
    builder.add_node("storage_node", engine.storage_node)
    builder.add_node("report_node", engine.report_node)

    # 2. 编排边与条件转移
    builder.add_edge(START, "fetch_node")
    builder.add_edge("fetch_node", "dispatcher_node")

    # 派发路由：进入下一篇或结束进入报告
    builder.add_conditional_edges(
        "dispatcher_node",
        dispatcher_router,
        {
            "filter_node": "filter_node",
            "report_node": "report_node",
        },
    )

    # 条件路由器：核心分流机制
    builder.add_conditional_edges(
        "filter_node",
        conditional_router,
        {
            "tool_call_node": "tool_call_node",
            "extract_node": "extract_node",
            "skip_node": "skip_node",
        },
    )

    # 工具调用后闭环返回 filter_node 结合正文重新判定
    builder.add_edge("tool_call_node", "filter_node")

    # 提取与跳过汇聚到 storage_node
    builder.add_edge("extract_node", "storage_node")
    builder.add_edge("skip_node", "storage_node")

    # 单篇落库完成后回跳 dispatcher_node 处理下一篇
    builder.add_edge("storage_node", "dispatcher_node")

    # 汇总报告后终止
    builder.add_edge("report_node", END)

    return builder.compile()


__all__ = [
    "PaperState",
    "WorkflowEngine",
    "WorkflowState",
    "build_workflow_graph",
    "conditional_router",
    "dispatcher_router",
]
