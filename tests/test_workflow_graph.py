"""单测与集成测试：基于 LangGraph 的主状态图编排 (Task 4.1)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

from src.agents.classifier_agent import ClassifierAgent, FilterDecision
from src.core.schema import RawPaper
from src.storage import DBManager, VectorStore
from src.workflows.graph import (
    PaperState,
    WorkflowState,
    build_workflow_graph,
    conditional_router,
    dispatcher_router,
)


def _create_paper(
    arxiv_id: str,
    title: str,
    abstract: str,
) -> RawPaper:
    return RawPaper(
        arxiv_id=arxiv_id,
        title=title,
        authors=["Author A", "Author B"],
        abstract=abstract,
        categories=["cs.RO", "cs.AI"],
        published_date=datetime(2024, 1, 15, tzinfo=UTC),
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
    )


# =========================================================================
# 1. 条件路由器单元测试 (Conditional Router)
# =========================================================================


def test_conditional_router_decisions() -> None:
    paper = _create_paper("2401.00001", "Title", "Abstract")

    # 1. 高置信度且相关 -> extract_node
    st1: WorkflowState = {
        "current_paper_state": PaperState(
            paper=paper,
            is_world_model=True,
            confidence="high",
            tool_called=False,
        )
    }
    assert conditional_router(st1) == "extract_node"

    # 2. 高置信度且不相关 -> skip_node
    st2: WorkflowState = {
        "current_paper_state": PaperState(
            paper=paper,
            is_world_model=False,
            confidence="high",
            tool_called=False,
        )
    }
    assert conditional_router(st2) == "skip_node"

    # 3. 置信度低且首次进入 -> tool_call_node
    st3: WorkflowState = {
        "current_paper_state": PaperState(
            paper=paper,
            is_world_model=True,
            confidence="low",
            tool_called=False,
        )
    }
    assert conditional_router(st3) == "tool_call_node"

    # 4. 置信度低但已经调用过工具 -> 不再重复调用，直接走向对应决策
    st4: WorkflowState = {
        "current_paper_state": PaperState(
            paper=paper,
            is_world_model=True,
            confidence="low",
            tool_called=True,
        )
    }
    assert conditional_router(st4) == "extract_node"

    # 5. 发生错误样本 -> 隔离并走向 skip_node
    st5: WorkflowState = {
        "current_paper_state": PaperState(
            paper=paper,
            error="Connection timeout",
        )
    }
    assert conditional_router(st5) == "skip_node"


def test_dispatcher_router() -> None:
    paper = _create_paper("2401.00001", "Title", "Abstract")
    assert (
        dispatcher_router({"current_paper_state": PaperState(paper=paper)})
        == "filter_node"
    )
    assert dispatcher_router({"current_paper_state": None}) == "report_node"


# =========================================================================
# 2. 全流程集成测试：多样本分支、工具调用与错误隔离
# =========================================================================


def test_workflow_end_to_end_execution() -> None:
    """验证主状态图编排各节点流转：

    - Paper 1 (明确的世界模型): Filter -> Extract -> Storage (落库 SQLite + Qdrant)
    - Paper 2 (明确无关文献): Filter -> Skip -> Storage (仅落库 SQLite)
    - Paper 3 (模糊文献): Filter -> ToolCall -> Filter -> Extract -> Storage
    - Paper 4 (故障异常样本): FilterNode 抛异常被隔离，不阻塞流水线，正常输出 Trace
    """
    # 使用内存库隔离测试环境
    db = DBManager(db_path=":memory:")
    vstore = VectorStore(storage_path=":memory:", collection_name="wf_test")

    # 样本构造
    p1 = _create_paper(
        arxiv_id="2401.0001",
        title="WorldGym: Action-Conditioned World Model for Robot Manipulation",
        abstract=(
            "We propose an action-conditioned world model. "
            "Latent dynamics are learned from robot actions and visual transitions on CALVIN."
        ),
    )
    p2 = _create_paper(
        arxiv_id="2401.0002",
        title="Social Media Sentiment Analysis with Transformers",
        abstract="We analyze financial stock sentiment and social media text using BERT.",
    )
    p3 = _create_paper(
        arxiv_id="2401.0003",
        title="Video Prediction for Robotics Exploration",
        abstract="We explore future video prediction on robotic platforms without explicit action details.",
    )
    p4 = _create_paper(
        arxiv_id="2401.0004",
        title="Corrupted Paper Sample",
        abstract="This paper triggers a mock parser crash.",
    )

    # 模拟分类器行为
    class MockClassifier(ClassifierAgent):
        def classify(
            self, paper: RawPaper, section_text: str | None = None
        ) -> FilterDecision:
            if paper.arxiv_id == "2401.0004":
                raise RuntimeError("Simulated unexpected LLM parsing crash!")

            if paper.arxiv_id == "2401.0001":
                return FilterDecision(
                    is_world_model=True,
                    confidence="high",
                    reasoning="明确的机器人动作条件化世界模型。",
                )
            if paper.arxiv_id == "2401.0002":
                return FilterDecision(
                    is_world_model=False,
                    confidence="high",
                    reasoning="纯文本情感分析，与物理机器人无关。",
                )
            if paper.arxiv_id == "2401.0003":
                # 首次未拿到正文时置信度低
                if not section_text:
                    return FilterDecision(
                        is_world_model=True,
                        confidence="low",
                        needs_full_text=True,
                        reasoning="视频预测未确认动作闭环，需调阅正文。",
                    )
                # 调阅正文后确认动作闭环
                return FilterDecision(
                    is_world_model=True,
                    confidence="high",
                    reasoning="调阅正文确认模型显式输入机械臂关节角控制指令动作。",
                )
            return super().classify(paper, section_text)

    app = build_workflow_graph(
        db_manager=db,
        vector_store=vstore,
        classifier=MockClassifier(),
    )

    # 模拟 ToolCallNode 调阅正文返回
    with patch(
        "src.workflows.graph.get_paper_section",
        return_value="3 Methodology: We inject control action vector a_t into the dynamics model.",
    ):
        initial_state: WorkflowState = {
            "input_papers": [p1, p2, p3, p4],
            "execution_trace": [],
        }

        final_state: dict[str, Any] = app.invoke(initial_state)

    # 1. 验证结果统计
    report = final_state["final_report"]
    assert report["total_fetched"] == 4
    assert report["total_processed"] == 4
    # p1 和 p3 成功入库
    assert report["world_models_stored"] == 2
    # p2 被标记跳过
    assert report["skipped_unrelated"] == 1
    # p4 单个样本失败被隔离，计入 errors，未中断流水线
    assert report["errors"] == 1
    # p3 触发了 1 次 ToolCall
    assert report["tool_calls_made"] == 1

    # 2. 验证 SQLite 持久化
    assert db.exists("2401.0001") is True
    assert db.exists("2401.0002") is True
    assert db.exists("2401.0003") is True
    assert db.exists("2401.0004") is True
    # 仅世界模型记录标记 is_wm = 1
    rec1 = db.get_paper("2401.0001")
    rec2 = db.get_paper("2401.0002")
    assert rec1 is not None and rec1["is_wm"] == 1
    assert rec2 is not None and rec2["is_wm"] == 0

    # 3. 验证 Qdrant 向量库持久化
    # 所有成功落库的有效文献 (p1, p2, p3) 均入向量库支持全库 RAG，失败的 p4 不入向量库
    assert vstore.paper_exists("2401.0001") is True
    assert vstore.paper_exists("2401.0002") is True
    assert vstore.paper_exists("2401.0003") is True
    assert vstore.paper_exists("2401.0004") is False

    # 4. 验证完整 Execution Trace 状态
    trace = final_state["execution_trace"]
    assert len(trace) > 0
    node_names = [item["node"] for item in trace]
    assert "FetchNode" in node_names
    assert "FilterNode" in node_names
    assert "ToolCallNode" in node_names
    assert "ExtractNode" in node_names
    assert "SkipNode" in node_names
    assert "StorageNode" in node_names
    assert "ReportNode" in node_names

    # 5. 验证 Markdown 报告已生成且包含关键字段
    report_md = final_state["report_markdown"]
    assert "# 🤖 RoboWorld 机器人世界模型每日处理简报" in report_md
    assert "WorldGym" in report_md


def test_workflow_empty_papers_handling() -> None:
    """验证当未抓取到任何论文时的健壮处理与正常收尾。"""
    db = DBManager(db_path=":memory:")
    vstore = VectorStore(storage_path=":memory:", collection_name="wf_empty")

    app = build_workflow_graph(
        db_manager=db,
        vector_store=vstore,
    )

    initial_state: WorkflowState = {
        "input_papers": [],
        "execution_trace": [],
    }

    final_state: dict[str, Any] = app.invoke(initial_state)

    report = final_state["final_report"]
    assert report["total_fetched"] == 0
    assert report["total_processed"] == 0
    assert (
        "今日未发现符合标准的具身机器人世界模型文献" in final_state["report_markdown"]
    )
