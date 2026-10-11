"""Unit and integration tests: LangGraph-based main state graph orchestration (Task 4.1)."""

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
# 1. Conditional Router Unit Tests
# =========================================================================


def test_conditional_router_decisions() -> None:
    paper = _create_paper("2401.00001", "Title", "Abstract")

    # 1. High confidence and relevant -> extract_node
    st1: WorkflowState = {
        "current_paper_state": PaperState(
            paper=paper,
            is_world_model=True,
            confidence="high",
            tool_called=False,
        )
    }
    assert conditional_router(st1) == "extract_node"

    # 2. High confidence and irrelevant -> skip_node
    st2: WorkflowState = {
        "current_paper_state": PaperState(
            paper=paper,
            is_world_model=False,
            confidence="high",
            tool_called=False,
        )
    }
    assert conditional_router(st2) == "skip_node"

    # 3. Low confidence and first entry -> tool_call_node
    st3: WorkflowState = {
        "current_paper_state": PaperState(
            paper=paper,
            is_world_model=True,
            confidence="low",
            tool_called=False,
        )
    }
    assert conditional_router(st3) == "tool_call_node"

    # 4. Low confidence but tool already called -> do not repeat tool call, route directly to decision
    st4: WorkflowState = {
        "current_paper_state": PaperState(
            paper=paper,
            is_world_model=True,
            confidence="low",
            tool_called=True,
        )
    }
    assert conditional_router(st4) == "extract_node"

    # 5. Error occurred on sample -> isolate and route to skip_node
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
# 2. End-to-end Integration Test: Multi-sample branching, tool calls, and error isolation
# =========================================================================


def test_workflow_end_to_end_execution() -> None:
    """Verify transitions across main state graph nodes:

    - Paper 1 (Clear world model): Filter -> Extract -> Storage (SQLite + Qdrant)
    - Paper 2 (Clearly irrelevant paper): Filter -> Skip -> Storage (SQLite only)
    - Paper 3 (Ambiguous paper): Filter -> ToolCall -> Filter -> Extract -> Storage
    - Paper 4 (Faulty sample): FilterNode raises exception, isolated, pipeline continues with execution trace
    """
    # Use in-memory databases to isolate test environment
    db = DBManager(db_path=":memory:")
    vstore = VectorStore(storage_path=":memory:", collection_name="wf_test")

    # Sample construction
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

    # Mock classifier behavior
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
                    reasoning="Explicit robot action-conditioned world model.",
                )
            if paper.arxiv_id == "2401.0002":
                return FilterDecision(
                    is_world_model=False,
                    confidence="high",
                    reasoning="Pure text sentiment analysis, unrelated to physical robotics.",
                )
            if paper.arxiv_id == "2401.0003":
                # Low confidence initially without full text
                if not section_text:
                    return FilterDecision(
                        is_world_model=True,
                        confidence="low",
                        needs_full_text=True,
                        reasoning="Video prediction without confirmed action closure, full text needed.",
                    )
                # Confirm action closure after consulting full text
                return FilterDecision(
                    is_world_model=True,
                    confidence="high",
                    reasoning="Full text confirms explicit robot arm joint angle control action inputs.",
                )
            return super().classify(paper, section_text)

    app = build_workflow_graph(
        db_manager=db,
        vector_store=vstore,
        classifier=MockClassifier(),
    )

    # Mock ToolCallNode returning section text
    with patch(
        "src.workflows.graph.get_paper_section",
        return_value="3 Methodology: We inject control action vector a_t into the dynamics model.",
    ):
        initial_state: WorkflowState = {
            "input_papers": [p1, p2, p3, p4],
            "execution_trace": [],
        }

        final_state: dict[str, Any] = app.invoke(initial_state)

    # 1. Verify statistics
    report = final_state["final_report"]
    assert report["total_fetched"] == 4
    assert report["total_processed"] == 4
    # p1 and p3 stored successfully
    assert report["world_models_stored"] == 2
    # p2 marked as skipped
    assert report["skipped_unrelated"] == 1
    # p4 failed sample is isolated, counted in errors, pipeline not interrupted
    assert report["errors"] == 1
    # p3 triggered 1 ToolCall
    assert report["tool_calls_made"] == 1

    # 2. Verify SQLite persistence
    assert db.exists("2401.0001") is True
    assert db.exists("2401.0002") is True
    assert db.exists("2401.0003") is True
    assert db.exists("2401.0004") is True
    # Only world model records marked is_wm = 1
    rec1 = db.get_paper("2401.0001")
    rec2 = db.get_paper("2401.0002")
    assert rec1 is not None and rec1["is_wm"] == 1
    assert rec2 is not None and rec2["is_wm"] == 0

    # 3. Verify Qdrant vector store persistence
    # All successfully stored valid papers (p1, p2, p3) enter vector store for full-corpus RAG; failed p4 does not
    assert vstore.paper_exists("2401.0001") is True
    assert vstore.paper_exists("2401.0002") is True
    assert vstore.paper_exists("2401.0003") is True
    assert vstore.paper_exists("2401.0004") is False

    # 4. Verify complete Execution Trace state
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

    # 5. Verify Markdown report generated and contains key fields
    report_md = final_state["report_markdown"]
    assert "# 🤖 RoboWorld Robotics World Models Daily Briefing" in report_md
    assert "WorldGym" in report_md


def test_workflow_empty_papers_handling() -> None:
    """Verify robust handling and clean exit when no papers are fetched."""
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
        "No robotics world model papers meeting criteria were found today."
        in final_state["report_markdown"]
    )
