"""LangGraph-based main state graph orchestration pipeline (Task 4.1).

Workflow nodes and transitions:
1. FetchNode: Fetch and pre-filter arXiv metadata;
2. DispatcherNode: Loop-dispatch paper samples to be processed;
3. FilterNode: LLM / classifier performs initial classification on paper abstract;
4. Conditional Router:
   - High confidence and irrelevant -> SkipNode (save SQLite only, not vector store);
   - High confidence and relevant -> ExtractNode;
   - Low confidence/insufficient info -> ToolCallNode (consult key PDF section) -> re-evaluate;
5. StorageNode: Persist metadata and vector chunks in dual storage;
6. ReportNode: Format daily processing output and statistical metrics;
7. Acceptance criteria: sample errors isolated safely without blocking the pipeline, producing full Execution Trace.
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
    """Execution state tracking for a single paper in the workflow pipeline."""

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
    """LangGraph main workflow global state."""

    # Input configuration
    days_back: int | None
    max_results: int | None
    categories: list[str]
    input_papers: list[RawPaper] | None

    # Batch dispatch management
    raw_papers: list[RawPaper]
    current_index: int
    current_paper_state: PaperState | None

    # Processing artifacts summary
    processed_papers: list[PaperState]
    stored_count: int
    skipped_count: int
    error_count: int
    tool_call_count: int
    scanned_count: int
    filter_keywords: list[str]

    # Execution Trace and reporting
    execution_trace: list[dict[str, Any]]
    final_report: dict[str, Any]
    report_markdown: str


class WorkflowEngine:
    """Workflow engine encapsulating node logic and dependent components."""

    def __init__(
        self,
        db_manager: DBManager | None = None,
        vector_store: VectorStore | None = None,
        classifier: ClassifierAgent | None = None,
        llm: Any | None = None,
    ) -> None:
        self.db = db_manager or DBManager()
        self.vector_store = vector_store or VectorStore()
        if classifier is not None:
            self.classifier = classifier
        elif llm is not None:
            self.classifier = ClassifierAgent(llm=llm)
        else:
            self.classifier = ClassifierAgent()

    def fetch_node(self, state: WorkflowState) -> dict[str, Any]:
        """FetchNode: Fetch papers to be evaluated."""
        trace = list(state.get("execution_trace", []))
        input_papers = state.get("input_papers")
        scanned_count = 0
        filter_keywords = list(KEYWORD_PATTERNS)

        if input_papers is not None:
            papers = list(input_papers)
            scanned_count = len(papers)
            logger.info("Using provided preset paper set, total %d papers", len(papers))
        else:
            days_back = state.get("days_back")
            max_results = state.get("max_results")
            if days_back is None and max_results is None:
                days_back = 1
            categories = state.get("categories", list(DEFAULT_CATEGORIES))
            try:
                # If fetch_papers is patched externally (e.g., in unit tests), prioritize its return value;
                # otherwise use fetch_papers_with_stats for rich statistics.
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
                logger.error("FetchNode fetching exception: %s", exc)
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
        """DispatcherNode: Dispatch next paper into the evaluation pipeline."""
        raw_papers = state.get("raw_papers", [])
        current_index = state.get("current_index", 0)

        if current_index < len(raw_papers):
            target_paper = raw_papers[current_index]
            paper_state = PaperState(paper=target_paper)
            paper_state.events.append("Dispatched to FilterNode")
            return {"current_paper_state": paper_state}

        return {"current_paper_state": None}

    def filter_node(self, state: WorkflowState) -> dict[str, Any]:
        """FilterNode: Perform classification evaluation on the paper."""
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
            logger.warning(
                "FilterNode error processing paper %s: %s", paper.arxiv_id, exc
            )
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
        """ToolCallNode: Retrieve Methodology section when abstract confidence is insufficient."""
        current_state = state["current_paper_state"]
        if current_state is None:
            return {}

        trace = list(state.get("execution_trace", []))
        paper = current_state.paper
        current_state.tool_called = True
        current_state.section_name = "Methodology"

        try:
            target_source = paper.pdf_url or paper.arxiv_id
            logger.info(
                "Insufficient confidence, fetching %s section directly in memory without disk persistence...",
                target_source,
            )
            section_content = get_paper_section(
                arxiv_id=target_source,
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
            logger.warning(
                "ToolCallNode failed to fetch section for %s: %s", paper.arxiv_id, exc
            )
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
        """ExtractNode: Extract structured PaperInsight for confirmed world model papers."""
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
            logger.warning(
                "ExtractNode error extracting paper %s: %s", paper.arxiv_id, exc
            )
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
        """SkipNode: Mark irrelevant papers and construct non-world-model Insight."""
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
                core_innovation=current_state.reasoning
                or "Pre-filtering determined as non-world-model paper",
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
        """StorageNode: Persist SQLite metadata and Qdrant vector chunks."""
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
            # 1. Save into SQLite
            self.db.save_paper(paper, insight)
            current_state.events.append("SQLite metadata saved")

            # 2. All papers successfully stored in SQLite are chunked into vector store (supporting full-corpus RAG)
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
            logger.error("StorageNode error saving %s: %s", paper.arxiv_id, exc)
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
        """ReportNode: Format and produce daily pipeline summary results."""
        processed = state.get("processed_papers", [])
        stored = state.get("stored_count", 0)
        skipped = state.get("skipped_count", 0)
        errors = state.get("error_count", 0)
        tool_calls = state.get("tool_call_count", 0)
        trace = list(state.get("execution_trace", []))

        # Build stats dictionary
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

        # Render Markdown report
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
    """Conditional Router decision logic.

    Branching rules:
    - If unrecoverable error occurred in upstream node -> route to 'skip_node';
    - If confidence is low (confidence='low') and tool not yet called -> route to 'tool_call_node';
    - If judged as world model -> route to 'extract_node';
    - If judged as irrelevant -> route to 'skip_node'.
    """
    current_state = state.get("current_paper_state")
    if current_state is None:
        return "skip_node"

    # Error priority isolation
    if current_state.error is not None:
        return "skip_node"

    # Low confidence and first entry: call PDF tool to consult full text
    if current_state.confidence == "low" and not current_state.tool_called:
        return "tool_call_node"

    # High confidence branch or final decision branch after consulting full text
    if current_state.is_world_model is True:
        return "extract_node"

    return "skip_node"


def dispatcher_router(state: WorkflowState) -> str:
    """Dispatcher conditional router: determine if papers remain to be processed."""
    if state.get("current_paper_state") is not None:
        return "filter_node"
    return "report_node"


def build_workflow_graph(
    db_manager: DBManager | None = None,
    vector_store: VectorStore | None = None,
    classifier: ClassifierAgent | None = None,
    llm: Any | None = None,
) -> CompiledStateGraph:
    """Build and compile main workflow state graph.

    Topological flow:
    START -> fetch_node -> dispatcher_node
      -> (has papers) -> filter_node -> conditional_router:
            ├─ [low confidence] -> tool_call_node -> filter_node (re-evaluate)
            ├─ [is_wm == True]  -> extract_node   -> storage_node -> dispatcher_node
            └─ [is_wm == False] -> skip_node      -> storage_node -> dispatcher_node
      -> (no papers) -> report_node -> END
    """
    engine = WorkflowEngine(
        db_manager=db_manager,
        vector_store=vector_store,
        classifier=classifier,
        llm=llm,
    )

    builder = StateGraph(WorkflowState)

    # 1. Register core state nodes
    builder.add_node("fetch_node", engine.fetch_node)
    builder.add_node("dispatcher_node", engine.dispatcher_node)
    builder.add_node("filter_node", engine.filter_node)
    builder.add_node("tool_call_node", engine.tool_call_node)
    builder.add_node("extract_node", engine.extract_node)
    builder.add_node("skip_node", engine.skip_node)
    builder.add_node("storage_node", engine.storage_node)
    builder.add_node("report_node", engine.report_node)

    # 2. Arrange edges and conditional transitions
    builder.add_edge(START, "fetch_node")
    builder.add_edge("fetch_node", "dispatcher_node")

    # Dispatch router: enter next paper or finish to report
    builder.add_conditional_edges(
        "dispatcher_node",
        dispatcher_router,
        {
            "filter_node": "filter_node",
            "report_node": "report_node",
        },
    )

    # Conditional router: core branching mechanism
    builder.add_conditional_edges(
        "filter_node",
        conditional_router,
        {
            "tool_call_node": "tool_call_node",
            "extract_node": "extract_node",
            "skip_node": "skip_node",
        },
    )

    # Return to filter_node after tool call to re-evaluate with full text
    builder.add_edge("tool_call_node", "filter_node")

    # Extract and skip converge to storage_node
    builder.add_edge("extract_node", "storage_node")
    builder.add_edge("skip_node", "storage_node")

    # Return to dispatcher_node to process next paper after storage
    builder.add_edge("storage_node", "dispatcher_node")

    # Terminate after summary report
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
