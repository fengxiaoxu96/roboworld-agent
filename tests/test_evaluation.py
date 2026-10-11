from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.outputs import Generation, LLMResult

from src.agents.classifier_agent import ClassifierAgent
from src.core.schema import RawPaper
from src.evaluation.dataset import (
    DEFAULT_GOLDEN_DATASET_PATH,
    GoldenPaperSample,
    load_golden_dataset,
    validate_golden_dataset,
)
from src.evaluation.eval_runner import main as eval_runner_main
from src.evaluation.evaluator import DatasetEvaluator
from src.evaluation.tracing import (
    TokenUsageTracker,
    TracingManager,
    setup_tracing,
    trace_run,
)
from src.storage import DBManager, VectorStore
from src.workflows.scheduler import run_daily_pipeline

# ============================================================================
# 1. Golden Dataset Validation Tests
# ============================================================================


def test_golden_dataset_file_exists_and_valid() -> None:
    """Verify existence and size of local data/golden_dataset.json."""
    assert DEFAULT_GOLDEN_DATASET_PATH.exists()
    samples = load_golden_dataset()

    assert len(samples) == 20
    wm_samples = [s for s in samples if s.is_world_model]
    non_wm_samples = [s for s in samples if not s.is_world_model]

    assert len(wm_samples) == 10
    assert len(non_wm_samples) == 10

    is_valid, msg = validate_golden_dataset(samples)
    assert is_valid
    assert "Validation passed" in msg


def test_golden_dataset_validation_failures() -> None:
    """Test validator catching abnormal sample distributions."""
    # Insufficient samples
    valid, msg = validate_golden_dataset([])
    assert not valid
    assert "less than 20" in msg.lower()

    # Insufficient positive samples (< 10)
    dummy = [
        GoldenPaperSample(
            arxiv_id=f"2401.0000{i}",
            title=f"Sample {i}",
            abstract="Abstract text",
            published_date="2024-01-01T00:00:00Z",
            pdf_url="https://arxiv.org/pdf/test",
            is_world_model=(i < 5),  # only 5 are True
        )
        for i in range(20)
    ]
    valid, msg = validate_golden_dataset(dummy)
    assert not valid
    assert "less than 10" in msg.lower()


def test_golden_sample_conversion_to_schema() -> None:
    """Test conversion of GoldenPaperSample to domain entities RawPaper and PaperInsight."""
    sample = GoldenPaperSample(
        arxiv_id="2405.99999",
        title="Test Embodied Dynamics",
        abstract="A test abstract for world modeling.",
        published_date="2024-05-15T12:00:00Z",
        pdf_url="https://arxiv.org/pdf/2405.99999.pdf",
        is_world_model=True,
        model_paradigm="Flow-Matching",
        action_conditioned=True,
        eval_environments=["CALVIN"],
        core_innovation="Test continuous flow",
    )

    raw_paper = sample.to_raw_paper()
    assert isinstance(raw_paper, RawPaper)
    assert raw_paper.arxiv_id == "2405.99999"
    assert raw_paper.title == "Test Embodied Dynamics"

    insight = sample.to_paper_insight()
    assert insight.is_world_model is True
    assert insight.model_paradigm == "Flow-Matching"
    assert insight.action_conditioned is True


# ============================================================================
# 2. TokenUsageTracker and Observability Tracing Unit Tests
# ============================================================================


def test_token_usage_tracker_llm_lifecycle() -> None:
    """Test TokenUsageTracker recording LLM start, end, token accumulation, and latency."""
    tracker = TokenUsageTracker()

    # 1. on_llm_start
    tracker.on_llm_start(
        serialized={},
        prompts=["Analyze this paper: World model for robotic manipulation"],
        run_id="run-1",
    )
    assert tracker.llm_call_count == 1

    # 2. on_llm_end carrying official token_usage
    fake_response = LLMResult(
        generations=[
            [Generation(text='{"is_world_model": true, "confidence": "high"}')]
        ],
        llm_output={
            "token_usage": {"prompt_tokens": 42, "completion_tokens": 15},
            "model_name": "gpt-4o",
        },
    )
    tracker.on_llm_end(fake_response, run_id="run-1")

    assert tracker.prompt_tokens == 42
    assert tracker.completion_tokens == 15
    assert tracker.total_tokens == 57
    assert len(tracker.records) == 1
    assert tracker.records[0].model_name == "gpt-4o"
    assert tracker.records[0].latency_seconds >= 0.0

    # 3. Summary statistics
    summary = tracker.get_summary()
    assert summary["total_tokens"] == 57
    assert summary["llm_call_count"] == 1
    assert summary["error_count"] == 0

    # 4. Reset
    tracker.reset()
    assert tracker.total_tokens == 0
    assert tracker.llm_call_count == 0


def test_token_usage_tracker_fallback_estimation() -> None:
    """Test fallback character token estimation logic when token_usage dict is absent."""
    tracker = TokenUsageTracker()
    tracker.on_llm_start(
        serialized={},
        prompts=["1234567890" * 4],  # 40 chars -> approx 10 tokens
        run_id="run-est",
    )
    fake_response = LLMResult(
        generations=[
            [Generation(text="1234567890" * 8)]  # 80 chars -> approx 20 tokens
        ],
        llm_output={},
    )
    tracker.on_llm_end(fake_response, run_id="run-est")

    assert tracker.prompt_tokens == 10
    assert tracker.completion_tokens == 20
    assert tracker.total_tokens == 30


def test_token_usage_tracker_error_and_chains() -> None:
    """Test error recording and Chain/Tool topology counts."""
    tracker = TokenUsageTracker()
    tracker.on_chain_start({}, {})
    tracker.on_tool_start({}, "arxiv_id: 2405.00001")
    tracker.on_llm_error(RuntimeError("API Timeout"), run_id="run-err")

    summary = tracker.get_summary()
    assert summary["chain_call_count"] == 1
    assert summary["tool_call_count"] == 1
    assert summary["error_count"] == 1


def test_setup_tracing_and_trace_run() -> None:
    """Test setup_tracing and trace_run context manager."""
    mgr = setup_tracing(project_name="test-project", enable_langsmith=False)
    assert mgr.project_name == "test-project"
    callbacks = mgr.get_callbacks()
    assert any(isinstance(cb, TokenUsageTracker) for cb in callbacks)

    config = mgr.get_langgraph_config()
    assert "callbacks" in config
    assert config["metadata"]["project"] == "test-project"

    with trace_run(
        "test_scope",
        project_name="test-project",
        enable_langsmith=False,
    ) as run_mgr:
        assert isinstance(run_mgr, TracingManager)


def test_tracing_dummy_or_empty_key_disables_langsmith() -> None:
    """Verify placeholder or empty API Key automatically disables LangSmith and environment variables."""
    with patch.dict(
        "os.environ",
        {
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_API_KEY": "your_langsmith_api_key_here",
        },
    ):
        mgr = TracingManager(project_name="dummy-proj", validate_connection=False)
        assert mgr.langsmith_enabled is False
        assert mgr.client is None
        assert (
            "false" in mgr.get_callbacks().__class__.__name__
            or len(mgr.get_callbacks()) == 1
        )


def test_tracing_invalid_credentials_gracefully_degrades() -> None:
    """Verify graceful degradation to local Tracing when credentials fail or network errors occur."""
    with patch("langsmith.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.has_project.side_effect = RuntimeError(
            "403 Client Error: Forbidden"
        )
        mock_client_cls.return_value = mock_client

        mgr = TracingManager(
            project_name="test-forbidden",
            api_key="lsv2_pt_invalid_key",
            enable_langsmith=True,
            validate_connection=True,
        )

        assert mgr.langsmith_enabled is False
        assert mgr.client is None
        callbacks = mgr.get_callbacks()
        # Keep local TokenUsageTracker only, without LangChainTracer
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], TokenUsageTracker)


def test_tracing_valid_credentials_enables_langsmith() -> None:
    """Verify successful client initialization and mounting of LangChainTracer with valid credentials."""
    with patch("langsmith.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.has_project.return_value = True
        mock_client_cls.return_value = mock_client

        mgr = TracingManager(
            project_name="test-valid",
            api_key="lsv2_pt_valid_key_12345",
            endpoint="https://api.smith.langchain.com",
            enable_langsmith=True,
            validate_connection=True,
        )

        assert mgr.langsmith_enabled is True
        assert mgr.client is mock_client

        from langchain_core.tracers import LangChainTracer

        callbacks = mgr.get_callbacks()
        assert len(callbacks) == 2
        tracers = [cb for cb in callbacks if isinstance(cb, LangChainTracer)]
        assert len(tracers) == 1
        assert tracers[0].project_name == "test-valid"
        assert tracers[0].client is mock_client


# ============================================================================
# 3. Golden Dataset Automated Evaluation and Acceptance Criteria Tests (>=90%)
# ============================================================================


def test_dataset_evaluator_on_golden_dataset() -> None:
    """Core acceptance criteria: Classification accuracy >= 90% on Golden Dataset without parsing errors."""
    samples = load_golden_dataset()
    classifier = ClassifierAgent()
    evaluator = DatasetEvaluator(classifier=classifier)

    result = evaluator.evaluate(samples=samples)

    # 1. Verify sample total and confusion matrix completeness
    assert result.total_samples == 20
    assert result.tp + result.fp + result.tn + result.fn == 20

    # 2. Verify acceptance gate: Accuracy >= 90%
    assert (
        result.accuracy >= 0.90
    ), f"Classification accuracy below 90%: {result.accuracy}"
    assert result.passes_threshold(0.90) is True

    # 3. Verify Precision and Recall computation
    assert result.precision >= 0.90
    assert result.recall >= 0.90
    assert result.f1_score >= 0.90

    # 4. Verify paradigm and action conditioning accuracy
    assert result.paradigm_accuracy >= 0.80
    assert result.action_conditioned_accuracy >= 0.80

    # 5. Verify sample detail list
    assert len(result.details) == 20


def test_evaluation_result_markdown_report_formatting() -> None:
    """Test generated Markdown report structure and completeness."""
    samples = load_golden_dataset()[:2]
    classifier = ClassifierAgent()
    evaluator = DatasetEvaluator(classifier=classifier)
    result = evaluator.evaluate(samples=samples)

    md = result.to_markdown_report()
    assert "# 🤖 RoboWorld Agent - Automated Evaluation Report" in md
    assert "Confusion Matrix" in md
    assert "Observability and Token Usage Monitoring" in md
    assert "Sample-by-Sample Evaluation Details" in md


# ============================================================================
# 4. CLI Evaluation Script (eval_runner) Tests
# ============================================================================


def test_eval_runner_cli_execution(tmp_path: Path) -> None:
    """Test eval_runner command line interface and report saving."""
    report_file = tmp_path / "test_report.md"

    test_args = [
        "eval_runner",
        "--dataset",
        str(DEFAULT_GOLDEN_DATASET_PATH),
        "--output",
        str(report_file),
        "--min-accuracy",
        "0.90",
        "--verbose",
    ]

    with patch("sys.argv", test_args):
        exit_code = eval_runner_main()

    assert exit_code == 0
    assert report_file.exists()
    content = report_file.read_text(encoding="utf-8")
    assert "RoboWorld Agent - Automated Evaluation Report" in content


# ============================================================================
# 5. Workflow Pipeline End-to-end Observability Integration Tests
# ============================================================================


def test_run_daily_pipeline_with_tracing(tmp_path: Path) -> None:
    """Verify run_daily_pipeline mounts TracingManager and logs token metrics."""
    db_file = tmp_path / "papers.db"
    qdrant_dir = tmp_path / "qdrant"

    db_manager = DBManager(db_path=db_file)
    vector_store = VectorStore(storage_path=qdrant_dir)
    classifier = ClassifierAgent()

    sample_paper = RawPaper(
        arxiv_id="2405.77777",
        title="Diffusion World Model for CALVIN Franka Manipulation",
        authors=["Alice", "Bob"],
        abstract="We propose an action-conditioned diffusion world model on CALVIN.",
        categories=["cs.RO"],
        published_date=datetime(2024, 5, 20, 12, 0, tzinfo=UTC),
        pdf_url="https://arxiv.org/pdf/2405.77777.pdf",
    )

    with patch("src.workflows.graph.fetch_papers", return_value=[sample_paper]):
        result = run_daily_pipeline(
            days_back=1,
            max_results=5,
            output_dir=tmp_path / "reports",
            db_manager=db_manager,
            vector_store=vector_store,
            classifier=classifier,
        )

    assert "token_usage" in result
    assert "trace_summary" in result
    token_usage = result["token_usage"]
    assert "total_tokens" in token_usage
    assert "llm_call_count" in token_usage
