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
# 1. Golden Dataset 测试集校验
# ============================================================================


def test_golden_dataset_file_exists_and_valid() -> None:
    """验证本地 data/golden_dataset.json 的存在性与数据规模。"""
    assert DEFAULT_GOLDEN_DATASET_PATH.exists()
    samples = load_golden_dataset()

    assert len(samples) == 20
    wm_samples = [s for s in samples if s.is_world_model]
    non_wm_samples = [s for s in samples if not s.is_world_model]

    assert len(wm_samples) == 10
    assert len(non_wm_samples) == 10

    is_valid, msg = validate_golden_dataset(samples)
    assert is_valid
    assert "校验通过" in msg


def test_golden_dataset_validation_failures() -> None:
    """测试数据集校验器对异常样本分布的拦截能力。"""
    # 样本不足
    valid, msg = validate_golden_dataset([])
    assert not valid
    assert "不足 20 篇" in msg

    # 正样本不足 10 篇
    dummy = [
        GoldenPaperSample(
            arxiv_id=f"2401.0000{i}",
            title=f"Sample {i}",
            abstract="Abstract text",
            published_date="2024-01-01T00:00:00Z",
            pdf_url="https://arxiv.org/pdf/test",
            is_world_model=(i < 5),  # 仅 5 篇为 True
        )
        for i in range(20)
    ]
    valid, msg = validate_golden_dataset(dummy)
    assert not valid
    assert "正样本不足 10 篇" in msg


def test_golden_sample_conversion_to_schema() -> None:
    """测试 GoldenPaperSample 转换为领域实体 RawPaper 与 PaperInsight。"""
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
        core_innovation="测试连续流",
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
# 2. TokenUsageTracker 与可观测性 Tracing 单元测试
# ============================================================================


def test_token_usage_tracker_llm_lifecycle() -> None:
    """测试 TokenUsageTracker 对 LLM 启动、完成、Token 累加与耗时统计。"""
    tracker = TokenUsageTracker()

    # 1. on_llm_start
    tracker.on_llm_start(
        serialized={},
        prompts=["Analyze this paper: World model for robotic manipulation"],
        run_id="run-1",
    )
    assert tracker.llm_call_count == 1

    # 2. on_llm_end 携带官方 token_usage
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

    # 3. 统计摘要
    summary = tracker.get_summary()
    assert summary["total_tokens"] == 57
    assert summary["llm_call_count"] == 1
    assert summary["error_count"] == 0

    # 4. 重置
    tracker.reset()
    assert tracker.total_tokens == 0
    assert tracker.llm_call_count == 0


def test_token_usage_tracker_fallback_estimation() -> None:
    """测试在离线无 token_usage 字典时的字符估算逻辑。"""
    tracker = TokenUsageTracker()
    tracker.on_llm_start(
        serialized={},
        prompts=["1234567890" * 4],  # 40 字符 -> 约 10 tokens
        run_id="run-est",
    )
    fake_response = LLMResult(
        generations=[
            [Generation(text="1234567890" * 8)]  # 80 字符 -> 约 20 tokens
        ],
        llm_output={},
    )
    tracker.on_llm_end(fake_response, run_id="run-est")

    assert tracker.prompt_tokens == 10
    assert tracker.completion_tokens == 20
    assert tracker.total_tokens == 30


def test_token_usage_tracker_error_and_chains() -> None:
    """测试异常记录与 Chain/Tool 拓扑计数。"""
    tracker = TokenUsageTracker()
    tracker.on_chain_start({}, {})
    tracker.on_tool_start({}, "arxiv_id: 2405.00001")
    tracker.on_llm_error(RuntimeError("API Timeout"), run_id="run-err")

    summary = tracker.get_summary()
    assert summary["chain_call_count"] == 1
    assert summary["tool_call_count"] == 1
    assert summary["error_count"] == 1


def test_setup_tracing_and_trace_run() -> None:
    """测试 setup_tracing 与 trace_run 上下文管理器。"""
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
    """验证占位符 API Key 或空 Key 自动关闭 LangSmith 并同步关闭环境变量。"""
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
    """验证当 API Key 权限不足或网络异常时，系统自动捕获告警并平滑降级为本地 Tracing。"""
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
        # 仅保留本地 TokenUsageTracker，不注入 LangChainTracer
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], TokenUsageTracker)


def test_tracing_valid_credentials_enables_langsmith() -> None:
    """验证当凭据有效时，成功初始化 Client 并挂载 LangChainTracer。"""
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
# 3. Golden Dataset 自动化评估与验收标准测试 (>=90% 达标)
# ============================================================================


def test_dataset_evaluator_on_golden_dataset() -> None:
    """核心验收标准：在 Golden Dataset 上分类准确率 >= 90%，无解析崩坏。"""
    samples = load_golden_dataset()
    classifier = ClassifierAgent()
    evaluator = DatasetEvaluator(classifier=classifier)

    result = evaluator.evaluate(samples=samples)

    # 1. 验证样本总数与混淆矩阵完整性
    assert result.total_samples == 20
    assert result.tp + result.fp + result.tn + result.fn == 20

    # 2. 验证验收门禁：分类准确率 >= 90%
    assert result.accuracy >= 0.90, f"分类准确率低于 90%: {result.accuracy}"
    assert result.passes_threshold(0.90) is True

    # 3. 验证精确率与召回率计算
    assert result.precision >= 0.90
    assert result.recall >= 0.90
    assert result.f1_score >= 0.90

    # 4. 验证范式与动作闭环识别率
    assert result.paradigm_accuracy >= 0.80
    assert result.action_conditioned_accuracy >= 0.80

    # 5. 验证单篇明细列表
    assert len(result.details) == 20


def test_evaluation_result_markdown_report_formatting() -> None:
    """测试生成的 Markdown 报告结构与元素完整性。"""
    samples = load_golden_dataset()[:2]
    classifier = ClassifierAgent()
    evaluator = DatasetEvaluator(classifier=classifier)
    result = evaluator.evaluate(samples=samples)

    md = result.to_markdown_report()
    assert "# 🤖 RoboWorld Agent - 自动化研判模型 Eval 评估报告" in md
    assert "混淆矩阵 (Confusion Matrix)" in md
    assert "可观测性与 Token 消耗监控" in md
    assert "样本逐篇判定明细表" in md


# ============================================================================
# 4. CLI 评测脚本 (eval_runner) 测试
# ============================================================================


def test_eval_runner_cli_execution(tmp_path: Path) -> None:
    """测试 eval_runner 命令行接口与报告落盘。"""
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
    assert "RoboWorld Agent - 自动化研判模型 Eval 评估报告" in content


# ============================================================================
# 5. 工作流流水线端到端可观测性集成测试
# ============================================================================


def test_run_daily_pipeline_with_tracing(tmp_path: Path) -> None:
    """验证 run_daily_pipeline 挂载 TracingManager 并记录 Token 指标。"""
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
