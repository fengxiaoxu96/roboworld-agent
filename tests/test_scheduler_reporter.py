"""单测与集成测试：Jinja2 简报生成与定时调度器 (Task 5.1)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from src.core.schema import PaperInsight, RawPaper
from src.storage import DBManager, VectorStore
from src.tools.reporter import MarkdownReporter
from src.workflows.graph import PaperState
from src.workflows.scheduler import (
    RoboWorldScheduler,
    generate_crontab_command,
    run_daily_pipeline,
)


def _create_sample_paper(
    arxiv_id: str,
    title: str,
    abstract: str,
) -> RawPaper:
    return RawPaper(
        arxiv_id=arxiv_id,
        title=title,
        authors=["Alice Researcher", "Bob Engineer"],
        abstract=abstract,
        categories=["cs.RO", "cs.AI"],
        published_date=datetime(2026, 10, 1, 12, 0, tzinfo=UTC),
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
    )


def _create_sample_insight(
    model_paradigm: str = "Flow-Matching",
    action_conditioned: bool = True,
    eval_environments: list[str] | None = None,
    core_innovation: str = "提出基于连续流匹配的动作闭环世界模型。",
) -> PaperInsight:
    return PaperInsight(
        is_world_model=True,
        model_paradigm=model_paradigm,  # type: ignore[arg-type]
        action_conditioned=action_conditioned,
        eval_environments=eval_environments or ["CALVIN", "Real Franka Robot"],
        core_innovation=core_innovation,
    )


# =========================================================================
# 1. MarkdownReporter 模板渲染与格式测试
# =========================================================================


def test_markdown_reporter_rendering_by_paradigm() -> None:
    reporter = MarkdownReporter()

    p1 = _create_sample_paper(
        arxiv_id="2601.00001",
        title="FlowWorld: Flow-Matching for Robot Dynamics",
        abstract="Continuous flow matching world model.",
    )
    ins1 = _create_sample_insight(
        model_paradigm="Flow-Matching",
        action_conditioned=True,
        eval_environments=["CALVIN", "Isaac Lab"],
        core_innovation="首次实现高保真连续流匹配机械臂转移预测。",
    )

    p2 = _create_sample_paper(
        arxiv_id="2601.00002",
        title="DiffWorld: Diffusion Dynamics for Humanoid Navigation",
        abstract="Denoising diffusion visual world model.",
    )
    ins2 = _create_sample_insight(
        model_paradigm="Diffusion-based",
        action_conditioned=False,
        eval_environments=["Habitat"],
        core_innovation="针对人形导航的多视角色散去噪预测模型。",
    )

    records = [
        PaperState(paper=p1, insight=ins1, status="stored", is_world_model=True),
        PaperState(paper=p2, insight=ins2, status="stored", is_world_model=True),
    ]

    stats = {
        "total_fetched": 15,
        "world_models_stored": 2,
        "skipped_unrelated": 12,
        "tool_calls_made": 3,
        "errors": 1,
    }

    markdown = reporter.render(papers=records, stats=stats)

    # 1. 验证总览标题与统计指标表格
    assert "# 🤖 RoboWorld 机器人世界模型每日处理简报" in markdown
    assert "15" in markdown
    assert "2" in markdown
    assert "12" in markdown

    # 2. 验证按模型范式分块归类
    assert "🌊 Flow-Matching 模型范式" in markdown
    assert "🌫️ Diffusion-based 模型范式" in markdown

    # 3. 验证 Action-conditioned 高亮视觉区分
    assert "🟢 **【是】显式支持动作转移预测**" in markdown
    assert "⚪ 否" in markdown

    # 4. 验证评测环境与创新点
    assert "`CALVIN`" in markdown
    assert "`Isaac Lab`" in markdown
    assert "首次实现高保真连续流匹配机械臂转移预测" in markdown

    # 5. 验证 arXiv 链接
    assert "https://arxiv.org/pdf/2601.00001.pdf" in markdown
    assert "https://arxiv.org/abs/2601.00001" in markdown


def test_markdown_reporter_save_report_to_disk(tmp_path: Path) -> None:
    reporter = MarkdownReporter(default_output_dir=tmp_path)
    sample_content = "# Test Daily Report Content\n\nSome body text."

    # 1. 自定义文件名
    file1 = reporter.save_report(
        markdown_content=sample_content,
        output_dir=tmp_path,
        filename="custom_report.md",
    )
    assert file1.is_file()
    assert file1.read_text(encoding="utf-8") == sample_content

    # 2. 默认文件名 (roboworld_daily_YYYY-MM-DD.md)
    file2 = reporter.save_report(
        markdown_content=sample_content,
        output_dir=tmp_path,
    )
    assert file2.is_file()
    assert "roboworld_daily_" in file2.name
    assert file2.name.endswith(".md")


def test_markdown_reporter_send_webhook_disabled() -> None:
    reporter = MarkdownReporter()
    # 软件推送功能已暂时下线，无论输入何种 webhook_url 均应平滑返回 False 且不发出网络请求
    with patch("requests.post") as mock_post:
        res = reporter.send_webhook(
            markdown_content="# Summary Text",
            webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/xxx",
        )
        assert res is False
        assert mock_post.call_count == 0

    assert reporter.send_webhook(markdown_content="Test", webhook_url="") is False


def test_markdown_reporter_generate_and_save(tmp_path: Path) -> None:
    reporter = MarkdownReporter(default_output_dir=tmp_path)
    p = _create_sample_paper("2601.12345", "Test Title", "Abstract")
    ins = _create_sample_insight()

    with patch.object(reporter, "send_webhook", return_value=True):
        saved_file, md_text, webhook_sent = reporter.generate_and_save(
            papers=[(p, ins)],
            output_dir=tmp_path,
            filename="full_test.md",
            webhook_url="https://example.com/webhook",
        )

        assert saved_file.is_file()
        assert saved_file.name == "full_test.md"
        assert "Test Title" in md_text
        assert webhook_sent is True


# =========================================================================
# 2. Scheduler 定时调度与流水线集成测试
# =========================================================================


def test_generate_crontab_command() -> None:
    cmd = generate_crontab_command(
        cron_schedule="30 7 * * *",
        project_dir="/custom/roboworld",
        python_bin="/custom/venv/bin/python",
        log_file="/custom/log.txt",
    )
    assert "30 7 * * *" in cmd
    assert "cd /custom/roboworld" in cmd
    assert "/custom/venv/bin/python -m src.workflows.scheduler --run-now" in cmd
    assert ">> /custom/log.txt 2>&1" in cmd


def test_scheduler_add_daily_job() -> None:
    scheduler = RoboWorldScheduler(blocking=False)

    # 1. 按小时分钟注册
    job1_id = scheduler.add_daily_job(hour=8, minute=30, job_id="job1")
    assert job1_id == "job1"
    job1 = scheduler.scheduler.get_job("job1")
    assert job1 is not None

    # 2. 按标准 Crontab 表达式注册
    job2_id = scheduler.add_daily_job(cron_expr="0 9 * * 1-5", job_id="job2")
    assert job2_id == "job2"
    job2 = scheduler.scheduler.get_job("job2")
    assert job2 is not None

    # 3. 启动与停止
    scheduler.start()
    assert scheduler.scheduler.running is True
    scheduler.shutdown(wait=False)
    assert scheduler.scheduler.running is False


def test_acceptance_criteria_daily_report_artifact(tmp_path: Path) -> None:
    """验收标准测试：单次流水线结束后，能在指定输出目录产出结构优雅的日报文档。"""
    db = DBManager(db_path=":memory:")
    vstore = VectorStore(
        storage_path=":memory:", collection_name="test_pipeline_report"
    )
    output_dir = tmp_path / "daily_reports"

    p1 = _create_sample_paper(
        arxiv_id="2601.55555",
        title="RobotWorld: Fast Action-Conditioned Latent Dynamics",
        abstract="We present an action-conditioned world model on CALVIN and real robots.",
    )
    p2 = _create_sample_paper(
        arxiv_id="2601.66666",
        title="Unrelated Financial Sentiment Analysis",
        abstract="Stock market predictions with sentiment analysis.",
    )

    # 模拟抓取返回预设论文
    with patch(
        "src.workflows.graph.fetch_papers",
        return_value=[p1, p2],
    ):
        result = run_daily_pipeline(
            days_back=1,
            max_results=10,
            output_dir=output_dir,
            filename="acceptance_report.md",
            db_manager=db,
            vector_store=vstore,
        )

    # 1. 验证返回数据结构
    assert "report_path" in result
    assert "report_markdown" in result
    assert "stats" in result

    # 2. 验收关键要求：在指定输出目录产出日报文档
    report_file = Path(result["report_path"])
    assert report_file.is_file()
    assert report_file.parent == output_dir.resolve()
    assert report_file.name == "acceptance_report.md"

    # 3. 验证文档结构优雅、包含归类与字段
    content = report_file.read_text(encoding="utf-8")
    assert "# 🤖 RoboWorld 机器人世界模型每日处理简报" in content
    assert "📊 今日收录与处理概览" in content
    assert "RobotWorld: Fast Action-Conditioned Latent Dynamics" in content
    assert "动作闭环 (Action-Conditioned)" in content
    assert "验证环境与基准" in content
    assert "💡 核心创新概述" in content
    assert "📥 下载论文 PDF" in content
