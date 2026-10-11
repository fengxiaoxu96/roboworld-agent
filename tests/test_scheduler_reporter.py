"""Unit and integration tests: Jinja2 briefing generation and cron scheduler (Task 5.1)."""

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
    core_innovation: str = "Proposes an action-conditioned world model based on continuous flow matching.",
) -> PaperInsight:
    return PaperInsight(
        is_world_model=True,
        model_paradigm=model_paradigm,  # type: ignore[arg-type]
        action_conditioned=action_conditioned,
        eval_environments=eval_environments or ["CALVIN", "Real Franka Robot"],
        core_innovation=core_innovation,
    )


# =========================================================================
# 1. MarkdownReporter template rendering and format tests
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
        core_innovation="First high-fidelity continuous flow matching robot arm transition prediction.",
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
        core_innovation="Multi-view chromatic denoising prediction model for humanoid navigation.",
    )

    records = [
        PaperState(paper=p1, insight=ins1, status="stored", is_world_model=True),
        PaperState(paper=p2, insight=ins2, status="stored", is_world_model=True),
    ]

    stats = {
        "scanned_count": 88,
        "total_fetched": 15,
        "world_models_stored": 2,
        "skipped_unrelated": 12,
        "tool_calls_made": 3,
        "errors": 1,
        "filter_keywords": ["world[\\s-]?models?", "latent[\\s-]?dynamics"],
    }

    markdown = reporter.render(papers=records, stats=stats)

    # 1. Verify overview title and metrics table
    assert "# 🤖 RoboWorld Robotics World Models Daily Briefing" in markdown
    assert "88" in markdown
    assert "15" in markdown
    assert "2" in markdown
    assert "12" in markdown
    assert "Pre-filtering Rule Note" in markdown
    assert "world model, latent dynamics" in markdown

    # 2. Verify grouped by model paradigm
    assert "🌊 Flow-Matching Paradigm" in markdown
    assert "🌫️ Diffusion-based Paradigm" in markdown

    # 3. Verify Action-conditioned visual highlight
    assert "🟢 **[Yes] Explicitly supports action transition prediction**" in markdown
    assert "⚪ No" in markdown

    # 4. Verify evaluation environments and innovation summary
    assert "`CALVIN`" in markdown
    assert "`Isaac Lab`" in markdown
    assert (
        "First high-fidelity continuous flow matching robot arm transition prediction."
        in markdown
    )

    # 5. Verify arXiv links
    assert "https://arxiv.org/pdf/2601.00001.pdf" in markdown
    assert "https://arxiv.org/abs/2601.00001" in markdown


def test_markdown_reporter_save_report_to_disk(tmp_path: Path) -> None:
    reporter = MarkdownReporter(default_output_dir=tmp_path)
    sample_content = "# Test Daily Report Content\n\nSome body text."

    # 1. Custom filename
    file1 = reporter.save_report(
        markdown_content=sample_content,
        output_dir=tmp_path,
        filename="custom_report.md",
    )
    assert file1.is_file()
    assert file1.read_text(encoding="utf-8") == sample_content

    # 2. Default filename (roboworld_daily_YYYY-MM-DD.md)
    file2 = reporter.save_report(
        markdown_content=sample_content,
        output_dir=tmp_path,
    )
    assert file2.is_file()
    assert "roboworld_daily_" in file2.name
    assert file2.name.endswith(".md")


def test_markdown_reporter_send_webhook_disabled() -> None:
    reporter = MarkdownReporter()
    # Software push service suspended; returns False smoothly without sending requests
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
# 2. Scheduler timing and pipeline integration tests
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

    # 1. Register by hour/minute
    job1_id = scheduler.add_daily_job(hour=8, minute=30, job_id="job1")
    assert job1_id == "job1"
    job1 = scheduler.scheduler.get_job("job1")
    assert job1 is not None

    # 2. Register by crontab expression
    job2_id = scheduler.add_daily_job(cron_expr="0 9 * * 1-5", job_id="job2")
    assert job2_id == "job2"
    job2 = scheduler.scheduler.get_job("job2")
    assert job2 is not None

    # 3. Start and stop
    scheduler.start()
    assert scheduler.scheduler.running is True
    scheduler.shutdown(wait=False)
    assert scheduler.scheduler.running is False


def test_acceptance_criteria_daily_report_artifact(tmp_path: Path) -> None:
    """Acceptance criteria test: produces well-structured daily report artifact in output dir after pipeline run."""
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

    # Mock fetch returning preset papers
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

    # 1. Verify return data structure
    assert "report_path" in result
    assert "report_markdown" in result
    assert "stats" in result

    # 2. Acceptance requirement: output daily report artifact in specified output directory
    report_file = Path(result["report_path"])
    assert report_file.is_file()
    assert report_file.parent == output_dir.resolve()
    assert report_file.name == "acceptance_report.md"

    # 3. Verify structured document containing classifications and fields
    content = report_file.read_text(encoding="utf-8")
    assert "# 🤖 RoboWorld Robotics World Models Daily Briefing" in content
    assert "📊 Today's Processing Overview" in content
    assert "RobotWorld: Fast Action-Conditioned Latent Dynamics" in content
    assert "Action-Conditioned" in content
    assert "Evaluation Environments & Benchmarks" in content
    assert "💡 Core Innovation Summary" in content
    assert "📥 Download PDF" in content
