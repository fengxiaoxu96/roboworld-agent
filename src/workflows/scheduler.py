"""Scheduled runner and daily briefing automated pipeline (Task 5.1).

Main responsibilities:
1. Encapsulate end-to-end scheduled task `run_daily_pipeline`:
   - Trigger LangGraph main state graph fetching, pre-filtering, classification, full-text retrieval, dual storage;
   - Call Jinja2 MarkdownReporter to render and persist structured briefing;
   - Support Webhook (Feishu/DingTalk/WeCom/Slack) notifications;
2. Provide system-level scheduled triggers based on APScheduler (supporting standard 5-part Crontab expressions or daily specific times);
3. Provide Linux Crontab command generator for native OS scheduling;
4. Provide CLI command-line entry point supporting `--run-now` immediate run and `--daemon` resident scheduling.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from apscheduler.schedulers.background import (
    BackgroundScheduler,  # type: ignore[import-untyped]
)
from apscheduler.schedulers.blocking import (
    BlockingScheduler,  # type: ignore[import-untyped]
)
from apscheduler.triggers.cron import CronTrigger  # type: ignore[import-untyped]
from dotenv import load_dotenv

from src.agents.classifier_agent import ClassifierAgent
from src.evaluation.tracing import TracingManager, setup_tracing
from src.fetchers.arxiv_fetcher import DEFAULT_CATEGORIES, resolve_fetch_limits
from src.storage.db_manager import DBManager
from src.storage.vector_store import VectorStore
from src.tools.reporter import MarkdownReporter
from src.workflows.graph import WorkflowState, build_workflow_graph

load_dotenv()

logger = logging.getLogger(__name__)

# Default schedule time: 08:00 UTC daily
DEFAULT_SCHEDULE_HOUR = int(os.getenv("SCHEDULE_HOUR", "8"))
DEFAULT_SCHEDULE_MINUTE = int(os.getenv("SCHEDULE_MINUTE", "0"))
DEFAULT_CRON_SCHEDULE = os.getenv("CRON_SCHEDULE")


def run_daily_pipeline(
    days_back: int | None = None,
    max_results: int | None = None,
    categories: Sequence[str] | None = None,
    output_dir: Path | str | None = None,
    filename: str | None = None,
    webhook_url: str | None = None,
    db_manager: DBManager | None = None,
    vector_store: VectorStore | None = None,
    classifier: ClassifierAgent | None = None,
    tracing_manager: TracingManager | None = None,
    llm: Any | None = None,
) -> dict[str, Any]:
    """Execute a single full run of daily robotics world model literature evaluation and briefing pipeline.

    Execution flow:
    1. Build and invoke LangGraph orchestration workflow (Fetch -> Filter -> ToolCall -> Extract -> Storage -> Report);
    2. Attach LangSmith and TokenUsageTracker observability tracing callbacks throughout;
    3. Extract processed items and statistical metrics;
    4. Call Jinja2 to render standard Markdown briefing and persist to output directory;
    5. Push Webhook notification if configured;
    6. Return summary dictionary containing report path, content, run status, and token monitoring metrics.
    """
    days_back, max_results = resolve_fetch_limits(days_back, max_results)
    logger.info("=== Starting RoboWorld daily automated briefing pipeline ===")
    start_time = datetime.now(UTC)

    # 1. Initialize observability tracing
    tracer = tracing_manager or setup_tracing()
    run_config = tracer.get_langgraph_config()

    # 2. Build and execute LangGraph state graph
    app = build_workflow_graph(
        db_manager=db_manager,
        vector_store=vector_store,
        classifier=classifier,
        llm=llm,
    )

    initial_state: WorkflowState = {
        "days_back": days_back,
        "max_results": max_results,
        "categories": list(categories or DEFAULT_CATEGORIES),
        "execution_trace": [],
    }

    final_state: dict[str, Any] = app.invoke(initial_state, config=run_config)

    processed_papers = final_state.get("processed_papers", [])
    stats = final_state.get("final_report", {})

    # 3. Render and persist Markdown briefing
    reporter = MarkdownReporter(default_output_dir=output_dir)
    skipped_list = [
        r for r in processed_papers if getattr(r, "status", "") == "skipped"
    ]

    report_path, report_md, webhook_ok = reporter.generate_and_save(
        papers=processed_papers,
        stats=stats,
        skipped_records=skipped_list,
        output_dir=output_dir,
        filename=filename,
        webhook_url=webhook_url,
    )

    elapsed_seconds = (datetime.now(UTC) - start_time).total_seconds()
    token_summary = tracer.tracker.get_summary()
    logger.info(
        "Pipeline finished (elapsed: %.2fs, tokens: %d), report generated: %s, webhook status: %s",
        elapsed_seconds,
        token_summary.get("total_tokens", 0),
        report_path,
        webhook_ok,
    )

    return {
        "report_path": report_path,
        "report_markdown": report_md,
        "stats": stats,
        "webhook_sent": webhook_ok,
        "elapsed_seconds": elapsed_seconds,
        "processed_count": len(processed_papers),
        "token_usage": token_summary,
        "trace_summary": token_summary,
    }


class RoboWorldScheduler:
    """Scheduled job manager based on APScheduler."""

    def __init__(
        self,
        db_manager: DBManager | None = None,
        vector_store: VectorStore | None = None,
        classifier: ClassifierAgent | None = None,
        blocking: bool = False,
        llm: Any | None = None,
    ) -> None:
        self.db = db_manager
        self.vector_store = vector_store
        self.classifier = classifier
        self.blocking = blocking
        self.llm = llm

        # Select scheduler type based on mode
        if blocking:
            self.scheduler: Any = BlockingScheduler()
        else:
            self.scheduler = BackgroundScheduler()

    def add_daily_job(
        self,
        hour: int = DEFAULT_SCHEDULE_HOUR,
        minute: int = DEFAULT_SCHEDULE_MINUTE,
        cron_expr: str | None = DEFAULT_CRON_SCHEDULE,
        days_back: int | None = None,
        max_results: int | None = None,
        output_dir: Path | str | None = None,
        webhook_url: str | None = None,
        job_id: str = "daily_roboworld_pipeline",
    ) -> str:
        """Register daily scheduled execution job."""
        if cron_expr:
            trigger = CronTrigger.from_crontab(cron_expr)
            logger.info(
                "Register scheduled job [%s], trigger rule (Crontab): %s",
                job_id,
                cron_expr,
            )
        else:
            trigger = CronTrigger(hour=hour, minute=minute)
            logger.info(
                "Register scheduled job [%s], trigger time: daily %02d:%02d UTC",
                job_id,
                hour,
                minute,
            )

        def _job_wrapper() -> None:
            try:
                run_daily_pipeline(
                    days_back=days_back,
                    max_results=max_results,
                    output_dir=output_dir,
                    webhook_url=webhook_url,
                    db_manager=self.db,
                    vector_store=self.vector_store,
                    classifier=self.classifier,
                    llm=self.llm,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Uncaught exception in scheduled job execution: %s", exc)

        self.scheduler.add_job(
            _job_wrapper,
            trigger=trigger,
            id=job_id,
            replace_existing=True,
            name="RoboWorld Daily Pipeline",
        )
        return job_id

    def start(self) -> None:
        """Start scheduled runner."""
        logger.info("Starting RoboWorld scheduler (blocking=%s)...", self.blocking)
        self.scheduler.start()

    def shutdown(self, wait: bool = True) -> None:
        """Stop scheduled runner."""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=wait)
            logger.info("RoboWorld scheduler stopped.")

    def run_once_now(
        self,
        days_back: int | None = None,
        max_results: int | None = None,
        output_dir: Path | str | None = None,
        webhook_url: str | None = None,
    ) -> dict[str, Any]:
        """Execute a single pipeline run immediately."""
        return run_daily_pipeline(
            days_back=days_back,
            max_results=max_results,
            output_dir=output_dir,
            webhook_url=webhook_url,
            db_manager=self.db,
            vector_store=self.vector_store,
            classifier=self.classifier,
        )


def generate_crontab_command(
    cron_schedule: str = "0 8 * * *",
    project_dir: Path | str | None = None,
    python_bin: str | None = None,
    log_file: str | None = None,
) -> str:
    """Generate standard Linux Crontab configuration command.

    Allows convenient configuration directly into the OS Crontab daemon.
    """
    proj_path = Path(project_dir or Path(__file__).resolve().parents[2]).resolve()
    py_path = python_bin or (proj_path / ".venv" / "bin" / "python")
    log_path = log_file or (proj_path / "data" / "reports" / "cron_run.log")

    return (
        f"{cron_schedule} cd {proj_path} && {py_path} -m src.workflows.scheduler "
        f"--run-now >> {log_path} 2>&1"
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point supporting manual execution and scheduled daemon."""
    parser = argparse.ArgumentParser(
        description="RoboWorld daily academic briefing automated scheduler and generator (Task 5.1)"
    )
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Immediately execute fetch, evaluation, persistence, and briefing generation",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Start resident daemon scheduler (blocking)",
    )
    parser.add_argument(
        "--cron",
        type=str,
        default=None,
        help="Specify custom Crontab expression (e.g. '0 8 * * *')",
    )
    parser.add_argument(
        "--generate-crontab",
        action="store_true",
        help="Output scheduled command suitable for Linux crontab -e",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=None,
        help="Days to look back based on updated date; consumes full window when used alone",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Maximum papers to keep; omitted means no count truncation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Local directory to save reports (default: data/reports)",
    )
    parser.add_argument(
        "--webhook-url",
        type=str,
        default=None,
        help="Target Webhook push URL",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.generate_crontab:
        cron_expr = args.cron or "0 8 * * *"
        cmd = generate_crontab_command(cron_schedule=cron_expr)
        print("\n# Add the following line to system Crontab (run `crontab -e`):")
        print(cmd)
        print()
        sys.exit(0)

    if args.run_now:
        days_back, max_results = resolve_fetch_limits(args.days_back, args.max_results)
        result = run_daily_pipeline(
            days_back=days_back,
            max_results=max_results,
            output_dir=args.output_dir,
            webhook_url=args.webhook_url,
        )
        print(f"\n✅ Report generated successfully: {result['report_path']}")
        print(f"📊 Statistics summary: {result['stats']}")
        sys.exit(0)

    if args.daemon:
        days_back, max_results = resolve_fetch_limits(args.days_back, args.max_results)
        sched = RoboWorldScheduler(blocking=True)
        sched.add_daily_job(
            cron_expr=args.cron,
            days_back=days_back,
            max_results=max_results,
            output_dir=args.output_dir,
            webhook_url=args.webhook_url,
        )
        print("🤖 RoboWorld scheduler daemon started, press Ctrl+C to exit...")
        try:
            sched.start()
        except (KeyboardInterrupt, SystemExit):
            sched.shutdown()
            print("Scheduler exited cleanly.")
            sys.exit(0)

    # Print help by default if no flags provided
    parser.print_help()


if __name__ == "__main__":
    main()
