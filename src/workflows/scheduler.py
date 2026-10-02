"""定时调度器与早报自动化流水线 (Task 5.1).

主要职责：
1. 封装端到端定时任务 `run_daily_pipeline`：
   - 触发 LangGraph 主状态图拉取、初筛、分类、正文调阅、双库落库；
   - 调用 Jinja2 MarkdownReporter 渲染并持久化结构化简报；
   - 支持 Webhook（飞书/钉钉/企微/Slack）推送；
2. 基于 APScheduler 提供系统级定时触发（支持标准 5 位 Crontab 表达式或每日指定时间）；
3. 提供 Linux Crontab 指令生成器，支持原生系统定时运行；
4. 提供 CLI 命令行入口，支持 `--run-now` 一键即时运行与 `--daemon` 常驻调度。
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
from src.fetchers.arxiv_fetcher import DEFAULT_CATEGORIES
from src.storage.db_manager import DBManager
from src.storage.vector_store import VectorStore
from src.tools.reporter import MarkdownReporter
from src.workflows.graph import WorkflowState, build_workflow_graph

from src.evaluation.tracing import TracingManager, setup_tracing

load_dotenv()

logger = logging.getLogger(__name__)

# 默认调度时间：每日早间 08:00 (UTC)
DEFAULT_SCHEDULE_HOUR = int(os.getenv("SCHEDULE_HOUR", "8"))
DEFAULT_SCHEDULE_MINUTE = int(os.getenv("SCHEDULE_MINUTE", "0"))
DEFAULT_CRON_SCHEDULE = os.getenv("CRON_SCHEDULE")


def run_daily_pipeline(
    days_back: int = 1,
    max_results: int = 50,
    categories: Sequence[str] | None = None,
    output_dir: Path | str | None = None,
    filename: str | None = None,
    webhook_url: str | None = None,
    db_manager: DBManager | None = None,
    vector_store: VectorStore | None = None,
    classifier: ClassifierAgent | None = None,
    tracing_manager: TracingManager | None = None,
) -> dict[str, Any]:
    """执行单次完整的每日世界模型文献自动化研判与简报生成流水线。

    执行流程：
    1. 构建并调用 LangGraph 编排工作流（Fetch -> Filter -> ToolCall -> Extract -> Storage -> Report）；
    2. 全流程挂载 LangSmith 与 TokenUsageTracker 可观测性追踪回调；
    3. 提取处理产物与统计指标；
    4. 调用 Jinja2 渲染规范的 Markdown 简报，并落盘至指定输出目录；
    5. 若配置 Webhook 则同步推送通知；
    6. 返回包含报告路径、内容、运行状态与 Token 监控指标的汇总字典。
    """
    logger.info("=== 启动 RoboWorld 每日自动化早报流水线 ===")
    start_time = datetime.now(UTC)

    # 1. 初始化可观测性追踪管理
    tracer = tracing_manager or setup_tracing()
    run_config = tracer.get_langgraph_config()

    # 2. 构建并执行 LangGraph 图
    app = build_workflow_graph(
        db_manager=db_manager,
        vector_store=vector_store,
        classifier=classifier,
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

    # 3. 渲染并落盘 Markdown 简报
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
        "流水线执行完毕 (耗时: %.2fs, Tokens: %d)，报告已生成: %s，Webhook状态: %s",
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
    """基于 APScheduler 的定时任务管理器。"""

    def __init__(
        self,
        db_manager: DBManager | None = None,
        vector_store: VectorStore | None = None,
        classifier: ClassifierAgent | None = None,
        blocking: bool = False,
    ) -> None:
        self.db = db_manager
        self.vector_store = vector_store
        self.classifier = classifier
        self.blocking = blocking

        # 根据模式选择调度器类型
        if blocking:
            self.scheduler: Any = BlockingScheduler()
        else:
            self.scheduler = BackgroundScheduler()

    def add_daily_job(
        self,
        hour: int = DEFAULT_SCHEDULE_HOUR,
        minute: int = DEFAULT_SCHEDULE_MINUTE,
        cron_expr: str | None = DEFAULT_CRON_SCHEDULE,
        days_back: int = 1,
        max_results: int = 50,
        output_dir: Path | str | None = None,
        webhook_url: str | None = None,
        job_id: str = "daily_roboworld_pipeline",
    ) -> str:
        """注册每日定时执行任务。"""
        if cron_expr:
            trigger = CronTrigger.from_crontab(cron_expr)
            logger.info("注册定时任务 [%s]，触发规则 (Crontab): %s", job_id, cron_expr)
        else:
            trigger = CronTrigger(hour=hour, minute=minute)
            logger.info(
                "注册定时任务 [%s]，触发时间: 每日 %02d:%02d UTC", job_id, hour, minute
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
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("定时任务执行发生未捕获异常: %s", exc)

        self.scheduler.add_job(
            _job_wrapper,
            trigger=trigger,
            id=job_id,
            replace_existing=True,
            name="RoboWorld Daily Pipeline",
        )
        return job_id

    def start(self) -> None:
        """启动定时调度器。"""
        logger.info("启动 RoboWorld 定时调度器 (blocking=%s)...", self.blocking)
        self.scheduler.start()

    def shutdown(self, wait: bool = True) -> None:
        """关闭定时调度器。"""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=wait)
            logger.info("RoboWorld 定时调度器已停止。")

    def run_once_now(
        self,
        days_back: int = 1,
        max_results: int = 50,
        output_dir: Path | str | None = None,
        webhook_url: str | None = None,
    ) -> dict[str, Any]:
        """立即执行单次流水线。"""
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
    """生成规范的 Linux Crontab 定时配置命令。

    便于用户直接将流水线配置至操作系统的 Crontab 守护进程中。
    """
    proj_path = Path(project_dir or Path(__file__).resolve().parents[2]).resolve()
    py_path = python_bin or (proj_path / ".venv" / "bin" / "python")
    log_path = log_file or (proj_path / "data" / "reports" / "cron_run.log")

    return (
        f"{cron_schedule} cd {proj_path} && {py_path} -m src.workflows.scheduler "
        f"--run-now >> {log_path} 2>&1"
    )


def main(argv: list[str] | None = None) -> None:
    """CLI 入口，支持手动运行与定时守护。"""
    parser = argparse.ArgumentParser(
        description="RoboWorld 每日学术简报自动化调度与生成系统 (Task 5.1)"
    )
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="立即执行一次抓取、研判、落库与简报生成",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="启动常驻守护调度器（阻塞运行）",
    )
    parser.add_argument(
        "--cron",
        type=str,
        default=None,
        help="指定自定义 Crontab 表达式 (例如 '0 8 * * *')",
    )
    parser.add_argument(
        "--generate-crontab",
        action="store_true",
        help="输出适用于 Linux 系统 crontab -e 的定时配置命令",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=1,
        help="回溯最近几天更新的论文 (默认: 1)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=50,
        help="最多检索多少篇论文 (默认: 50)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="报告本地保存目录 (默认: data/reports)",
    )
    parser.add_argument(
        "--webhook-url",
        type=str,
        default=None,
        help="Webhook 推送目标地址",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.generate_crontab:
        cron_expr = args.cron or "0 8 * * *"
        cmd = generate_crontab_command(cron_schedule=cron_expr)
        print("\n# 将以下行添加到系统 Crontab (运行 `crontab -e`):")
        print(cmd)
        print()
        sys.exit(0)

    if args.run_now:
        result = run_daily_pipeline(
            days_back=args.days_back,
            max_results=args.max_results,
            output_dir=args.output_dir,
            webhook_url=args.webhook_url,
        )
        print(f"\n✅ 报告生成成功: {result['report_path']}")
        print(f"📊 统计概览: {result['stats']}")
        sys.exit(0)

    if args.daemon:
        sched = RoboWorldScheduler(blocking=True)
        sched.add_daily_job(
            cron_expr=args.cron,
            days_back=args.days_back,
            max_results=args.max_results,
            output_dir=args.output_dir,
            webhook_url=args.webhook_url,
        )
        print("🤖 RoboWorld 定时调度守护进程已启动，按 Ctrl+C 退出...")
        try:
            sched.start()
        except (KeyboardInterrupt, SystemExit):
            sched.shutdown()
            print("调度器已正常安全退出。")
            sys.exit(0)

    # 默认未指定参数时打印帮助
    parser.print_help()


if __name__ == "__main__":
    main()
