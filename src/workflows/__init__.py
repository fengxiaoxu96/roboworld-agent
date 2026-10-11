"""RoboWorld Agent workflow and orchestration engine (Task 4.1 & Task 5.1)."""

from typing import Any

from src.workflows.graph import (
    PaperState,
    WorkflowEngine,
    WorkflowState,
    build_workflow_graph,
    conditional_router,
    dispatcher_router,
)


def __getattr__(name: str) -> Any:
    if name in {"RoboWorldScheduler", "generate_crontab_command", "run_daily_pipeline"}:
        from src.workflows import scheduler

        return getattr(scheduler, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PaperState",
    "RoboWorldScheduler",
    "WorkflowEngine",
    "WorkflowState",
    "build_workflow_graph",
    "conditional_router",
    "dispatcher_router",
    "generate_crontab_command",
    "run_daily_pipeline",
]
