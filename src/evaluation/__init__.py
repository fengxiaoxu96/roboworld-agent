"""Observability tracing, evaluation, and Golden Dataset benchmark module (Task 5.3)."""

from __future__ import annotations

from src.evaluation.dataset import (
    DEFAULT_GOLDEN_DATASET_PATH,
    GoldenPaperSample,
    load_golden_dataset,
    validate_golden_dataset,
)
from src.evaluation.evaluator import (
    DatasetEvaluator,
    EvaluationResult,
    SampleEvaluationDetail,
)
from src.evaluation.tracing import (
    DEFAULT_LANGCHAIN_ENDPOINT,
    DEFAULT_LANGCHAIN_PROJECT,
    TokenRecord,
    TokenUsageTracker,
    TracingManager,
    setup_tracing,
    trace_run,
)

__all__ = [
    "DEFAULT_GOLDEN_DATASET_PATH",
    "DEFAULT_LANGCHAIN_ENDPOINT",
    "DEFAULT_LANGCHAIN_PROJECT",
    "DatasetEvaluator",
    "EvaluationResult",
    "GoldenPaperSample",
    "SampleEvaluationDetail",
    "TokenRecord",
    "TokenUsageTracker",
    "TracingManager",
    "load_golden_dataset",
    "setup_tracing",
    "trace_run",
    "validate_golden_dataset",
]
