"""Automated Eval evaluation engine and metric computation (Task 5.3).

Main functions:
1. Batch run ClassifierAgent classification and extraction across Golden Dataset;
2. Compute confusion matrix (TP, FP, TN, FN), Accuracy, Precision, Recall, F1-Score;
3. Calculate model paradigm and action conditioning prediction accuracy;
4. Integrated with TokenUsageTracker throughout to record token consumption and inference latency;
5. Output standard Markdown evaluation report with CI threshold gating (passes_threshold).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.agents.classifier_agent import ClassifierAgent
from src.core.schema import PaperInsight, RawPaper
from src.evaluation.dataset import GoldenPaperSample
from src.evaluation.tracing import TokenUsageTracker

logger = logging.getLogger(__name__)


class SampleEvaluationDetail(BaseModel):
    """Detailed evaluation record for a single sample."""

    model_config = ConfigDict(extra="ignore")

    arxiv_id: str
    title: str
    ground_truth_is_wm: bool
    predicted_is_wm: bool
    is_correct: bool
    confidence: str
    reasoning: str
    gt_paradigm: str
    pred_paradigm: str | None = None
    gt_action_cond: bool
    pred_action_cond: bool | None = None
    latency_seconds: float = 0.0


class EvaluationResult(BaseModel):
    """Evaluation metrics and results summary."""

    model_config = ConfigDict(extra="ignore")

    total_samples: int = Field(..., description="Total evaluation sample count")
    tp: int = Field(0, description="True Positives")
    fp: int = Field(0, description="False Positives")
    tn: int = Field(0, description="True Negatives")
    fn: int = Field(0, description="False Negatives")

    accuracy: float = Field(0.0, description="Classification accuracy")
    precision: float = Field(0.0, description="Precision")
    recall: float = Field(0.0, description="Recall")
    f1_score: float = Field(0.0, description="F1 Score")

    paradigm_accuracy: float = Field(
        0.0, description="Model paradigm prediction accuracy"
    )
    action_conditioned_accuracy: float = Field(
        0.0, description="Action conditioning accuracy"
    )

    total_latency_seconds: float = Field(0.0, description="Total latency in seconds")
    average_latency_seconds: float = Field(
        0.0, description="Average latency per paper in seconds"
    )

    token_usage: dict[str, Any] = Field(
        default_factory=dict, description="Evaluation token usage statistics"
    )
    evaluated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    details: list[SampleEvaluationDetail] = Field(
        default_factory=list, description="Per-sample evaluation details"
    )

    def passes_threshold(self, min_accuracy: float = 0.90) -> bool:
        """Check if accuracy meets acceptance threshold (default 90%)."""
        return self.accuracy >= min_accuracy

    def to_markdown_report(self) -> str:
        """Generate standardized Markdown evaluation report."""
        status_badge = (
            "✅ PASS (>=90%)" if self.passes_threshold(0.90) else "❌ FAIL (<90%)"
        )

        lines: list[str] = [
            "# 🤖 RoboWorld Agent - Automated Evaluation Report",
            "",
            f"- **Evaluation Time**: `{self.evaluated_at}`",
            f"- **Acceptance Status**: **{status_badge}**",
            f"- **Total Samples**: `{self.total_samples}` papers",
            "",
            "## 1. Classification Metrics",
            "",
            "| Metric | Value | Baseline | Status |",
            "| :--- | :---: | :---: | :---: |",
            f"| **Accuracy** | **`{self.accuracy * 100:.2f}%`** | `>= 90.00%` | {'✅ Met' if self.accuracy >= 0.90 else '❌ Unmet'} |",
            f"| **Precision** | `{self.precision * 100:.2f}%` | - | - |",
            f"| **Recall** | `{self.recall * 100:.2f}%` | - | - |",
            f"| **F1-Score** | `{self.f1_score:.4f}` | - | - |",
            f"| **Paradigm Acc** | `{self.paradigm_accuracy * 100:.2f}%` | - | - |",
            f"| **Action Acc** | `{self.action_conditioned_accuracy * 100:.2f}%` | - | - |",
            "",
            "## 2. Confusion Matrix",
            "",
            "| Actual \\ Predicted | Predicted World Model (Positive) | Predicted Non-World Model (Negative) | Total |",
            "| :--- | :---: | :---: | :---: |",
            f"| **Actual World Model (True)** | **TP = {self.tp}** | FN = {self.fn} | `{self.tp + self.fn}` |",
            f"| **Actual Non-World Model (False)** | FP = {self.fp} | **TN = {self.tn}** | `{self.fp + self.tn}` |",
            f"| **Total Predicted** | `{self.tp + self.fp}` | `{self.fn + self.tn}` | **`{self.total_samples}`** |",
            "",
            "## 3. Observability and Token Usage Monitoring",
            "",
            f"- **Total Latency**: `{self.total_latency_seconds:.2f} s` (Average per paper: `{self.average_latency_seconds:.3f} s`)",
            f"- **Prompt Tokens**: `{self.token_usage.get('prompt_tokens', 0)}`",
            f"- **Completion Tokens**: `{self.token_usage.get('completion_tokens', 0)}`",
            f"- **Total Tokens Consumed**: `{self.token_usage.get('total_tokens', 0)}`",
            f"- **LLM Call Count**: `{self.token_usage.get('llm_call_count', 0)}`",
            "",
            "## 4. Sample-by-Sample Evaluation Details",
            "",
            "| arXiv ID | Title | Actual | Predicted | Result | Confidence | Paradigm (GT / Pred) |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
        ]

        for d in self.details:
            correct_icon = "✅ Correct" if d.is_correct else "❌ Error"
            gt_str = "WM" if d.ground_truth_is_wm else "Non-WM"
            pred_str = "WM" if d.predicted_is_wm else "Non-WM"
            title_trunc = d.title[:35] + "..." if len(d.title) > 35 else d.title
            paradigm_comp = f"{d.gt_paradigm} / {d.pred_paradigm or 'N/A'}"
            lines.append(
                f"| [`{d.arxiv_id}`](https://arxiv.org/abs/{d.arxiv_id}) | "
                f"{title_trunc} | `{gt_str}` | `{pred_str}` | {correct_icon} | "
                f"`{d.confidence}` | {paradigm_comp} |"
            )

        # Misclassified sample analysis
        errors = [d for d in self.details if not d.is_correct]
        if errors:
            lines.extend(
                [
                    "",
                    "## 5. Misclassified Samples Root Cause Analysis",
                    "",
                ]
            )
            for err in errors:
                lines.append(
                    f"- **[{err.arxiv_id}] {err.title}**  \n"
                    f"  - Ground truth: `{err.ground_truth_is_wm}`, Predicted: `{err.predicted_is_wm}`  \n"
                    f"  - Reasoning: {err.reasoning}"
                )

        return "\n".join(lines)


class DatasetEvaluator:
    """Automated Golden Dataset evaluation runner."""

    def __init__(
        self,
        classifier: ClassifierAgent | None = None,
        tracker: TokenUsageTracker | None = None,
    ) -> None:
        self.classifier = classifier or ClassifierAgent()
        self.tracker = tracker or TokenUsageTracker()

    def evaluate(
        self,
        samples: list[GoldenPaperSample],
        section_fetcher: Callable[[str, str], str] | None = None,
    ) -> EvaluationResult:
        """Run batch evaluation and return detailed metrics.

        Args:
            samples: List of Golden Dataset annotated samples.
            section_fetcher: Optional section fetcher function simulating ToolCallNode logic.
        """
        logger.info(
            "Starting Golden Dataset automated evaluation, total %d samples",
            len(samples),
        )
        start_time = time.perf_counter()

        tp = 0
        fp = 0
        tn = 0
        fn = 0

        paradigm_correct = 0
        wm_samples_count = 0
        action_cond_correct = 0

        details: list[SampleEvaluationDetail] = []

        for sample in samples:
            paper: RawPaper = sample.to_raw_paper()
            sample_t0 = time.perf_counter()

            # 1. Abstract initial classification
            decision = self.classifier.classify(paper=paper)

            # 2. If confidence is low and fetcher provided, simulate ToolCall re-evaluation
            section_text: str | None = None
            if (
                decision.confidence == "low" or decision.needs_full_text
            ) and section_fetcher is not None:
                try:
                    section_text = section_fetcher(sample.arxiv_id, "Methodology")
                    decision = self.classifier.classify(
                        paper=paper, section_text=section_text
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "Failed to fetch section, keeping initial filtering decision: %s (%s)",
                        sample.arxiv_id,
                        exc,
                    )

            pred_is_wm = decision.is_world_model
            sample_latency = time.perf_counter() - sample_t0

            # 3. Update confusion matrix
            if sample.is_world_model and pred_is_wm:
                tp += 1
            elif not sample.is_world_model and pred_is_wm:
                fp += 1
            elif not sample.is_world_model and not pred_is_wm:
                tn += 1
            else:
                fn += 1

            is_correct = pred_is_wm == sample.is_world_model

            # 4. Extract structured insight for confirmed world model papers
            pred_insight: PaperInsight | None = None
            if pred_is_wm:
                try:
                    pred_insight = self.classifier.extract_insight(
                        paper=paper, section_text=section_text
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Error extracting insight: %s (%s)", sample.arxiv_id, exc
                    )

            pred_paradigm = pred_insight.model_paradigm if pred_insight else None
            pred_action = pred_insight.action_conditioned if pred_insight else None

            if sample.is_world_model:
                wm_samples_count += 1
                if pred_paradigm == sample.model_paradigm:
                    paradigm_correct += 1
                if pred_action == sample.action_conditioned:
                    action_cond_correct += 1

            details.append(
                SampleEvaluationDetail(
                    arxiv_id=sample.arxiv_id,
                    title=sample.title,
                    ground_truth_is_wm=sample.is_world_model,
                    predicted_is_wm=pred_is_wm,
                    is_correct=is_correct,
                    confidence=decision.confidence,
                    reasoning=decision.reasoning,
                    gt_paradigm=sample.model_paradigm,
                    pred_paradigm=pred_paradigm,
                    gt_action_cond=sample.action_conditioned,
                    pred_action_cond=pred_action,
                    latency_seconds=round(sample_latency, 4),
                )
            )

        total_elapsed = time.perf_counter() - start_time
        total_count = len(samples)

        # Calculate ratios
        accuracy = (tp + tn) / total_count if total_count > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        paradigm_acc = (
            paradigm_correct / wm_samples_count if wm_samples_count > 0 else 0.0
        )
        action_acc = (
            action_cond_correct / wm_samples_count if wm_samples_count > 0 else 0.0
        )
        avg_lat = total_elapsed / total_count if total_count > 0 else 0.0

        return EvaluationResult(
            total_samples=total_count,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            accuracy=round(accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1_score, 4),
            paradigm_accuracy=round(paradigm_acc, 4),
            action_conditioned_accuracy=round(action_acc, 4),
            total_latency_seconds=round(total_elapsed, 4),
            average_latency_seconds=round(avg_lat, 4),
            token_usage=self.tracker.get_summary(),
            details=details,
        )


__all__ = [
    "DatasetEvaluator",
    "EvaluationResult",
    "SampleEvaluationDetail",
]
