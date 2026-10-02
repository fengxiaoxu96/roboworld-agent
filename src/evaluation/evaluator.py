"""自动化 Eval 评测引擎与指标计算 (Task 5.3).

主要功能：
1. 在 Golden Dataset 上批量运行 ClassifierAgent 分类与提取流程；
2. 计算混淆矩阵 (TP, FP, TN, FN)、Accuracy、Precision、Recall、F1-Score；
3. 统计模型范式与动作条件预测准确率；
4. 全流程集成 TokenUsageTracker 记录评估过程中的 Token 消耗与推理延迟；
5. 输出美观且规范的 Markdown 评测报告，并支持 CI 门禁阈值判断 (passes_threshold)。
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
    """单篇样本评估详细记录。"""

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
    """评测指标与结果汇总。"""

    model_config = ConfigDict(extra="ignore")

    total_samples: int = Field(..., description="评测样本总数")
    tp: int = Field(0, description="真阳性 True Positives")
    fp: int = Field(0, description="假阳性 False Positives")
    tn: int = Field(0, description="真阴性 True Negatives")
    fn: int = Field(0, description="假阴性 False Negatives")

    accuracy: float = Field(0.0, description="分类准确率")
    precision: float = Field(0.0, description="精确率")
    recall: float = Field(0.0, description="召回率")
    f1_score: float = Field(0.0, description="F1 分数")

    paradigm_accuracy: float = Field(0.0, description="模型范式预测准确率")
    action_conditioned_accuracy: float = Field(0.0, description="动作条件识别准确率")

    total_latency_seconds: float = Field(0.0, description="总耗时(秒)")
    average_latency_seconds: float = Field(0.0, description="单篇平均耗时(秒)")

    token_usage: dict[str, Any] = Field(
        default_factory=dict, description="评估过程 Token 消耗统计"
    )
    evaluated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    details: list[SampleEvaluationDetail] = Field(
        default_factory=list, description="单篇评估明细"
    )

    def passes_threshold(self, min_accuracy: float = 0.90) -> bool:
        """检查准确率是否达到验收阈值（默认 90%）。"""
        return self.accuracy >= min_accuracy

    def to_markdown_report(self) -> str:
        """生成格式规范的 Markdown 评测报告。"""
        status_badge = (
            "✅ PASS (>=90%)" if self.passes_threshold(0.90) else "❌ FAIL (<90%)"
        )

        lines: list[str] = [
            "# 🤖 RoboWorld Agent - 自动化研判模型 Eval 评估报告",
            "",
            f"- **评估时间**: `{self.evaluated_at}`",
            f"- **验收状态**: **{status_badge}**",
            f"- **样本总数**: `{self.total_samples}` 篇",
            "",
            "## 1. 核心分类性能指标 (Classification Metrics)",
            "",
            "| 指标名称 | 计算值 | 验收基线 | 状态 |",
            "| :--- | :---: | :---: | :---: |",
            f"| **准确率 (Accuracy)** | **`{self.accuracy * 100:.2f}%`** | `>= 90.00%` | {'✅ 达标' if self.accuracy >= 0.90 else '❌ 未达标'} |",
            f"| **精确率 (Precision)** | `{self.precision * 100:.2f}%` | - | - |",
            f"| **召回率 (Recall)** | `{self.recall * 100:.2f}%` | - | - |",
            f"| **F1 分数 (F1-Score)** | `{self.f1_score:.4f}` | - | - |",
            f"| **范式识别率 (Paradigm Acc)** | `{self.paradigm_accuracy * 100:.2f}%` | - | - |",
            f"| **动作闭环识别率 (Action Acc)** | `{self.action_conditioned_accuracy * 100:.2f}%` | - | - |",
            "",
            "## 2. 混淆矩阵 (Confusion Matrix)",
            "",
            "| 实际 \\ 预测 | 预测为世界模型 (Positive) | 预测为非世界模型 (Negative) | 总计 |",
            "| :--- | :---: | :---: | :---: |",
            f"| **实际世界模型 (True)** | **TP = {self.tp}** | FN = {self.fn} | `{self.tp + self.fn}` |",
            f"| **实际非世界模型 (False)** | FP = {self.fp} | **TN = {self.tn}** | `{self.fp + self.tn}` |",
            f"| **预测总计** | `{self.tp + self.fp}` | `{self.fn + self.tn}` | **`{self.total_samples}`** |",
            "",
            "## 3. 可观测性与 Token 消耗监控",
            "",
            f"- **评估总耗时**: `{self.total_latency_seconds:.2f} s` (平均单篇: `{self.average_latency_seconds:.3f} s`)",
            f"- **Prompt Tokens**: `{self.token_usage.get('prompt_tokens', 0)}`",
            f"- **Completion Tokens**: `{self.token_usage.get('completion_tokens', 0)}`",
            f"- **总计消耗 Tokens**: `{self.token_usage.get('total_tokens', 0)}`",
            f"- **LLM 调用次数**: `{self.token_usage.get('llm_call_count', 0)}`",
            "",
            "## 4. 样本逐篇判定明细表",
            "",
            "| arXiv ID | 论文标题 | 实际标签 | 预测标签 | 判定结果 | 置信度 | 范式 (GT / Pred) |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
        ]

        for d in self.details:
            correct_icon = "✅ 正确" if d.is_correct else "❌ 错误"
            gt_str = "WM" if d.ground_truth_is_wm else "Non-WM"
            pred_str = "WM" if d.predicted_is_wm else "Non-WM"
            title_trunc = d.title[:35] + "..." if len(d.title) > 35 else d.title
            paradigm_comp = f"{d.gt_paradigm} / {d.pred_paradigm or 'N/A'}"
            lines.append(
                f"| [`{d.arxiv_id}`](https://arxiv.org/abs/{d.arxiv_id}) | "
                f"{title_trunc} | `{gt_str}` | `{pred_str}` | {correct_icon} | "
                f"`{d.confidence}` | {paradigm_comp} |"
            )

        # 错误样例剖析
        errors = [d for d in self.details if not d.is_correct]
        if errors:
            lines.extend(
                [
                    "",
                    "## 5. 判错样本根因剖析",
                    "",
                ]
            )
            for err in errors:
                lines.append(
                    f"- **[{err.arxiv_id}] {err.title}**  \n"
                    f"  - 真实: `{err.ground_truth_is_wm}`, 预测: `{err.predicted_is_wm}`  \n"
                    f"  - 模型研判理由: {err.reasoning}"
                )

        return "\n".join(lines)


class DatasetEvaluator:
    """自动化 Golden Dataset 评测执行器。"""

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
        """运行批量评估并返回详细评测结果。

        Args:
            samples: Golden Dataset 标注样本列表。
            section_fetcher: 可选的正文章节拉取函数，模拟 ToolCallNode 触发逻辑。
        """
        logger.info("启动 Golden Dataset 自动化评估，共 %d 篇样本", len(samples))
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

            # 1. 摘要初筛
            decision = self.classifier.classify(paper=paper)

            # 2. 若置信度不足且提供了正文拉取函数，模拟 ToolCall 重新研判
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
                        "获取正文失败，维持原初筛判定: %s (%s)",
                        sample.arxiv_id,
                        exc,
                    )

            pred_is_wm = decision.is_world_model
            sample_latency = time.perf_counter() - sample_t0

            # 3. 统计混淆矩阵
            if sample.is_world_model and pred_is_wm:
                tp += 1
            elif not sample.is_world_model and pred_is_wm:
                fp += 1
            elif not sample.is_world_model and not pred_is_wm:
                tn += 1
            else:
                fn += 1

            is_correct = pred_is_wm == sample.is_world_model

            # 4. 对确认相关的世界模型样本评估结构化洞察提取
            pred_insight: PaperInsight | None = None
            if pred_is_wm:
                try:
                    pred_insight = self.classifier.extract_insight(
                        paper=paper, section_text=section_text
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("提取 insight 异常: %s (%s)", sample.arxiv_id, exc)

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

        # 计算比率
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
