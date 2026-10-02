"""自动化 Eval 测试命令行入口 (Task 5.3).

用于：
1. 本地或 CI/CD 流水线中一键运行 Golden Dataset 自动化评估；
2. 监控 Prompt 迭代或模型切换对研判准确率的影响；
3. 输出高亮终端统计与格式化 Markdown 评测报告。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.agents.classifier_agent import ClassifierAgent
from src.evaluation.dataset import (
    DEFAULT_GOLDEN_DATASET_PATH,
    load_golden_dataset,
    validate_golden_dataset,
)
from src.evaluation.evaluator import DatasetEvaluator
from src.evaluation.tracing import TokenUsageTracker

logger = logging.getLogger(__name__)


def main() -> int:
    """CLI 入口函数。"""
    parser = argparse.ArgumentParser(
        description="RoboWorld Agent - Golden Dataset 自动化研判评测工具 (Task 5.3)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_GOLDEN_DATASET_PATH),
        help="Golden Dataset JSON 文件路径 (默认: data/golden_dataset.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Markdown 评估报告保存路径 (例如: reports/eval_report.md)",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.90,
        help="验收门禁最低准确率阈值 (默认: 0.90 即 90%%)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式向 stdout 输出完整评测结果",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印每篇样本的详细预测与比对结果",
    )

    args = parser.parse_args()

    # 1. 加载并校验数据集
    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        sys.stderr.write(f"错误: 找不到数据集文件: {dataset_path}\n")
        return 1

    try:
        samples = load_golden_dataset(dataset_path)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"错误: 加载数据集失败: {exc}\n")
        return 1

    is_valid, validation_msg = validate_golden_dataset(samples)
    if not is_valid:
        sys.stderr.write(f"警告: 数据集分布校验未通过: {validation_msg}\n")
    else:
        logger.info("数据集校验成功: %s", validation_msg)

    # 2. 运行评估
    tracker = TokenUsageTracker()
    classifier = ClassifierAgent()
    evaluator = DatasetEvaluator(classifier=classifier, tracker=tracker)

    result = evaluator.evaluate(samples=samples)

    # 3. 输出报告与终端摘要
    md_report = result.to_markdown_report()

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md_report, encoding="utf-8")
        sys.stdout.write(f"✅ 评测报告已保存至: {out_path}\n")

    if args.json:
        sys.stdout.write(
            json.dumps(result.model_dump(), indent=2, ensure_ascii=False) + "\n"
        )
    else:
        # 终端表格打印
        status = (
            "PASSED ✅" if result.passes_threshold(args.min_accuracy) else "FAILED ❌"
        )
        sys.stdout.write("\n" + "=" * 65 + "\n")
        sys.stdout.write(
            f"  RoboWorld Agent - Golden Dataset 自动化评估结果: {status}\n"
        )
        sys.stdout.write("=" * 65 + "\n")
        sys.stdout.write(f"  样本总数 (Total Samples):      {result.total_samples}\n")
        sys.stdout.write(
            f"  混淆矩阵 (Confusion Matrix):   TP={result.tp}, FP={result.fp}, TN={result.tn}, FN={result.fn}\n"
        )
        sys.stdout.write(
            f"  分类准确率 (Accuracy):         {result.accuracy * 100:.2f}% (阈值: {args.min_accuracy * 100:.1f}%)\n"
        )
        sys.stdout.write(
            f"  精确率 (Precision):            {result.precision * 100:.2f}%\n"
        )
        sys.stdout.write(
            f"  召回率 (Recall):               {result.recall * 100:.2f}%\n"
        )
        sys.stdout.write(f"  F1 分数 (F1-Score):            {result.f1_score:.4f}\n")
        sys.stdout.write(
            f"  范式识别率 (Paradigm Acc):     {result.paradigm_accuracy * 100:.2f}%\n"
        )
        sys.stdout.write(
            f"  动作条件识别率 (Action Acc):   {result.action_conditioned_accuracy * 100:.2f}%\n"
        )
        sys.stdout.write(
            f"  总耗时 (Total Latency):        {result.total_latency_seconds:.3f} s\n"
        )
        sys.stdout.write(
            f"  总 Token 消耗 (Total Tokens):  {result.token_usage.get('total_tokens', 0)}\n"
        )
        sys.stdout.write("=" * 65 + "\n\n")

        if args.verbose:
            sys.stdout.write("样本逐篇评估明细:\n")
            for d in result.details:
                mark = "✓" if d.is_correct else "✗"
                sys.stdout.write(
                    f"  [{mark}] {d.arxiv_id} | GT:{d.ground_truth_is_wm} Pred:{d.predicted_is_wm} | {d.title[:45]}\n"
                )
            sys.stdout.write("\n")

    return 0 if result.passes_threshold(args.min_accuracy) else 1


if __name__ == "__main__":
    sys.exit(main())
