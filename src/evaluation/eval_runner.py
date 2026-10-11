"""Automated Eval testing command-line interface (Task 5.3).

Used for:
1. One-click execution of Golden Dataset automated evaluation locally or in CI/CD pipelines;
2. Monitoring the impact of prompt iterations or model changes on classification accuracy;
3. Outputting highlighted terminal metrics and formatted Markdown evaluation reports.
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
    """CLI entry point function."""
    parser = argparse.ArgumentParser(
        description="RoboWorld Agent - Golden Dataset Automated Evaluation Tool (Task 5.3)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_GOLDEN_DATASET_PATH),
        help="Golden Dataset JSON file path (default: data/golden_dataset.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Markdown evaluation report save path (e.g. reports/eval_report.md)",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.90,
        help="Acceptance minimum accuracy threshold (default: 0.90 i.e. 90%%)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full evaluation results in JSON format to stdout",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed predictions and comparisons for each sample",
    )

    args = parser.parse_args()

    # 1. Load and validate dataset
    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        sys.stderr.write(f"Error: Dataset file not found: {dataset_path}\n")
        return 1

    try:
        samples = load_golden_dataset(dataset_path)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"Error: Failed to load dataset: {exc}\n")
        return 1

    is_valid, validation_msg = validate_golden_dataset(samples)
    if not is_valid:
        sys.stderr.write(
            f"Warning: Dataset distribution validation failed: {validation_msg}\n"
        )
    else:
        logger.info("Dataset validation passed: %s", validation_msg)

    # 2. Run evaluation
    tracker = TokenUsageTracker()
    classifier = ClassifierAgent()
    evaluator = DatasetEvaluator(classifier=classifier, tracker=tracker)

    result = evaluator.evaluate(samples=samples)

    # 3. Output report and terminal summary
    md_report = result.to_markdown_report()

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md_report, encoding="utf-8")
        sys.stdout.write(f"✅ Evaluation report saved to: {out_path}\n")

    if args.json:
        sys.stdout.write(
            json.dumps(result.model_dump(), indent=2, ensure_ascii=False) + "\n"
        )
    else:
        # Terminal table print
        status = (
            "PASSED ✅" if result.passes_threshold(args.min_accuracy) else "FAILED ❌"
        )
        sys.stdout.write("\n" + "=" * 65 + "\n")
        sys.stdout.write(
            f"  RoboWorld Agent - Golden Dataset Automated Evaluation Results: {status}\n"
        )
        sys.stdout.write("=" * 65 + "\n")
        sys.stdout.write(f"  Total Samples:                 {result.total_samples}\n")
        sys.stdout.write(
            f"  Confusion Matrix:              TP={result.tp}, FP={result.fp}, TN={result.tn}, FN={result.fn}\n"
        )
        sys.stdout.write(
            f"  Accuracy:                      {result.accuracy * 100:.2f}% (Threshold: {args.min_accuracy * 100:.1f}%)\n"
        )
        sys.stdout.write(
            f"  Precision:                     {result.precision * 100:.2f}%\n"
        )
        sys.stdout.write(
            f"  Recall:                        {result.recall * 100:.2f}%\n"
        )
        sys.stdout.write(f"  F1-Score:                      {result.f1_score:.4f}\n")
        sys.stdout.write(
            f"  Paradigm Acc:                  {result.paradigm_accuracy * 100:.2f}%\n"
        )
        sys.stdout.write(
            f"  Action Acc:                    {result.action_conditioned_accuracy * 100:.2f}%\n"
        )
        sys.stdout.write(
            f"  Total Latency:                 {result.total_latency_seconds:.3f} s\n"
        )
        sys.stdout.write(
            f"  Total Tokens:                  {result.token_usage.get('total_tokens', 0)}\n"
        )
        sys.stdout.write("=" * 65 + "\n\n")

        if args.verbose:
            sys.stdout.write("Sample-by-sample evaluation details:\n")
            for d in result.details:
                mark = "✓" if d.is_correct else "✗"
                sys.stdout.write(
                    f"  [{mark}] {d.arxiv_id} | GT:{d.ground_truth_is_wm} Pred:{d.predicted_is_wm} | {d.title[:45]}\n"
                )
            sys.stdout.write("\n")

    return 0 if result.passes_threshold(args.min_accuracy) else 1


if __name__ == "__main__":
    sys.exit(main())
