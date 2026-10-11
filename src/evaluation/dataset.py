"""Golden Dataset loading and domain sample model definitions (Task 2.2 / Task 5.3).

Responsibilities:
1. Standardize Golden Dataset evaluation sample format;
2. Provide loading, validation, and conversion for local JSON golden datasets;
3. Support seamless conversion of annotated samples to `RawPaper` and `PaperInsight`.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.schema import ModelParadigm, PaperInsight, RawPaper

logger = logging.getLogger(__name__)

DEFAULT_GOLDEN_DATASET_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "golden_dataset.json"
)


class GoldenPaperSample(BaseModel):
    """Single sample model in the Golden Dataset."""

    model_config = ConfigDict(extra="ignore")

    arxiv_id: str = Field(..., description="Unique arXiv paper identifier")
    title: str = Field(..., description="Paper title")
    authors: list[str] = Field(default_factory=list, description="Author list")
    abstract: str = Field(..., description="Paper abstract")
    categories: list[str] = Field(default_factory=list, description="arXiv categories")
    published_date: str = Field(..., description="Publication ISO date string")
    pdf_url: str = Field(..., description="PDF download URL")

    # Ground Truth
    is_world_model: bool = Field(
        ..., description="Whether it belongs to embodied robotics world models"
    )
    model_paradigm: ModelParadigm = Field(
        default="Other", description="Model paradigm category"
    )
    action_conditioned: bool = Field(
        default=False, description="Whether robot actions are explicitly conditioned"
    )
    eval_environments: list[str] = Field(
        default_factory=list, description="Evaluation benchmarks list"
    )
    core_innovation: str = Field(default="", description="Ground truth core innovation")
    label_notes: str = Field(
        default="", description="Annotation reasoning and confusion analysis"
    )

    def to_raw_paper(self) -> RawPaper:
        """Convert to standard RawPaper object."""
        try:
            pub_dt = datetime.fromisoformat(self.published_date)
        except Exception:  # noqa: BLE001
            pub_dt = datetime.now(UTC)

        return RawPaper(
            arxiv_id=self.arxiv_id,
            title=self.title,
            authors=self.authors or ["Unknown"],
            abstract=self.abstract,
            categories=self.categories or ["cs.RO"],
            published_date=pub_dt,
            pdf_url=self.pdf_url,
        )

    def to_paper_insight(self) -> PaperInsight:
        """Convert to standard PaperInsight structured model."""
        return PaperInsight(
            is_world_model=self.is_world_model,
            model_paradigm=self.model_paradigm,
            action_conditioned=self.action_conditioned,
            eval_environments=self.eval_environments,
            core_innovation=self.core_innovation,
        )


def load_golden_dataset(
    path: Path | str | None = None,
) -> list[GoldenPaperSample]:
    """Load Golden Dataset from local JSON file.

    Args:
        path: Dataset path. Defaults to data/golden_dataset.json.

    Returns:
        list[GoldenPaperSample]: List of validated samples.
    """
    target_path = Path(path or DEFAULT_GOLDEN_DATASET_PATH).resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"Golden Dataset file not found: {target_path}")

    with target_path.open("r", encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)

    samples = [GoldenPaperSample.model_validate(item) for item in data]
    logger.info(
        "Successfully loaded Golden Dataset: %s (total %d samples)",
        target_path,
        len(samples),
    )
    return samples


def validate_golden_dataset(
    samples: list[GoldenPaperSample],
) -> tuple[bool, str]:
    """Validate sample distribution integrity of Golden Dataset.

    Requirements:
    1. Total samples >= 20;
    2. Positive samples (world model) >= 10;
    3. Negative samples (confounding cases) >= 10;
    4. All required fields are non-empty.
    """
    if len(samples) < 20:
        return False, f"Total sample count is less than 20 (current: {len(samples)})"

    wm_count = sum(1 for s in samples if s.is_world_model)
    non_wm_count = sum(1 for s in samples if not s.is_world_model)

    if wm_count < 10:
        return (
            False,
            f"Positive samples (is_world_model=True) are less than 10 (current: {wm_count})",
        )

    if non_wm_count < 10:
        return (
            False,
            f"Negative samples (is_world_model=False) are less than 10 (current: {non_wm_count})",
        )

    for idx, s in enumerate(samples):
        if not s.arxiv_id or not s.title or not s.abstract:
            return (
                False,
                f"Sample at index {idx} is missing core fields (arxiv_id/title/abstract)",
            )

    return (
        True,
        f"Validation passed: {len(samples)} total samples (positive: {wm_count}, negative: {non_wm_count})",
    )


__all__ = [
    "DEFAULT_GOLDEN_DATASET_PATH",
    "GoldenPaperSample",
    "load_golden_dataset",
    "validate_golden_dataset",
]
