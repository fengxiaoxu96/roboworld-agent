from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ModelParadigm = Literal[
    "Diffusion-based",
    "Transformer/AutoRegressive",
    "Flow-Matching",
    "State-Space/Mamba",
    "Other",
]


class RawPaper(BaseModel):
    """Raw paper metadata after arXiv collection and cleaning."""

    model_config = ConfigDict(extra="ignore")

    arxiv_id: str = Field(
        ...,
        description="arXiv paper ID (version suffix stripped, e.g., '2609.12345')",
    )
    title: str = Field(..., description="Paper title")
    authors: list[str] = Field(..., description="Author list")
    abstract: str = Field(..., description="Paper abstract")
    categories: list[str] = Field(
        ...,
        description="arXiv category list (e.g., ['cs.RO', 'cs.AI', 'cs.LG'])",
    )
    published_date: datetime = Field(..., description="Paper published or updated date")
    pdf_url: str = Field(..., description="PDF download link")


class PaperInsight(BaseModel):
    """World model analysis and structured insight."""

    model_config = ConfigDict(extra="ignore")

    is_world_model: bool = Field(
        ...,
        description="Whether the paper features environment dynamics prediction / latent space modeling",
    )
    model_paradigm: ModelParadigm = Field(
        ...,
        description="Model paradigm category",
    )
    action_conditioned: bool = Field(
        ...,
        description="Whether it explicitly takes action as input for conditional transition prediction",
    )
    eval_environments: list[str] = Field(
        ...,
        description="Evaluation benchmarks and environments (e.g., Isaac Gym/Lab, RLBench, CALVIN, Real Robot, etc.)",
    )
    core_innovation: str = Field(
        ...,
        description="Core innovation summary within 150 words",
    )


__all__ = ["ModelParadigm", "PaperInsight", "RawPaper"]
