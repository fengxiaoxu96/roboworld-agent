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
    """arXiv 采集清洗后的原始论文元数据。"""

    model_config = ConfigDict(extra="ignore")

    arxiv_id: str = Field(
        ...,
        description="arXiv 论文编号（已去版本号后缀，例如 '2609.12345'）",
    )
    title: str = Field(..., description="论文标题")
    authors: list[str] = Field(..., description="作者列表")
    abstract: str = Field(..., description="论文摘要")
    categories: list[str] = Field(
        ...,
        description="arXiv 分类列表（例如 ['cs.RO', 'cs.AI', 'cs.LG']）",
    )
    published_date: datetime = Field(..., description="论文发布或更新时间")
    pdf_url: str = Field(..., description="PDF 下载链接")


class PaperInsight(BaseModel):
    """世界模型研判与结构化洞察。"""

    model_config = ConfigDict(extra="ignore")

    is_world_model: bool = Field(
        ...,
        description="是否具备环境动态预测/潜空间建模属性",
    )
    model_paradigm: ModelParadigm = Field(
        ...,
        description="模型范式类别",
    )
    action_conditioned: bool = Field(
        ...,
        description="是否显式输入 Action 进行条件转移预测",
    )
    eval_environments: list[str] = Field(
        ...,
        description="验证基准与评测环境（如 Isaac Gym/Lab, RLBench, CALVIN, Real Robot 等）",
    )
    core_innovation: str = Field(
        ...,
        description="150字以内核心贡献概述",
    )


__all__ = ["ModelParadigm", "PaperInsight", "RawPaper"]
