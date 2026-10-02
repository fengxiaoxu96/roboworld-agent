"""Golden Dataset 数据加载与领域样本模型定义 (Task 2.2 / Task 5.3).

职责：
1. 规范 Golden Dataset 测试集样本格式；
2. 提供本地 JSON 黄金数据集的加载、校验与转换；
3. 支持将标注样本无缝转换为 `RawPaper` 与 `PaperInsight`。
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
    """黄金标注测试集单篇样本模型。"""

    model_config = ConfigDict(extra="ignore")

    arxiv_id: str = Field(..., description="arXiv 论文唯一编号")
    title: str = Field(..., description="论文标题")
    authors: list[str] = Field(default_factory=list, description="作者列表")
    abstract: str = Field(..., description="论文摘要")
    categories: list[str] = Field(default_factory=list, description="arXiv 分类")
    published_date: str = Field(..., description="发布时间 ISO 字符串")
    pdf_url: str = Field(..., description="PDF 下载链接")

    # 黄金人工真值 (Ground Truth)
    is_world_model: bool = Field(..., description="是否属于具身机器人世界模型")
    model_paradigm: ModelParadigm = Field(default="Other", description="模型范式类别")
    action_conditioned: bool = Field(
        default=False, description="是否显式输入机器人动作"
    )
    eval_environments: list[str] = Field(
        default_factory=list, description="评测基准列表"
    )
    core_innovation: str = Field(default="", description="核心创新点真值")
    label_notes: str = Field(default="", description="标注理由与混淆点剖析")

    def to_raw_paper(self) -> RawPaper:
        """转换为系统标准的 RawPaper 对象。"""
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
        """转换为标准的 PaperInsight 结构化洞察模型。"""
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
    """从本地 JSON 文件加载 Golden Dataset 测试集。

    Args:
        path: 数据集路径。默认为 data/golden_dataset.json。

    Returns:
        list[GoldenPaperSample]: 加载并通过 Pydantic 校验的样本列表。
    """
    target_path = Path(path or DEFAULT_GOLDEN_DATASET_PATH).resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"Golden Dataset 文件不存在: {target_path}")

    with target_path.open("r", encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)

    samples = [GoldenPaperSample.model_validate(item) for item in data]
    logger.info("成功加载 Golden Dataset: %s (共 %d 篇样本)", target_path, len(samples))
    return samples


def validate_golden_dataset(
    samples: list[GoldenPaperSample],
) -> tuple[bool, str]:
    """对 Golden Dataset 样本分布进行完整性校验。

    要求：
    1. 样本总数 >= 20 篇；
    2. 正样本（世界模型）不少于 10 篇；
    3. 负样本（边缘混淆）不少于 10 篇；
    4. 所有必填字段非空。
    """
    if len(samples) < 20:
        return False, f"样本总数不足 20 篇 (当前: {len(samples)} 篇)"

    wm_count = sum(1 for s in samples if s.is_world_model)
    non_wm_count = sum(1 for s in samples if not s.is_world_model)

    if wm_count < 10:
        return False, f"标准世界模型正样本不足 10 篇 (当前: {wm_count} 篇)"

    if non_wm_count < 10:
        return False, f"边缘混淆负样本不足 10 篇 (当前: {non_wm_count} 篇)"

    for idx, s in enumerate(samples):
        if not s.arxiv_id or not s.title or not s.abstract:
            return False, f"第 {idx} 篇样本缺少核心字段 (arxiv_id/title/abstract)"

    return (
        True,
        f"校验通过: 共 {len(samples)} 篇 (正样本: {wm_count}, 负样本: {non_wm_count})",
    )


__all__ = [
    "DEFAULT_GOLDEN_DATASET_PATH",
    "GoldenPaperSample",
    "load_golden_dataset",
    "validate_golden_dataset",
]
