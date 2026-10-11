"""Markdown 每日简报生成与 Webhook 分发工具 (Task 5.1).

主要职责：
1. 使用 Jinja2 模板渲染学术早报 Markdown：
   - 自动按模型范式（Flow-Matching, Diffusion, Transformer, Mamba 等）层级归类；
   - 视觉高亮 Action-conditioned 动作闭环属性；
   - 突出评测基准环境（CALVIN, Franka, Isaac Gym 等）与 150 字核心创新概述；
2. 本地持久化：保存至指定目录（如 data/reports/roboworld_daily_YYYY-MM-DD.md）；
3. 多渠道 Webhook 推送：支持飞书、钉钉、企业微信、Slack 及通用 Webhook 协议分发。
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jinja2
from dotenv import load_dotenv

from src.core.schema import ModelParadigm, PaperInsight, RawPaper

load_dotenv()

logger = logging.getLogger(__name__)

# 默认简报本地存储路径
DEFAULT_REPORTS_DIR = Path(__file__).resolve().parents[2] / "data" / "reports"

# 默认模板定义
MARKDOWN_REPORT_TEMPLATE = """# 🤖 RoboWorld 机器人世界模型每日处理简报

> 专注具身物理交互与动作闭环预测 · 每日文献追踪简报  
> **生成时间**: `{{ generated_at }}`

---

## 📊 今日收录与处理概览

| 统计指标 | 数值 | 说明 |
| :--- | :--- | :--- |
| 🔍 **arXiv 扫描检索** | **{{ stats.scanned_count if stats.scanned_count is defined else stats.total_fetched }}** 篇 | 对应检索分类与时间窗内累计扫描总数 |
| 📥 **初筛保留文献** | **{{ stats.total_fetched }}** 篇 | 命中规则初筛关键词集合，进入研判流水线 |
| 🌟 **入库世界模型** | **{{ stats.world_models_stored }}** 篇 | 严格满足物理转移与动作条件 |
| ⏭️ **跳过无关文献** | **{{ stats.skipped_unrelated }}** 篇 | 包含通用无动作视频生成/非具身模型 |
| 🔍 **调阅正文深入研判** | **{{ stats.tool_calls_made }}** 次 | 摘要信息不足时自动触发 PDF 解析 |
| ⚠️ **异常隔离样本** | **{{ stats.errors }}** 篇 | 单样本容错，不中断整体流水线 |

{% if stats.filter_keywords %}
> 💡 **初筛规则说明**：本次初筛采用关键词规则过滤（不区分大小写检索标题与摘要），初筛所用关键词包括：{{ stats.clean_keywords if stats.clean_keywords else (stats.filter_keywords | join(', ')) }}。
{% endif %}

---

## 🌟 世界模型文献精选（按模型范式归类）

{% if paradigms_data %}
{% for paradigm, papers in paradigms_data.items() %}
### {{ paradigm_icons.get(paradigm, '📦') }} {{ paradigm }} 模型范式 (共 {{ papers|length }} 篇)

{% for item in papers %}
#### [{{ loop.index }}] [{{ item.title }}]({{ item.pdf_url }})

- **arXiv ID**: `{{ item.arxiv_id }}` | **发布日期**: `{{ item.published_date }}`
- **作者列表**: {{ item.authors }}
- **动作闭环 (Action-Conditioned)**: {% if item.action_conditioned %}🟢 **【是】显式支持动作转移预测**{% else %}⚪ 否 (纯无条件/离散轨迹预测){% endif %}
- **验证环境与基准**: 
{% for env in item.eval_environments %}
  `{{ env }}`{% endfor %}
- **💡 核心创新概述**:
  > {{ item.core_innovation }}

[📥 下载论文 PDF]({{ item.pdf_url }}) · [🔗 arXiv 论文页面](https://arxiv.org/abs/{{ item.arxiv_id }})

---
{% endfor %}
{% endfor %}
{% else %}
> ⚠️ **今日未发现符合标准的具身机器人世界模型文献。**
{% endif %}

{% if skipped_samples %}
## ⏭️ 部分跳过文献研判依据示例（边界把控）

<details>
<summary>展开查看跳过的边缘混淆或非具身文献（共 {{ skipped_samples|length }} 条样本）</summary>

| arXiv ID | 论文标题 | 判定跳过原因 |
| :--- | :--- | :--- |
{% for skip in skipped_samples %}
| `{{ skip.arxiv_id }}` | {{ skip.title }} | {{ skip.reasoning }} |
{% endfor %}

</details>
{% endif %}

---
*本早报由 RoboWorld Agent 全自动化生成与落库。*
"""

# 模型范式展示图标
PARADIGM_ICONS: dict[ModelParadigm | str, str] = {
    "Flow-Matching": "🌊",
    "Diffusion-based": "🌫️",
    "Transformer/AutoRegressive": "🔄",
    "State-Space/Mamba": "⚡",
    "Other": "🧩",
}


class FormattedPaperItem:
    """供模板渲染使用的论文条目规范数据结构。"""

    def __init__(
        self,
        arxiv_id: str,
        title: str,
        authors: str | list[str],
        published_date: str,
        pdf_url: str,
        model_paradigm: str,
        action_conditioned: bool,
        eval_environments: list[str],
        core_innovation: str,
    ) -> None:
        self.arxiv_id = arxiv_id
        self.title = title
        if isinstance(authors, list):
            self.authors = ", ".join(authors[:3]) + (" 等" if len(authors) > 3 else "")
        else:
            self.authors = str(authors)
        self.published_date = published_date
        self.pdf_url = pdf_url
        self.model_paradigm = model_paradigm
        self.action_conditioned = action_conditioned
        self.eval_environments = eval_environments
        self.core_innovation = core_innovation


class MarkdownReporter:
    """Markdown 学术早报渲染器与分发器。"""

    def __init__(
        self,
        template_string: str = MARKDOWN_REPORT_TEMPLATE,
        default_output_dir: Path | str | None = None,
    ) -> None:
        self.env = jinja2.Environment(
            loader=jinja2.BaseLoader(),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.template = self.env.from_string(template_string)
        self.default_output_dir = Path(
            default_output_dir or os.getenv("REPORTS_DIR") or DEFAULT_REPORTS_DIR
        )

    def render(
        self,
        papers: Sequence[Any],
        stats: dict[str, Any] | None = None,
        skipped_records: Sequence[Any] | None = None,
        generated_at: datetime | None = None,
    ) -> str:
        """使用 Jinja2 渲染生成完整的 Markdown 简报。

        Args:
            papers: 确认入库的世界模型论文列表（可以是 PaperState、元组或 dict）。
            stats: 流水线运行统计指标。
            skipped_records: 被跳过或边缘混淆论文列表（可选）。
            generated_at: 简报生成时间。
        """
        now_dt = generated_at or datetime.now(UTC)
        time_str = now_dt.strftime("%Y-%m-%d %H:%M:%S UTC")

        # 整理统计数据
        stat_dict = {
            "total_fetched": 0,
            "world_models_stored": 0,
            "skipped_unrelated": 0,
            "tool_calls_made": 0,
            "errors": 0,
        }
        if stats:
            stat_dict.update(stats)

        # 将正则形式的关键词整理为清晰、可读的关键词语句列表
        if stat_dict.get("filter_keywords"):
            raw_kws = stat_dict["filter_keywords"]
            clean_list: list[str] = []
            for kw in raw_kws:
                cleaned = str(kw).replace(r"[\s-]?", " ").replace("s?", "").strip()
                clean_list.append(cleaned)
            stat_dict["clean_keywords"] = ", ".join(clean_list)

        # 按模型范式归类整理论文条目
        paradigms_data: dict[str, list[FormattedPaperItem]] = {}
        for p in papers:
            item = self._extract_paper_item(p)
            if item is None:
                continue
            paradigm = item.model_paradigm
            paradigms_data.setdefault(paradigm, []).append(item)

        # 整理跳过样本说明
        skipped_data: list[dict[str, str]] = []
        if skipped_records:
            for s in skipped_records[:10]:  # 最多展示 10 条边缘混淆样本
                skip_item = self._extract_skipped_item(s)
                if skip_item:
                    skipped_data.append(skip_item)

        # 模板渲染
        return self.template.render(
            generated_at=time_str,
            stats=stat_dict,
            paradigms_data=paradigms_data,
            paradigm_icons=PARADIGM_ICONS,
            skipped_samples=skipped_data,
        )

    def save_report(
        self,
        markdown_content: str,
        output_dir: Path | str | None = None,
        filename: str | None = None,
    ) -> Path:
        """将 Markdown 简报保存到本地磁盘目录。

        Returns:
            Path: 生成的 Markdown 报告绝对路径。
        """
        target_dir = Path(output_dir or self.default_output_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        if not filename:
            date_tag = datetime.now(UTC).strftime("%Y-%m-%d")
            filename = f"roboworld_daily_{date_tag}.md"

        target_file = target_dir / filename
        target_file.write_text(markdown_content, encoding="utf-8")
        logger.info("日报 Markdown 文件已保存: %s", target_file)
        return target_file

    def send_webhook(
        self,
        markdown_content: str,
        webhook_url: str | None = None,
        title: str = "RoboWorld 机器人世界模型每日速递",
        timeout: float = 10.0,
    ) -> bool:
        """将简报通过 Webhook 推送至通信工具（飞书、钉钉、企微、Slack 等）。

        [注意] 当前已暂时停用外部软件推送服务，直接返回 False。
        """
        del (
            markdown_content,
            webhook_url,
            title,
            timeout,
        )  # 保持接口兼容同时避免未使用警告
        logger.info("软件推送服务当前已暂停使用，跳过 Webhook 消息推送。")
        return False

    def generate_and_save(
        self,
        papers: Sequence[Any],
        stats: dict[str, Any] | None = None,
        skipped_records: Sequence[Any] | None = None,
        output_dir: Path | str | None = None,
        filename: str | None = None,
        webhook_url: str | None = None,
    ) -> tuple[Path, str, bool]:
        """一站式生成简报、本地保存并执行 Webhook 推送。

        Returns:
            tuple[Path, str, bool]: (报告保存路径, Markdown文本内容, Webhook是否发送成功)
        """
        md_text = self.render(
            papers=papers,
            stats=stats,
            skipped_records=skipped_records,
        )
        saved_path = self.save_report(
            markdown_content=md_text,
            output_dir=output_dir,
            filename=filename,
        )
        webhook_ok = self.send_webhook(
            markdown_content=md_text,
            webhook_url=webhook_url,
        )
        return saved_path, md_text, webhook_ok

    @staticmethod
    def _extract_paper_item(item: Any) -> FormattedPaperItem | None:
        """从不同对象结构中提炼标准化渲染字段。"""
        # 1. PaperState (来自 LangGraph 状态)
        if hasattr(item, "paper") and hasattr(item, "insight"):
            paper: RawPaper = item.paper
            ins: PaperInsight | None = item.insight
            if ins is None or not ins.is_world_model:
                return None
            return FormattedPaperItem(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                authors=paper.authors,
                published_date=paper.published_date.strftime("%Y-%m-%d"),
                pdf_url=paper.pdf_url,
                model_paradigm=ins.model_paradigm,
                action_conditioned=ins.action_conditioned,
                eval_environments=ins.eval_environments,
                core_innovation=ins.core_innovation,
            )

        # 2. 元组 (RawPaper, PaperInsight)
        if isinstance(item, tuple) and len(item) == 2:
            p, ins = item
            if isinstance(p, RawPaper) and isinstance(ins, PaperInsight):
                if not ins.is_world_model:
                    return None
                return FormattedPaperItem(
                    arxiv_id=p.arxiv_id,
                    title=p.title,
                    authors=p.authors,
                    published_date=p.published_date.strftime("%Y-%m-%d"),
                    pdf_url=p.pdf_url,
                    model_paradigm=ins.model_paradigm,
                    action_conditioned=ins.action_conditioned,
                    eval_environments=ins.eval_environments,
                    core_innovation=ins.core_innovation,
                )

        # 3. 字典类型
        if isinstance(item, dict):
            if not item.get("is_world_model", True):
                return None
            return FormattedPaperItem(
                arxiv_id=str(item.get("arxiv_id", "")),
                title=str(item.get("title", "")),
                authors=item.get("authors", []),
                published_date=str(item.get("published_date", ""))[:10],
                pdf_url=str(item.get("pdf_url", "")),
                model_paradigm=str(item.get("model_paradigm", "Other")),
                action_conditioned=bool(item.get("action_conditioned", False)),
                eval_environments=list(item.get("eval_environments", [])),
                core_innovation=str(item.get("core_innovation", "")),
            )

        return None

    @staticmethod
    def _extract_skipped_item(item: Any) -> dict[str, str] | None:
        if hasattr(item, "paper"):
            paper = item.paper
            return {
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "reasoning": getattr(item, "reasoning", "未满足世界模型标准")
                or "非具身物理动态模型",
            }
        if isinstance(item, dict):
            return {
                "arxiv_id": str(item.get("arxiv_id", "")),
                "title": str(item.get("title", "")),
                "reasoning": str(item.get("reasoning", "未满足世界模型标准")),
            }
        return None


__all__ = [
    "DEFAULT_REPORTS_DIR",
    "MARKDOWN_REPORT_TEMPLATE",
    "PARADIGM_ICONS",
    "FormattedPaperItem",
    "MarkdownReporter",
]
