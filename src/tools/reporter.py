"""Markdown daily briefing generation and webhook dispatch tool (Task 5.1).

Main responsibilities:
1. Render academic briefing Markdown using Jinja2 templates:
   - Hierarchically categorize papers by model paradigm (Flow-Matching, Diffusion, Transformer, Mamba, etc.);
   - Visually highlight Action-conditioned properties;
   - Highlight evaluation benchmarks (CALVIN, Franka, Isaac Gym, etc.) and core innovation summaries;
2. Local persistence: save to designated directory (e.g. data/reports/roboworld_daily_YYYY-MM-DD.md);
3. Multi-channel webhook dispatch: support Feishu, DingTalk, WeCom, Slack, and generic webhooks.
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

# Default briefing local storage path
DEFAULT_REPORTS_DIR = Path(__file__).resolve().parents[2] / "data" / "reports"

# Default template definition
MARKDOWN_REPORT_TEMPLATE = """# 🤖 RoboWorld Robotics World Models Daily Briefing

> Focusing on Embodied Physical Interaction & Action-Conditioned Prediction · Daily Research Tracking  
> **Generated At**: `{{ generated_at }}`

---

## 📊 Today's Processing Overview

| Metric | Value | Description |
| :--- | :--- | :--- |
| 🔍 **arXiv Scanned Papers** | **{{ stats.scanned_count if stats.scanned_count is defined else stats.total_fetched }}** papers | Total papers scanned within target categories and time window |
| 📥 **Pre-filtered Candidates** | **{{ stats.total_fetched }}** papers | Matched keyword rules and entered analysis pipeline |
| 🌟 **Stored World Models** | **{{ stats.world_models_stored }}** papers | Strictly meet physical transitions and action conditioning criteria |
| ⏭️ **Skipped Irrelevant Papers** | **{{ stats.skipped_unrelated }}** papers | Including unconditional video generators and non-embodied models |
| 🔍 **Deep Section Inspections** | **{{ stats.tool_calls_made }}** times | Triggered PDF parsing when abstract information was insufficient |
| ⚠️ **Isolated Failure Samples** | **{{ stats.errors }}** papers | Isolated with single-sample fault tolerance without breaking pipeline |

{% if stats.filter_keywords %}
> 💡 **Pre-filtering Rule Note**: Keyword rule pre-filtering was applied (case-insensitive title and abstract search). Rules included: {{ stats.clean_keywords if stats.clean_keywords else (stats.filter_keywords | join(', ')) }}.
{% endif %}

---

## 🌟 Featured World Model Papers (Grouped by Model Paradigm)

{% if paradigms_data %}
{% for paradigm, papers in paradigms_data.items() %}
### {{ paradigm_icons.get(paradigm, '📦') }} {{ paradigm }} Paradigm (Total {{ papers|length }} papers)

{% for item in papers %}
#### [{{ loop.index }}] [{{ item.title }}]({{ item.pdf_url }})

- **arXiv ID**: `{{ item.arxiv_id }}` | **Published**: `{{ item.published_date }}`
- **Authors**: {{ item.authors }}
- **Action-Conditioned**: {% if item.action_conditioned %}🟢 **[Yes] Explicitly supports action transition prediction**{% else %}⚪ No (Pure unconditional / discrete trajectory prediction){% endif %}
- **Evaluation Environments & Benchmarks**: 
{% for env in item.eval_environments %}
  `{{ env }}`{% endfor %}
- **💡 Core Innovation Summary**:
  > {{ item.core_innovation }}

[📥 Download PDF]({{ item.pdf_url }}) · [🔗 arXiv Page](https://arxiv.org/abs/{{ item.arxiv_id }})

---
{% endfor %}
{% endfor %}
{% else %}
> ⚠️ **No robotics world model papers meeting criteria were found today.**
{% endif %}

{% if skipped_samples %}
## ⏭️ Sample Reasons for Skipped Papers (Boundary Control)

<details>
<summary>Expand to view skipped borderline or non-embodied papers (Total {{ skipped_samples|length }} samples)</summary>

| arXiv ID | Title | Rejection Reason |
| :--- | :--- | :--- |
{% for skip in skipped_samples %}
| `{{ skip.arxiv_id }}` | {{ skip.title }} | {{ skip.reasoning }} |
{% endfor %}

</details>
{% endif %}

---
*This briefing was fully automatically generated and persisted by RoboWorld Agent.*
"""

# Model paradigm display icons
PARADIGM_ICONS: dict[ModelParadigm | str, str] = {
    "Flow-Matching": "🌊",
    "Diffusion-based": "🌫️",
    "Transformer/AutoRegressive": "🔄",
    "State-Space/Mamba": "⚡",
    "Other": "🧩",
}


class FormattedPaperItem:
    """Standardized paper item data structure for template rendering."""

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
            self.authors = ", ".join(authors[:3]) + (
                " et al." if len(authors) > 3 else ""
            )
        else:
            self.authors = str(authors)
        self.published_date = published_date
        self.pdf_url = pdf_url
        self.model_paradigm = model_paradigm
        self.action_conditioned = action_conditioned
        self.eval_environments = eval_environments
        self.core_innovation = core_innovation


class MarkdownReporter:
    """Markdown academic briefing renderer and dispatcher."""

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
        """Render complete Markdown daily briefing using Jinja2.

        Args:
            papers: List of confirmed world model papers (can be PaperState, tuple, or dict).
            stats: Pipeline execution metrics.
            skipped_records: List of skipped or borderline papers (optional).
            generated_at: Briefing generation timestamp.
        """
        now_dt = generated_at or datetime.now(UTC)
        time_str = now_dt.strftime("%Y-%m-%d %H:%M:%S UTC")

        # Organize statistics
        stat_dict = {
            "total_fetched": 0,
            "world_models_stored": 0,
            "skipped_unrelated": 0,
            "tool_calls_made": 0,
            "errors": 0,
        }
        if stats:
            stat_dict.update(stats)

        # Clean regex keyword rules into human-readable list
        if stat_dict.get("filter_keywords"):
            raw_kws = stat_dict["filter_keywords"]
            clean_list: list[str] = []
            for kw in raw_kws:
                cleaned = str(kw).replace(r"[\s-]?", " ").replace("s?", "").strip()
                clean_list.append(cleaned)
            stat_dict["clean_keywords"] = ", ".join(clean_list)

        # Group papers by model paradigm
        paradigms_data: dict[str, list[FormattedPaperItem]] = {}
        for p in papers:
            item = self._extract_paper_item(p)
            if item is None:
                continue
            paradigm = item.model_paradigm
            paradigms_data.setdefault(paradigm, []).append(item)

        # Organize skipped sample explanations
        skipped_data: list[dict[str, str]] = []
        if skipped_records:
            for s in skipped_records[:10]:  # At most 10 samples
                skip_item = self._extract_skipped_item(s)
                if skip_item:
                    skipped_data.append(skip_item)

        # Template rendering
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
        """Save Markdown briefing to local disk directory.

        Returns:
            Path: Absolute path of generated Markdown report.
        """
        target_dir = Path(output_dir or self.default_output_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        if not filename:
            date_tag = datetime.now(UTC).strftime("%Y-%m-%d")
            filename = f"roboworld_daily_{date_tag}.md"

        target_file = target_dir / filename
        target_file.write_text(markdown_content, encoding="utf-8")
        logger.info("Daily report Markdown file saved: %s", target_file)
        return target_file

    def send_webhook(
        self,
        markdown_content: str,
        webhook_url: str | None = None,
        title: str = "RoboWorld Robotics World Models Daily Dispatch",
        timeout: float = 10.0,
    ) -> bool:
        """Push briefing via webhook to chat tools (Feishu, DingTalk, WeCom, Slack, etc.).

        [Note] External software push service is currently suspended, returns False directly.
        """
        del (
            markdown_content,
            webhook_url,
            title,
            timeout,
        )  # Maintain interface compatibility while avoiding unused warnings
        logger.info(
            "Software push service is currently suspended; skipping webhook dispatch."
        )
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
        """One-stop method to render briefing, save locally, and dispatch webhook.

        Returns:
            tuple[Path, str, bool]: (Report save path, Markdown text content, Webhook dispatch success status)
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
        """Extract standardized rendering fields from different object structures."""
        # 1. PaperState (from LangGraph state)
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

        # 2. Tuple (RawPaper, PaperInsight)
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

        # 3. Dict type
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
                "reasoning": getattr(
                    item, "reasoning", "Did not meet world model criteria"
                )
                or "Non-embodied physical dynamics model",
            }
        if isinstance(item, dict):
            return {
                "arxiv_id": str(item.get("arxiv_id", "")),
                "title": str(item.get("title", "")),
                "reasoning": str(
                    item.get("reasoning", "Did not meet world model criteria")
                ),
            }
        return None


__all__ = [
    "DEFAULT_REPORTS_DIR",
    "MARKDOWN_REPORT_TEMPLATE",
    "PARADIGM_ICONS",
    "FormattedPaperItem",
    "MarkdownReporter",
]
