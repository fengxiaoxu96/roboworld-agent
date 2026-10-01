"""arXiv 采集器与关键词规则初筛。

这一层只做两件事：按分类和时间窗口把论文元数据拉下来，再用标题/摘要关键词丢掉明显无关的文献。
是否真的是机器人世界模型，留给后续的分类 Agent 判断。这里故意保持高召回。

使用 ``src.core.schema.RawPaper`` 作为输出数据模型。
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime, timedelta
from typing import Protocol

import arxiv
import requests

from src.core.schema import RawPaper

logger = logging.getLogger(__name__)

# 任务书指定的三个检索分类。cat: 前缀是 arXiv API 的分类查询语法。
DEFAULT_CATEGORIES: tuple[str, ...] = ("cs.RO", "cs.AI", "cs.LG")

# 规则初筛关键词。任务书给出的是示例（“如 … 等”），这里覆盖这些说法，
# 并补上同一概念的常见写法。词与词之间允许空格或连字符，大小写不敏感。
# 这一步只求少漏掉候选，精确边界由任务 2.2 的分类 Agent 负责。
KEYWORD_PATTERNS: tuple[str, ...] = (
    r"world[\s-]?models?",
    r"latent[\s-]?dynamics",
    r"video[\s-]?prediction",
    r"forward[\s-]?models?",
    r"embodied",
    r"action[\s-]?conditioned",
    r"dynamics[\s-]?models?",
    r"latent[\s-]?world",
    r"visual[\s-]?foresight",
)

# arXiv 官方客户端在 num_retries>0 时会立刻重试，没有退避。
# 本模块把客户端内部重试关掉，由 _collect_with_backoff 统一做指数退避。
_CLIENT_NUM_RETRIES = 0
# arXiv API 使用条款要求相邻请求至少间隔 3 秒。
_CLIENT_DELAY_SECONDS = 3.0
_MAX_BACKOFF_SECONDS = 60.0

# 这些异常表示“这次请求没拿到可用结果”，值得隔一段时间再试。
# 参数错误、数据校验错误不应重试。
_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    arxiv.ArxivError,
    requests.exceptions.RequestException,
    TimeoutError,
    ConnectionError,
)

_KEYWORD_RE = re.compile(
    "|".join(rf"\b(?:{pattern})\b" for pattern in KEYWORD_PATTERNS),
    re.IGNORECASE,
)
_VERSION_SUFFIX_RE = re.compile(r"v\d+$")
_WHITESPACE_RE = re.compile(r"\s+")


class ArxivResultsClient(Protocol):
    """采集器只依赖 ``results()``，测试可以传入不访问网络的替身。"""

    def results(
        self, search: arxiv.Search, offset: int = 0
    ) -> Iterator[arxiv.Result]: ...


def build_search_query(
    days_back: int,
    *,
    categories: Sequence[str] = DEFAULT_CATEGORIES,
    now: datetime | None = None,
) -> str:
    """拼出 arXiv API 的 ``search_query``。

    ``days_back`` 是滚动时间窗：从 ``now``（默认当前 UTC）往回推这么多天，
    用 ``lastUpdatedDate`` 覆盖“近几天更新”，而不是只看首次提交。
    三个分类用 OR 连接，再和日期窗口 AND。
    """
    _require_positive_int(days_back, "days_back")
    if not categories:
        raise ValueError("categories 不能为空")

    end = now or datetime.now(UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    else:
        end = end.astimezone(UTC)
    start = end - timedelta(days=days_back)

    category_clause = " OR ".join(f"cat:{category}" for category in categories)
    date_clause = (
        "lastUpdatedDate:["
        f"{start.strftime('%Y%m%d%H%M')} TO {end.strftime('%Y%m%d%H%M')}"
        "]"
    )
    return f"({category_clause}) AND {date_clause}"


def matches_world_model_keywords(title: str, abstract: str) -> bool:
    """标题或摘要里出现任一关键词则保留。两个字段分开匹配，避免跨字段误粘成一个词。"""
    return any(
        _KEYWORD_RE.search(_normalize_whitespace(text)) is not None
        for text in (title, abstract)
    )


def result_to_raw_paper(result: arxiv.Result) -> RawPaper:
    """把 arXiv 客户端的 ``Result`` 转换为 ``RawPaper``。

    清洗包括：去掉 id 的版本后缀、压缩标题和摘要里的换行、作者只保留姓名、
    缺少 PDF 链接时按 id 补一个。
    """
    arxiv_id = _VERSION_SUFFIX_RE.sub("", result.get_short_id())
    pdf_url = result.pdf_url or f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    return RawPaper(
        arxiv_id=arxiv_id,
        title=_normalize_whitespace(result.title),
        authors=[author.name for author in result.authors],
        abstract=_normalize_whitespace(result.summary),
        categories=list(result.categories),
        published_date=result.published,
        pdf_url=pdf_url,
    )


def rule_filter(papers: Sequence[RawPaper]) -> list[RawPaper]:
    """按关键词丢掉无关论文，并写一条丢弃比例日志。

    比例 = 丢弃数 / 进入初筛的篇数。一篇都没有时比例记为 0，避免除零。
    """
    kept = [
        paper
        for paper in papers
        if matches_world_model_keywords(paper.title, paper.abstract)
    ]
    total = len(papers)
    discarded = total - len(kept)
    ratio = (discarded / total) if total else 0.0
    logger.info(
        "规则初筛：共 %d 篇，保留 %d 篇，丢弃 %d 篇，丢弃比例 %.1f%%",
        total,
        len(kept),
        discarded,
        ratio * 100,
    )
    return kept


def fetch_papers(
    days_back: int = 1,
    max_results: int = 50,
    *,
    categories: Sequence[str] = DEFAULT_CATEGORIES,
    client: ArxivResultsClient | None = None,
    max_retries: int = 5,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    now: datetime | None = None,
) -> list[RawPaper]:
    """拉取近 ``days_back`` 天的论文，最多 ``max_results`` 篇，再做规则初筛。

    ``max_results`` 限制的是 API 返回条数，不是初筛之后的条数。
    排序按最近更新时间降序，截断时优先留下较新的更新。
    返回的是清洗并过滤后的 ``list[RawPaper]``。
    """
    _require_positive_int(max_results, "max_results")
    query = build_search_query(days_back, categories=categories, now=now)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.LastUpdatedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    api = client or arxiv.Client(
        page_size=min(max_results, 100),
        delay_seconds=_CLIENT_DELAY_SECONDS,
        num_retries=_CLIENT_NUM_RETRIES,
    )
    logger.info("开始抓取 arXiv：%s（max_results=%d）", query, max_results)
    results = _collect_with_backoff(
        api,
        search,
        max_retries=max_retries,
        base_delay=base_delay,
        sleep=sleep,
    )
    papers = [result_to_raw_paper(result) for result in results]
    return rule_filter(papers)


def _collect_with_backoff(
    client: ArxivResultsClient,
    search: arxiv.Search,
    *,
    max_retries: int,
    base_delay: float,
    sleep: Callable[[float], None],
) -> list[arxiv.Result]:
    """把一次完整抓取包进指数退避。

    第 n 次失败后等待 ``min(base_delay * 2^n, 60)`` 秒再重试。
    生成器在翻页中途失败也算整次失败，下次从头再拉，避免交出半截结果。
    重试耗尽后把最后一次异常原样抛出。
    """
    if max_retries < 0:
        raise ValueError("max_retries 必须大于等于 0")
    if base_delay < 0:
        raise ValueError("base_delay 必须大于等于 0")

    attempt = 0
    while True:
        try:
            return list(client.results(search))
        except _RETRYABLE_EXCEPTIONS as exc:
            if attempt >= max_retries:
                logger.error("arXiv 抓取在 %d 次重试后仍失败：%s", max_retries, exc)
                raise
            delay = min(base_delay * (2**attempt), _MAX_BACKOFF_SECONDS)
            logger.warning(
                "arXiv 抓取失败（%s），第 %d/%d 次重试，等待 %.1f 秒",
                exc,
                attempt + 1,
                max_retries,
                delay,
            )
            sleep(delay)
            attempt += 1


def _normalize_whitespace(text: str) -> str:
    """arXiv 标题和摘要经常带换行，压成单行空格，方便关键词匹配和下游阅读。"""
    return _WHITESPACE_RE.sub(" ", text).strip()


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} 必须是大于等于 1 的整数")


def main(argv: list[str] | None = None) -> list[RawPaper]:
    """命令行入口：抓取、打印丢弃比例日志，并把保留下来的 id 和标题打到标准输出。"""
    parser = argparse.ArgumentParser(
        description="抓取近期 arXiv 论文，并按世界模型关键词做规则初筛。"
    )
    parser.add_argument("--days-back", type=int, default=3, help="回溯最近几天的更新")
    parser.add_argument(
        "--max-results", type=int, default=50, help="API 最多返回多少篇"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    papers = fetch_papers(days_back=args.days_back, max_results=args.max_results)
    for paper in papers:
        print(f"{paper.arxiv_id}\t{paper.title}")
    print(f"保留 {len(papers)} 篇")
    return papers


if __name__ == "__main__":
    main()
