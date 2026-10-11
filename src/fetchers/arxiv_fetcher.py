"""arXiv 采集器与关键词规则初筛。

这一层只做两件事：按分类把论文元数据拉下来，再按天数 / 篇数截断，
并用标题/摘要关键词丢掉明显无关的文献。
是否真的是机器人世界模型，留给后续的分类 Agent 判断。这里故意保持高召回。

使用 ``src.core.schema.RawPaper`` 作为输出数据模型。
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol, TypeVar

import arxiv
import requests

from src.core.schema import RawPaper

logger = logging.getLogger(__name__)

T = TypeVar("T")

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
    r"action[\s-]?conditioned",
    r"dynamics[\s-]?models?",
    r"latent[\s-]?world",
    r"visual[\s-]?foresight",
)


@dataclass(frozen=True)
class FetchStats:
    """arXiv 采集与规则初筛统计指标。

    Attributes:
        scanned_count: 遍历/扫描的 arXiv 论文总数。
        window_retained_count: 落在指定时间窗口（或 max_results 截断前）保留的篇数。
        rule_retained_count: 命中关键词初筛规则后保留的篇数。
        filter_keywords: 初筛所用的关键词规则集合。
    """

    scanned_count: int
    window_retained_count: int
    rule_retained_count: int
    filter_keywords: tuple[str, ...] = KEYWORD_PATTERNS


# arXiv 官方客户端在 num_retries>0 时会立刻重试，没有退避。
# 本模块把客户端内部重试关掉，由 _collect_with_backoff 统一做指数退避。
_CLIENT_NUM_RETRIES = 0
# arXiv API 使用条款要求相邻请求至少间隔 3 秒。
_CLIENT_DELAY_SECONDS = 3.0
_PAGE_SIZE = 100
_MAX_BACKOFF_SECONDS = 60.0
# 429 限流时从 60 秒起跳，封顶 5 分钟，避免在冷却期内连续撞墙。
_RATE_LIMIT_BASE_DELAY = 60.0
_RATE_LIMIT_MAX_BACKOFF_SECONDS = 300.0

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


def _as_utc(value: datetime) -> datetime:
    """把无时区的时间当成 UTC，有时区的统一转到 UTC，便于和窗口比较。"""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _window_bounds(
    days_back: int, now: datetime | None = None
) -> tuple[datetime, datetime]:
    """从 ``now``（默认当前 UTC）往回推 ``days_back`` 天，得到闭区间时间窗。"""
    _require_positive_int(days_back, "days_back")
    end = _as_utc(now or datetime.now(UTC))
    return end - timedelta(days=days_back), end


def _updated_in_window(result: arxiv.Result, start: datetime, end: datetime) -> bool:
    """用论文最近更新时间判断是否落在窗口内，覆盖修订稿而不只看首次提交。"""
    return start <= _as_utc(result.updated) <= end


def _http_status(exc: BaseException) -> int | None:
    """从 arXiv / requests 异常里取出 HTTP 状态码。"""
    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status
    return None


def _is_rate_limited(exc: BaseException) -> bool:
    return _http_status(exc) == 429 or "HTTP 429" in str(exc)


def _backoff_delay(exc: BaseException, attempt: int, base_delay: float) -> float:
    """普通失败用短退避；429 从 60 秒起按指数拉长，避免刚被限流又立刻重打。"""
    if _is_rate_limited(exc):
        return min(
            _RATE_LIMIT_BASE_DELAY * (2**attempt),
            _RATE_LIMIT_MAX_BACKOFF_SECONDS,
        )
    return min(base_delay * (2**attempt), _MAX_BACKOFF_SECONDS)


def _require_at_least_one_limit(days_back: int | None, max_results: int | None) -> None:
    if days_back is None and max_results is None:
        raise ValueError("days_back 和 max_results 至少需要提供一个")
    if days_back is not None:
        _require_positive_int(days_back, "days_back")
    if max_results is not None:
        _require_positive_int(max_results, "max_results")


def resolve_fetch_limits(
    days_back: int | None,
    max_results: int | None,
    *,
    default_days_back: int | None = 1,
) -> tuple[int | None, int | None]:
    """解析抓取截断条件。

    两者都没给时，用 ``default_days_back``（调度器默认按 1 天拉全量）。
    只给其中一个就只按那个截；两个都给则同时生效。
    """
    if days_back is None and max_results is None:
        if default_days_back is None:
            raise ValueError("days_back 和 max_results 至少需要提供一个")
        _require_positive_int(default_days_back, "days_back")
        return default_days_back, None
    _require_at_least_one_limit(days_back, max_results)
    return days_back, max_results


def build_search_query(
    days_back: int | None = None,
    *,
    categories: Sequence[str] = DEFAULT_CATEGORIES,
    now: datetime | None = None,
) -> str:
    """拼出 arXiv API 的 ``search_query``：只按分类检索。

    官方 ``search_query`` 不支持 ``lastUpdatedDate`` 过滤字段，写进去会得到
    空结果。日期窗口由 ``fetch_papers`` 在本地按 ``result.updated`` 截取。
    ``days_back`` 若传入则做正整数校验；``now`` 保留是为了兼容既有调用签名。
    """
    if days_back is not None:
        _require_positive_int(days_back, "days_back")
    if not categories:
        raise ValueError("categories 不能为空")
    del now

    category_clause = " OR ".join(f"cat:{category}" for category in categories)
    return f"({category_clause})"


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


def _take_results_with_stats(
    results: Iterator[arxiv.Result],
    *,
    window: tuple[datetime, datetime] | None,
    max_results: int | None,
) -> tuple[list[arxiv.Result], int]:
    """按更新时间降序消费结果，再按天数 / 篇数截断，并记录实际扫描篇数。

    有时间窗时：丢掉窗口之后的条目；一旦遇到窗口之前的条目就停止翻页
    （依赖 ``sortBy=lastUpdatedDate`` 降序）。有篇数上限时，凑够就停。
    只按天拉时不设 API ``max_results``，避免先被篇数截断。
    """
    selected: list[arxiv.Result] = []
    scanned = 0
    for result in results:
        scanned += 1
        if window is not None and not _updated_in_window(result, *window):
            if _as_utc(result.updated) < window[0]:
                break
            continue
        selected.append(result)
        if max_results is not None and len(selected) >= max_results:
            break
    logger.info("筛选完成：扫描 %d 篇，保留 %d 篇", scanned, len(selected))
    return selected, scanned


def _take_results(
    results: Iterator[arxiv.Result],
    *,
    window: tuple[datetime, datetime] | None,
    max_results: int | None,
) -> list[arxiv.Result]:
    """兼容旧接口的截断逻辑，只返回结果列表。"""
    selected, _ = _take_results_with_stats(
        results, window=window, max_results=max_results
    )
    return selected


def fetch_papers_with_stats(
    days_back: int | None = None,
    max_results: int | None = None,
    *,
    categories: Sequence[str] = DEFAULT_CATEGORIES,
    client: ArxivResultsClient | None = None,
    max_retries: int = 5,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    now: datetime | None = None,
) -> tuple[list[RawPaper], FetchStats]:
    """按分类拉取论文，进行截断与初筛，并返回清洗后的论文与统计指标 FetchStats。"""
    _require_at_least_one_limit(days_back, max_results)
    window = _window_bounds(days_back, now) if days_back is not None else None
    query = build_search_query(days_back, categories=categories, now=now)
    search_limit = max_results if window is None else None
    search = arxiv.Search(
        query=query,
        max_results=search_limit,
        sort_by=arxiv.SortCriterion.LastUpdatedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    page_size = _PAGE_SIZE
    if search_limit is not None:
        page_size = min(search_limit, _PAGE_SIZE)
    api = client or arxiv.Client(
        page_size=page_size,
        delay_seconds=_CLIENT_DELAY_SECONDS,
        num_retries=_CLIENT_NUM_RETRIES,
    )
    window_text = (
        f"{window[0].strftime('%Y%m%d%H%M')} TO {window[1].strftime('%Y%m%d%H%M')} UTC"
        if window
        else "不限"
    )
    limit_text = "不限" if max_results is None else str(max_results)
    logger.info(
        "开始抓取 arXiv：%s（days_back=%s, max_results=%s, window=%s）",
        query,
        days_back if days_back is not None else "不限",
        limit_text,
        window_text,
    )
    selected, scanned_count = _collect_with_backoff(
        api,
        search,
        consume=lambda results: _take_results_with_stats(
            results, window=window, max_results=max_results
        ),
        max_retries=max_retries,
        base_delay=base_delay,
        sleep=sleep,
    )
    window_retained_count = len(selected)
    papers = [result_to_raw_paper(result) for result in selected]
    filtered_papers = rule_filter(papers)
    stats = FetchStats(
        scanned_count=scanned_count,
        window_retained_count=window_retained_count,
        rule_retained_count=len(filtered_papers),
        filter_keywords=KEYWORD_PATTERNS,
    )
    return filtered_papers, stats


def fetch_papers(
    days_back: int | None = None,
    max_results: int | None = None,
    *,
    categories: Sequence[str] = DEFAULT_CATEGORIES,
    client: ArxivResultsClient | None = None,
    max_retries: int = 5,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    now: datetime | None = None,
) -> list[RawPaper]:
    """按分类拉取论文，再按天数和/或篇数截断，最后做规则初筛。

    三种合法组合：

    - 只给 ``days_back``：按最近更新时间翻页，直到走出时间窗，不先截篇数；
    - 只给 ``max_results``：取最近更新的 N 篇，不做日期过滤；
    - 两个都给：时间窗内再截最多 N 篇。

    查询只按分类检索，按最近更新时间降序。返回清洗并过滤后的
    ``list[RawPaper]``。
    """
    papers, _ = fetch_papers_with_stats(
        days_back=days_back,
        max_results=max_results,
        categories=categories,
        client=client,
        max_retries=max_retries,
        base_delay=base_delay,
        sleep=sleep,
        now=now,
    )
    return papers


def _collect_with_backoff(
    client: ArxivResultsClient,
    search: arxiv.Search,
    *,
    consume: Callable[[Iterator[arxiv.Result]], T] | None = None,
    max_retries: int,
    base_delay: float,
    sleep: Callable[[float], None],
) -> T:
    """把一次完整抓取包进指数退避。

    普通失败等待 ``min(base_delay * 2^n, 60)`` 秒。
    429 限流等待 ``min(60 * 2^n, 300)`` 秒。
    生成器在翻页中途失败也算整次失败，下次从头再拉，避免交出半截结果。
    重试耗尽后把最后一次异常原样抛出。
    """
    if max_retries < 0:
        raise ValueError("max_retries 必须大于等于 0")
    if base_delay < 0:
        raise ValueError("base_delay 必须大于等于 0")

    take = consume or (lambda results: list(results))
    attempt = 0
    while True:
        try:
            return take(client.results(search))
        except _RETRYABLE_EXCEPTIONS as exc:
            if attempt >= max_retries:
                logger.error("arXiv 抓取在 %d 次重试后仍失败：%s", max_retries, exc)
                raise
            delay = _backoff_delay(exc, attempt, base_delay)
            reason = "限流 429" if _is_rate_limited(exc) else str(exc)
            logger.warning(
                "arXiv 抓取失败（%s），第 %d/%d 次重试，等待 %.1f 秒",
                reason,
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
    parser.add_argument(
        "--days-back",
        type=int,
        default=None,
        help="按最近更新时间回溯几天；不与 --max-results 同用时拉满该窗口",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="最多保留多少篇；省略则不按篇数截断",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    days_back, max_results = resolve_fetch_limits(args.days_back, args.max_results)
    papers = fetch_papers(days_back=days_back, max_results=max_results)
    for paper in papers:
        print(f"{paper.arxiv_id}\t{paper.title}")
    print(f"保留 {len(papers)} 篇")
    return papers


if __name__ == "__main__":
    main()
