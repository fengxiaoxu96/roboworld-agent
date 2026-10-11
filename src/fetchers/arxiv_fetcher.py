"""arXiv fetcher and keyword-based rule pre-filtering.

This layer performs two functions: pulls paper metadata by category,
truncates by days / count, and drops clearly unrelated papers using title/abstract keywords.
Whether a paper is truly a robotics world model is left to the downstream classification Agent.
High recall is intentionally maintained here.

Uses ``src.core.schema.RawPaper`` as the output data model.
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

# Retrieval categories specified in project requirements. cat: prefix is arXiv API query syntax.
DEFAULT_CATEGORIES: tuple[str, ...] = ("cs.RO", "cs.AI", "cs.LG")

# Rule pre-filtering keywords covering common variations.
# Words allow spaces or hyphens, case-insensitive.
# This step prioritizes high recall; precise boundaries are handled by the classification Agent.
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
    """Statistical metrics for arXiv fetching and rule pre-filtering.

    Attributes:
        scanned_count: Total number of arXiv papers traversed/scanned.
        window_retained_count: Papers retained within the time window (or before max_results limit).
        rule_retained_count: Papers retained after matching keyword pre-filtering rules.
        filter_keywords: Set of keyword rules used for pre-filtering.
    """

    scanned_count: int
    window_retained_count: int
    rule_retained_count: int
    filter_keywords: tuple[str, ...] = KEYWORD_PATTERNS


# arXiv official client immediately retries on num_retries > 0 without backoff.
# Disable internal retry here; exponential backoff is handled by _collect_with_backoff.
_CLIENT_NUM_RETRIES = 0
# arXiv API terms of service require at least 3 seconds between adjacent requests.
_CLIENT_DELAY_SECONDS = 3.0
_PAGE_SIZE = 100
_MAX_BACKOFF_SECONDS = 60.0
# For 429 rate limits, start at 60s up to 5 min to avoid hammering during cooldown.
_RATE_LIMIT_BASE_DELAY = 60.0
_RATE_LIMIT_MAX_BACKOFF_SECONDS = 300.0

# These exceptions indicate temporary failures worth retrying after a delay.
# Programming errors and validation errors should not be retried.
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
    """Fetcher only depends on ``results()``; tests can pass a mock that does not access the network."""

    def results(
        self, search: arxiv.Search, offset: int = 0
    ) -> Iterator[arxiv.Result]: ...


def _as_utc(value: datetime) -> datetime:
    """Convert naive datetime to UTC and timezone-aware datetime to UTC for window comparison."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _window_bounds(
    days_back: int, now: datetime | None = None
) -> tuple[datetime, datetime]:
    """Roll back ``days_back`` days from ``now`` (default current UTC) to get a closed interval window."""
    _require_positive_int(days_back, "days_back")
    end = _as_utc(now or datetime.now(UTC))
    return end - timedelta(days=days_back), end


def _updated_in_window(result: arxiv.Result, start: datetime, end: datetime) -> bool:
    """Check if paper's latest update time falls within window, covering revisions as well as initial submissions."""
    return start <= _as_utc(result.updated) <= end


def _http_status(exc: BaseException) -> int | None:
    """Extract HTTP status code from arXiv / requests exceptions."""
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
    """Use short backoff for ordinary errors; exponential starting from 60s for 429 rate limits."""
    if _is_rate_limited(exc):
        return min(
            _RATE_LIMIT_BASE_DELAY * (2**attempt),
            _RATE_LIMIT_MAX_BACKOFF_SECONDS,
        )
    return min(base_delay * (2**attempt), _MAX_BACKOFF_SECONDS)


def _require_at_least_one_limit(days_back: int | None, max_results: int | None) -> None:
    if days_back is None and max_results is None:
        raise ValueError("At least one of days_back and max_results must be provided")
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
    """Resolve fetch truncation conditions.

    When neither is given, use ``default_days_back`` (scheduler defaults to 1 day for full fetch).
    If only one is given, truncate only by that; if both are given, both take effect.
    """
    if days_back is None and max_results is None:
        if default_days_back is None:
            raise ValueError(
                "At least one of days_back and max_results must be provided"
            )
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
    """Build arXiv API ``search_query``: query only by categories.

    Official ``search_query`` does not support ``lastUpdatedDate`` filtering;
    passing it produces empty results. Date window is filtered locally by ``fetch_papers`` via ``result.updated``.
    ``days_back`` is validated if provided; ``now`` is preserved for signature compatibility.
    """
    if days_back is not None:
        _require_positive_int(days_back, "days_back")
    if not categories:
        raise ValueError("categories cannot be empty")
    del now

    category_clause = " OR ".join(f"cat:{category}" for category in categories)
    return f"({category_clause})"


def matches_world_model_keywords(title: str, abstract: str) -> bool:
    """Retain paper if any keyword appears in title or abstract. Fields matched separately to avoid false concatenation."""
    return any(
        _KEYWORD_RE.search(_normalize_whitespace(text)) is not None
        for text in (title, abstract)
    )


def result_to_raw_paper(result: arxiv.Result) -> RawPaper:
    """Convert arXiv client ``Result`` to ``RawPaper``.

    Sanitization includes: stripping version suffix from id, normalizing whitespace
    in title and abstract, keeping author names only, and falling back to id-based PDF url if missing.
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
    """Discard unrelated papers by keyword and log discard ratio.

    Ratio = discarded / input papers. When empty, ratio is 0 to avoid division by zero.
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
        "Rule pre-filtering: total %d papers, retained %d papers, discarded %d papers, discard ratio %.1f%%",
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
    """Consume results in descending update time order, truncate by days/count, and record scanned count.

    With time window: discard entries after window; stop paging once an entry before window is encountered
    (relies on ``sortBy=lastUpdatedDate`` descending). With count limit, stop once reached.
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
    logger.info(
        "Filtering complete: scanned %d papers, retained %d papers",
        scanned,
        len(selected),
    )
    return selected, scanned


def _take_results(
    results: Iterator[arxiv.Result],
    *,
    window: tuple[datetime, datetime] | None,
    max_results: int | None,
) -> list[arxiv.Result]:
    """Backward-compatible truncation returning only result list."""
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
    """Fetch papers by category, truncate and pre-filter, returning cleaned papers and FetchStats."""
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
        else "unlimited"
    )
    limit_text = "unlimited" if max_results is None else str(max_results)
    logger.info(
        "Starting arXiv fetch: %s (days_back=%s, max_results=%s, window=%s)",
        query,
        days_back if days_back is not None else "unlimited",
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
    """Fetch papers by category, truncate by days and/or count, and apply rule pre-filtering.

    Three valid combinations:

    - ``days_back`` only: page backwards by update time until out of window;
    - ``max_results`` only: fetch top N most recently updated without date filter;
    - both: take at most N papers within the time window.

    Queries only by category, descending by update time. Returns cleaned and filtered
    ``list[RawPaper]``.
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
    """Wrap a full fetch in exponential backoff.

    Standard failure waits ``min(base_delay * 2^n, 60)`` seconds.
    429 rate limit waits ``min(60 * 2^n, 300)`` seconds.
    Failure during generator paging retries from the beginning to avoid partial results.
    Re-raises the last exception once retries are exhausted.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be greater than or equal to 0")
    if base_delay < 0:
        raise ValueError("base_delay must be greater than or equal to 0")

    take = consume or (lambda results: list(results))
    attempt = 0
    while True:
        try:
            return take(client.results(search))
        except _RETRYABLE_EXCEPTIONS as exc:
            if attempt >= max_retries:
                logger.error(
                    "arXiv fetch failed after %d retries: %s", max_retries, exc
                )
                raise
            delay = _backoff_delay(exc, attempt, base_delay)
            reason = "Rate limited 429" if _is_rate_limited(exc) else str(exc)
            logger.warning(
                "arXiv fetch failed (%s), retry %d/%d, waiting %.1f seconds",
                reason,
                attempt + 1,
                max_retries,
                delay,
            )
            sleep(delay)
            attempt += 1


def _normalize_whitespace(text: str) -> str:
    """arXiv titles and abstracts often contain newlines; collapse to single spaces for keyword matching and downstream use."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer greater than or equal to 1")


def main(argv: list[str] | None = None) -> list[RawPaper]:
    """CLI entrypoint: fetch, log discard ratio, and print retained IDs and titles to stdout."""
    parser = argparse.ArgumentParser(
        description="Fetch recent arXiv papers and apply world model keyword rule pre-filtering."
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=None,
        help="Number of days to look back by last update time; fetches the entire window when not combined with --max-results",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Maximum number of papers to retain; omitted to not truncate by count",
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
    print(f"Retained {len(papers)} papers")
    return papers


if __name__ == "__main__":
    main()
