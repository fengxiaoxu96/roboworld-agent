"""采集器与规则初筛的单元测试。不访问 arXiv 网络。"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import arxiv
import pytest
from pydantic import ValidationError

from src.core.schema import RawPaper
from src.fetchers import arxiv_fetcher as fetcher


class _FakeClient:
    """按预定次数抛出可重试错误，之后返回固定结果。"""

    def __init__(self, results: list[arxiv.Result], failures: int = 0) -> None:
        self._results = results
        self._failures = failures
        self.calls = 0

    def results(self, search: arxiv.Search, offset: int = 0):
        self.calls += 1
        if self.calls <= self._failures:
            raise arxiv.HTTPError("https://export.arxiv.org/api/query", self.calls, 503)
        return iter(self._results)


def _result(
    *,
    entry_id: str = "https://arxiv.org/abs/2609.12345v2",
    title: str = "A World Model for Manipulation",
    summary: str = "We learn latent dynamics for embodied control.",
    pdf: str | None = "https://arxiv.org/pdf/2609.12345v2",
) -> arxiv.Result:
    published = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
    links = []
    if pdf is not None:
        links.append(arxiv.Result.Link(pdf, title="pdf"))
    return arxiv.Result(
        entry_id=entry_id,
        updated=published,
        published=published,
        title=title,
        authors=[
            arxiv.Result.Author("Ada Lovelace"),
            arxiv.Result.Author("Alan Turing"),
        ],
        summary=summary,
        primary_category="cs.RO",
        categories=["cs.RO", "cs.LG"],
        links=links,
    )


def test_build_search_query_covers_categories_and_update_window() -> None:
    now = datetime(2026, 9, 29, 15, 30, tzinfo=UTC)
    query = fetcher.build_search_query(2, now=now)

    assert "cat:cs.RO" in query
    assert "cat:cs.AI" in query
    assert "cat:cs.LG" in query
    assert "lastUpdatedDate:[202609271530 TO 202609291530]" in query
    assert query.startswith("(") and " AND " in query


def test_keyword_filter_keeps_world_model_and_drops_unrelated() -> None:
    assert fetcher.matches_world_model_keywords(
        "WORLD-MODEL for control",
        "unrelated abstract",
    )
    assert fetcher.matches_world_model_keywords(
        "Planning",
        "A forward-model of video prediction.",
    )
    assert not fetcher.matches_world_model_keywords(
        "Image classification with transformers",
        "We report accuracy on ImageNet.",
    )


def test_keyword_match_does_not_cross_title_and_abstract() -> None:
    assert not fetcher.matches_world_model_keywords("A world", "model of weather")


def test_result_to_raw_paper_cleans_id_whitespace_and_pdf() -> None:
    paper = fetcher.result_to_raw_paper(
        _result(
            title="A World Model\nfor Manipulation",
            summary="Latent dynamics\n\nin embodied agents.",
            pdf=None,
        )
    )

    assert isinstance(paper, RawPaper)
    assert paper.arxiv_id == "2609.12345"
    assert paper.title == "A World Model for Manipulation"
    assert paper.abstract == "Latent dynamics in embodied agents."
    assert paper.authors == ["Ada Lovelace", "Alan Turing"]
    assert paper.categories == ["cs.RO", "cs.LG"]
    assert paper.pdf_url == "https://arxiv.org/pdf/2609.12345.pdf"
    assert paper.published_date == datetime(2026, 9, 28, 12, 0, tzinfo=UTC)


def test_fetch_papers_returns_filtered_raw_papers_and_logs_discard_ratio(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _FakeClient(
        [
            _result(),
            _result(
                entry_id="https://arxiv.org/abs/2609.99999v1",
                title="Image classification",
                summary="Convolutional networks on ImageNet.",
            ),
        ]
    )

    with caplog.at_level(logging.INFO):
        papers = fetcher.fetch_papers(
            days_back=3,
            max_results=10,
            client=client,
            now=datetime(2026, 9, 29, tzinfo=UTC),
        )

    assert client.calls == 1
    assert len(papers) == 1
    assert papers[0].arxiv_id == "2609.12345"
    assert "丢弃比例 50.0%" in caplog.text
    assert "保留 1 篇" in caplog.text


def test_fetch_retries_with_exponential_backoff_then_succeeds() -> None:
    slept: list[float] = []
    client = _FakeClient([_result()], failures=3)

    papers = fetcher.fetch_papers(
        days_back=1,
        max_results=5,
        client=client,
        max_retries=3,
        base_delay=1.0,
        sleep=slept.append,
        now=datetime(2026, 9, 29, tzinfo=UTC),
    )

    assert slept == [1.0, 2.0, 4.0]
    assert client.calls == 4
    assert len(papers) == 1


def test_fetch_raises_after_retries_are_exhausted() -> None:
    slept: list[float] = []
    client = _FakeClient([], failures=5)

    with pytest.raises(arxiv.HTTPError):
        fetcher.fetch_papers(
            days_back=1,
            max_results=5,
            client=client,
            max_retries=2,
            base_delay=0.5,
            sleep=slept.append,
            now=datetime(2026, 9, 29, tzinfo=UTC),
        )

    assert slept == [0.5, 1.0]
    assert client.calls == 3


def test_programming_errors_are_not_retried() -> None:
    class _BrokenClient:
        def __init__(self) -> None:
            self.calls = 0

        def results(self, search: arxiv.Search, offset: int = 0):
            self.calls += 1
            raise ValueError("bad query")

    client = _BrokenClient()
    with pytest.raises(ValueError, match="bad query"):
        fetcher.fetch_papers(
            days_back=1, max_results=1, client=client, sleep=lambda _: None
        )

    assert client.calls == 1


def test_invalid_window_and_limit_are_rejected() -> None:
    with pytest.raises(ValueError):
        fetcher.build_search_query(0)
    with pytest.raises(ValueError):
        fetcher.fetch_papers(days_back=1, max_results=0)


def test_raw_paper_requires_published_date() -> None:
    with pytest.raises(ValidationError):
        RawPaper(  # type: ignore[call-arg]
            arxiv_id="1",
            title="t",
            authors=["a"],
            abstract="world model",
            categories=["cs.RO"],
            pdf_url="https://arxiv.org/pdf/1.pdf",
        )
