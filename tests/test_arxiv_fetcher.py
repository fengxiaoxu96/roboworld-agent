"""Unit tests for fetcher and rule pre-filtering. Does not access arXiv network."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import arxiv
import pytest
from pydantic import ValidationError

from src.core.schema import RawPaper
from src.fetchers import arxiv_fetcher as fetcher


class _FakeClient:
    """Raises retryable errors a predetermined number of times, then returns fixed results."""

    def __init__(
        self,
        results: list[arxiv.Result],
        failures: int = 0,
        status: int = 503,
    ) -> None:
        self._results = results
        self._failures = failures
        self._status = status
        self.calls = 0
        self.last_search: arxiv.Search | None = None

    def results(self, search: arxiv.Search, offset: int = 0):
        self.calls += 1
        self.last_search = search
        if self.calls <= self._failures:
            raise arxiv.HTTPError(
                "https://export.arxiv.org/api/query", self.calls, self._status
            )
        return iter(self._results)


def _result(
    *,
    entry_id: str = "https://arxiv.org/abs/2609.12345v2",
    title: str = "A World Model for Manipulation",
    summary: str = "We learn latent dynamics for embodied control.",
    pdf: str | None = "https://arxiv.org/pdf/2609.12345v2",
    published: datetime | None = None,
    updated: datetime | None = None,
) -> arxiv.Result:
    published = published or datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
    updated = updated if updated is not None else published
    links = []
    if pdf is not None:
        links.append(arxiv.Result.Link(pdf, title="pdf"))
    return arxiv.Result(
        entry_id=entry_id,
        updated=updated,
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


def test_build_search_query_covers_categories_without_date_filter() -> None:
    now = datetime(2026, 9, 29, 15, 30, tzinfo=UTC)
    query = fetcher.build_search_query(2, now=now)

    assert query == "(cat:cs.RO OR cat:cs.AI OR cat:cs.LG)"
    assert "lastUpdatedDate" not in query
    assert "submittedDate" not in query


def test_window_bounds_use_utc_rolling_interval() -> None:
    now = datetime(2026, 9, 29, 15, 30, tzinfo=UTC)
    start, end = fetcher._window_bounds(2, now)

    assert start == datetime(2026, 9, 27, 15, 30, tzinfo=UTC)
    assert end == now


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
    assert not fetcher.matches_world_model_keywords(
        "Embodied Agent Navigation",
        "We propose an embodied benchmark for indoor navigation.",
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
    assert "Filtering complete: scanned 2 papers, retained 2 papers" in caplog.text
    assert "discard ratio 50.0%" in caplog.text
    assert "retained 1 papers" in caplog.text


def test_fetch_papers_with_stats_records_counts_and_keywords() -> None:
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

    papers, stats = fetcher.fetch_papers_with_stats(
        days_back=3,
        max_results=10,
        client=client,
        now=datetime(2026, 9, 29, tzinfo=UTC),
    )

    assert len(papers) == 1
    assert stats.scanned_count == 2
    assert stats.window_retained_count == 2
    assert stats.rule_retained_count == 1
    assert stats.filter_keywords == fetcher.KEYWORD_PATTERNS


def test_fetch_papers_drops_updates_outside_local_window(
    caplog: pytest.LogCaptureFixture,
) -> None:
    now = datetime(2026, 9, 29, 15, 30, tzinfo=UTC)
    client = _FakeClient(
        [
            _result(
                entry_id="https://arxiv.org/abs/2609.00002v1",
                title="Naive TZ World Model",
                updated=datetime(2026, 9, 28, 18, 0),  # noqa: DTZ001 intentionally test naive datetime
            ),
            _result(),
            _result(
                entry_id="https://arxiv.org/abs/2609.00001v1",
                title="An Older World Model",
                updated=datetime(2026, 9, 20, 12, 0, tzinfo=UTC),
            ),
        ]
    )

    with caplog.at_level(logging.INFO):
        papers = fetcher.fetch_papers(
            days_back=2,
            max_results=10,
            client=client,
            now=now,
        )

    assert {paper.arxiv_id for paper in papers} == {"2609.12345", "2609.00002"}
    assert "Filtering complete: scanned 3 papers, retained 2 papers" in caplog.text
    assert "discard ratio 0.0%" in caplog.text
    assert client.last_search is not None
    assert client.last_search.max_results is None


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
    with pytest.raises(ValueError, match="At least one of days_back and max_results"):
        fetcher.fetch_papers()
    with pytest.raises(ValueError):
        fetcher.fetch_papers(days_back=1, max_results=0)
    with pytest.raises(ValueError):
        fetcher.fetch_papers(max_results=0)


def test_days_only_pages_until_window_without_count_cap() -> None:
    now = datetime(2026, 9, 29, 15, 30, tzinfo=UTC)
    client = _FakeClient(
        [
            _result(entry_id="https://arxiv.org/abs/2609.00011v1"),
            _result(
                entry_id="https://arxiv.org/abs/2609.00012v1",
                title="Second World Model",
                updated=datetime(2026, 9, 28, 8, 0, tzinfo=UTC),
            ),
            _result(
                entry_id="https://arxiv.org/abs/2609.00013v1",
                title="Outside World Model",
                updated=datetime(2026, 9, 20, 12, 0, tzinfo=UTC),
            ),
        ]
    )

    papers = fetcher.fetch_papers(days_back=2, client=client, now=now)

    assert [paper.arxiv_id for paper in papers] == ["2609.00011", "2609.00012"]
    assert client.last_search is not None
    assert client.last_search.max_results is None


def test_count_only_keeps_newest_without_date_window() -> None:
    client = _FakeClient(
        [
            _result(entry_id="https://arxiv.org/abs/2609.00021v1"),
            _result(
                entry_id="https://arxiv.org/abs/2609.00022v1",
                title="Old but counted World Model",
                updated=datetime(2020, 1, 1, tzinfo=UTC),
            ),
            _result(entry_id="https://arxiv.org/abs/2609.00023v1"),
        ]
    )

    papers = fetcher.fetch_papers(max_results=2, client=client)

    assert [paper.arxiv_id for paper in papers] == ["2609.00021", "2609.00022"]
    assert client.last_search is not None
    assert client.last_search.max_results == 2


def test_days_and_count_apply_together() -> None:
    now = datetime(2026, 9, 29, 15, 30, tzinfo=UTC)
    client = _FakeClient(
        [
            _result(entry_id="https://arxiv.org/abs/2609.00031v1"),
            _result(entry_id="https://arxiv.org/abs/2609.00032v1"),
            _result(entry_id="https://arxiv.org/abs/2609.00033v1"),
        ]
    )

    papers = fetcher.fetch_papers(
        days_back=3,
        max_results=2,
        client=client,
        now=now,
    )

    assert [paper.arxiv_id for paper in papers] == ["2609.00031", "2609.00032"]
    assert client.last_search is not None
    assert client.last_search.max_results is None


def test_rate_limit_429_uses_longer_backoff() -> None:
    slept: list[float] = []
    client = _FakeClient([_result()], failures=2, status=429)

    papers = fetcher.fetch_papers(
        days_back=1,
        client=client,
        max_retries=3,
        base_delay=1.0,
        sleep=slept.append,
        now=datetime(2026, 9, 29, tzinfo=UTC),
    )

    assert slept == [60.0, 120.0]
    assert client.calls == 3
    assert len(papers) == 1


def test_resolve_fetch_limits_defaults_to_one_day() -> None:
    assert fetcher.resolve_fetch_limits(None, None) == (1, None)
    assert fetcher.resolve_fetch_limits(2, None) == (2, None)
    assert fetcher.resolve_fetch_limits(None, 50) == (None, 50)
    assert fetcher.resolve_fetch_limits(1, 50) == (1, 50)


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
