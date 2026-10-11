from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.core.schema import ModelParadigm, PaperInsight, RawPaper

# ============================================================================
# Tests for RawPaper
# ============================================================================


def test_raw_paper_valid_instantiation_with_datetime() -> None:
    published = datetime(2026, 9, 30, 12, 0, 0, tzinfo=UTC)
    paper = RawPaper(
        arxiv_id="2609.12345",
        title="A World Model for Robotic Manipulation",
        authors=["Ada Lovelace", "Alan Turing"],
        abstract="We propose a latent dynamics model conditioned on robot actions.",
        categories=["cs.RO", "cs.AI"],
        published_date=published,
        pdf_url="https://arxiv.org/pdf/2609.12345.pdf",
    )

    assert paper.arxiv_id == "2609.12345"
    assert paper.title == "A World Model for Robotic Manipulation"
    assert paper.authors == ["Ada Lovelace", "Alan Turing"]
    assert (
        paper.abstract
        == "We propose a latent dynamics model conditioned on robot actions."
    )
    assert paper.categories == ["cs.RO", "cs.AI"]
    assert paper.published_date == published
    assert paper.pdf_url == "https://arxiv.org/pdf/2609.12345.pdf"


def test_raw_paper_valid_from_json_and_iso_datetime() -> None:
    raw_data = {
        "arxiv_id": "2609.54321",
        "title": "Latent Dynamics in CALVIN",
        "authors": ["John Doe"],
        "abstract": "Benchmarking world models in simulated manipulation tasks.",
        "categories": ["cs.RO"],
        "published_date": "2026-09-28T08:30:00Z",
        "pdf_url": "https://arxiv.org/pdf/2609.54321.pdf",
    }
    paper = RawPaper.model_validate(raw_data)
    assert paper.arxiv_id == "2609.54321"
    assert paper.published_date.year == 2026
    assert paper.published_date.month == 9
    assert paper.published_date.day == 28

    json_str = paper.model_dump_json()
    reconstituted = RawPaper.model_validate_json(json_str)
    assert reconstituted == paper


@pytest.mark.parametrize(
    "missing_field",
    [
        "arxiv_id",
        "title",
        "authors",
        "abstract",
        "categories",
        "published_date",
        "pdf_url",
    ],
)
def test_raw_paper_missing_required_fields_raises_validation_error(
    missing_field: str,
) -> None:
    full_data = {
        "arxiv_id": "2609.12345",
        "title": "Title",
        "authors": ["Author"],
        "abstract": "Abstract",
        "categories": ["cs.RO"],
        "published_date": datetime(2026, 9, 30, tzinfo=UTC),
        "pdf_url": "https://arxiv.org/pdf/2609.12345.pdf",
    }
    data = {k: v for k, v in full_data.items() if k != missing_field}
    with pytest.raises(ValidationError) as exc_info:
        RawPaper.model_validate(data)

    errors = exc_info.value.errors()
    assert any(err["loc"] == (missing_field,) for err in errors)


def test_raw_paper_invalid_date_type_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        RawPaper(
            arxiv_id="2609.12345",
            title="Title",
            authors=["Author"],
            abstract="Abstract",
            categories=["cs.RO"],
            published_date="invalid-date-string",  # type: ignore[arg-type]
            pdf_url="https://arxiv.org/pdf/2609.12345.pdf",
        )


def test_raw_paper_invalid_authors_type_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        RawPaper(
            arxiv_id="2609.12345",
            title="Title",
            authors="Not A List",  # type: ignore[arg-type]
            abstract="Abstract",
            categories=["cs.RO"],
            published_date=datetime(2026, 9, 30, tzinfo=UTC),
            pdf_url="https://arxiv.org/pdf/2609.12345.pdf",
        )


# ============================================================================
# Tests for PaperInsight
# ============================================================================


@pytest.mark.parametrize(
    "paradigm",
    [
        "Diffusion-based",
        "Transformer/AutoRegressive",
        "Flow-Matching",
        "State-Space/Mamba",
        "Other",
    ],
)
def test_paper_insight_valid_paradigms(paradigm: ModelParadigm) -> None:
    insight = PaperInsight(
        is_world_model=True,
        model_paradigm=paradigm,
        action_conditioned=True,
        eval_environments=["Isaac Lab", "Real Robot"],
        core_innovation="Proposes a flow-matching-based embodied dynamics prediction model, achieving closed-loop action control on a real robot arm.",
    )

    assert insight.is_world_model is True
    assert insight.model_paradigm == paradigm
    assert insight.action_conditioned is True
    assert insight.eval_environments == ["Isaac Lab", "Real Robot"]
    assert "flow-matching" in insight.core_innovation.lower()


def test_paper_insight_serialization_round_trip() -> None:
    data = {
        "is_world_model": False,
        "model_paradigm": "Transformer/AutoRegressive",
        "action_conditioned": False,
        "eval_environments": [],
        "core_innovation": "Pure generic video generation model without robot action or dynamics modeling.",
    }
    insight = PaperInsight.model_validate(data)
    json_bytes = insight.model_dump_json()
    reconstituted = PaperInsight.model_validate_json(json_bytes)
    assert reconstituted == insight
    assert reconstituted.model_dump() == data


@pytest.mark.parametrize(
    "invalid_paradigm",
    [
        "GAN",
        "diffusion-based",  # Case-sensitive check
        "other",
        "Transformer",
        "Mamba",
        "",
        "Random",
    ],
)
def test_paper_insight_invalid_model_paradigm_raises_validation_error(
    invalid_paradigm: str,
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        PaperInsight(
            is_world_model=True,
            model_paradigm=invalid_paradigm,  # type: ignore[arg-type]
            action_conditioned=True,
            eval_environments=["Isaac Gym"],
            core_innovation="Innovation summary",
        )

    errors = exc_info.value.errors()
    assert any("model_paradigm" in str(err["loc"]) for err in errors)


@pytest.mark.parametrize(
    "missing_field",
    [
        "is_world_model",
        "model_paradigm",
        "action_conditioned",
        "eval_environments",
        "core_innovation",
    ],
)
def test_paper_insight_missing_required_fields_raises_validation_error(
    missing_field: str,
) -> None:
    full_data = {
        "is_world_model": True,
        "model_paradigm": "Diffusion-based",
        "action_conditioned": True,
        "eval_environments": ["CALVIN"],
        "core_innovation": "Test innovation",
    }
    data = {k: v for k, v in full_data.items() if k != missing_field}
    with pytest.raises(ValidationError) as exc_info:
        PaperInsight.model_validate(data)

    errors = exc_info.value.errors()
    assert any(err["loc"] == (missing_field,) for err in errors)


def test_paper_insight_invalid_field_types_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        PaperInsight(
            is_world_model="not_a_boolean",  # type: ignore[arg-type]
            model_paradigm="Diffusion-based",
            action_conditioned=True,
            eval_environments=["CALVIN"],
            core_innovation="Test innovation",
        )

    with pytest.raises(ValidationError):
        PaperInsight(
            is_world_model=True,
            model_paradigm="Diffusion-based",
            action_conditioned=True,
            eval_environments="not_a_list",  # type: ignore[arg-type]
            core_innovation="Test innovation",
        )
