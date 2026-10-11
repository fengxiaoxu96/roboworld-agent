"""Unit and integration tests: PDF targeted section parser tool (Task 3.1)."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pymupdf
from langchain_core.tools import BaseTool

from src.tools.pdf_parser import (
    GetPaperSectionInput,
    PDFParser,
    clean_arxiv_id,
    filter_formulas_and_noise,
    get_paper_section,
    is_formula_line,
)


def _create_sample_pdf_bytes() -> bytes:
    """Build a multi-page synthetic PDF containing typical academic paper structure for testing."""
    doc = pymupdf.open()

    # Page 1: Title, abstract, introduction
    page1 = doc.new_page()
    page1.insert_text(
        (50, 50),
        (
            "arXiv:2401.12345v1 [cs.RO] 15 Jan 2024\n\n"
            "RoboWorld: An Action-Conditioned World Model for Robot Manipulation\n\n"
            "Abstract\n"
            "We propose RoboWorld, an action-conditioned latent dynamics model.\n\n"
            "1 Introduction\n"
            "Robotics manipulation requires forward physical predictive models.\n"
            "Standard video generators lack action conditioning.\n"
        ),
    )

    # Page 2: Related work and methodology (including formulas and subheadings)
    page2 = doc.new_page()
    page2.insert_text(
        (50, 50),
        (
            "2 Related Work\n"
            "Video prediction models without action inputs have limited utility.\n\n"
            "3 Methodology\n"
            "We design a latent world model conditioned on robot actions.\n"
            "Our architecture comprises a continuous latent encoder and a dynamics transformer.\n\n"
            "3.1 Latent Space Representation\n"
            "The visual observation is encoded into a compact vector representation.\n"
            "\\min_{\\theta} \\sum_{t=1}^T \\| z_t - \\hat{z}_t \\|_2^2 + \\lambda \\mathcal{L}_{reg}  (1)\n"
            "The latent space captures spatial dynamics.\n\n"
            "3.2 Action-Conditioned Dynamics\n"
            "Given action a_t, the transition model predicts future states.\n"
            "z_{t+1} = f_\\theta(z_t, a_t) + \\epsilon, \\quad \\epsilon \\sim \\mathcal{N}(0, I)\n"
            "This enables forward simulation for policy evaluation.\n"
        ),
    )

    # Page 3: Experiments and conclusion
    page3 = doc.new_page()
    page3.insert_text(
        (50, 50),
        (
            "4 Experiments\n"
            "We evaluate RoboWorld on CALVIN simulation benchmark and real Franka Panda robots.\n\n"
            "4.1 CALVIN Benchmark Results\n"
            "On CALVIN long-horizon tasks, our approach achieves 92.5% success rate.\n\n"
            "4.2 Real-Robot Experiments\n"
            "On real-world table manipulation, our model accurately anticipates contact transitions.\n\n"
            "5 Conclusion\n"
            "We presented an action-conditioned world model.\n\n"
            "References\n"
            "[1] Ha and Schmidhuber. World Models. NeurIPS 2018.\n"
            "[2] Hafner et al. DreamerV3. Nature 2025.\n"
        ),
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# =========================================================================
# 1. Identifier cleaning and formula filtering tests
# =========================================================================


def test_clean_arxiv_id() -> None:
    assert clean_arxiv_id("2401.12345") == "2401.12345"
    assert clean_arxiv_id("2401.12345v2") == "2401.12345v2"
    assert clean_arxiv_id("arxiv:2401.12345") == "2401.12345"
    assert clean_arxiv_id("arXiv: 2401.12345v1") == "2401.12345v1"
    assert clean_arxiv_id("https://arxiv.org/abs/2401.12345") == "2401.12345"
    assert clean_arxiv_id("https://arxiv.org/pdf/2401.12345.pdf") == "2401.12345"
    assert clean_arxiv_id("2401.12345.pdf") == "2401.12345"


def test_is_formula_line() -> None:
    # Typical standalone formula with equation label (1)
    eq_line = r"\min_{\theta} \sum_{t=1}^T \| z_t - \hat{z}_t \|_2^2 + \lambda \mathcal{L}_{reg}  (1)"
    assert is_formula_line(eq_line) is True

    # Typical dense math symbols line
    math_line = r"z_{t+1} = f_\theta(z_t, a_t) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)"
    assert is_formula_line(math_line) is True

    # Regular natural language statement (even with variable) should not be treated as formula
    text_line = "We design a latent world model conditioned on robot actions."
    assert is_formula_line(text_line) is False

    text_with_inline_var = (
        "Given the action sequence a_t, our model predicts the future latent states."
    )
    assert is_formula_line(text_with_inline_var) is False


def test_filter_formulas_and_noise() -> None:
    raw = (
        "arXiv:2401.12345v1 [cs.RO] 15 Jan 2024\n"
        "Page 2 of 10\n"
        "This is valid methodology text describing the model architecture.\n"
        r"\min_{\theta} \sum_{t=1}^T \| z_t - \hat{z}_t \|_2^2  (1)"
        "\n"
        "Here is the subsequent explanation of the dynamics loss."
    )
    cleaned = filter_formulas_and_noise(raw)
    assert "arXiv:2401" not in cleaned
    assert "Page 2" not in cleaned
    assert r"\min_{\theta}" not in cleaned
    assert "This is valid methodology text" in cleaned
    assert "Here is the subsequent explanation" in cleaned


# =========================================================================
# 2. Section extraction core logic tests
# =========================================================================


def test_extract_methodology_section() -> None:
    pdf_bytes = _create_sample_pdf_bytes()
    parser = PDFParser()

    method_text = parser.get_section_text(pdf_bytes, "Methodology")
    assert "3 Methodology" in method_text or "Methodology" in method_text
    assert "latent world model conditioned on robot actions" in method_text
    # Must include subsections 3.1 and 3.2
    assert "3.1 Latent Space Representation" in method_text
    assert "3.2 Action-Conditioned Dynamics" in method_text
    assert "policy evaluation" in method_text
    # Long formulas must be filtered
    assert r"\min_{\theta}" not in method_text
    # Must not cross into 4 Experiments
    assert "4 Experiments" not in method_text
    # Must never include references
    assert "References" not in method_text
    assert "DreamerV3" not in method_text


def test_extract_experiments_section() -> None:
    pdf_bytes = _create_sample_pdf_bytes()
    parser = PDFParser()

    exp_text = parser.get_section_text(pdf_bytes, "Experiments")
    assert "4 Experiments" in exp_text
    assert "CALVIN simulation benchmark" in exp_text
    assert "4.1 CALVIN Benchmark Results" in exp_text
    assert "4.2 Real-Robot Experiments" in exp_text
    # Must not cross into 5 Conclusion or References
    assert "5 Conclusion" not in exp_text
    assert "References" not in exp_text
    assert "Ha and Schmidhuber" not in exp_text


def test_synonym_matching() -> None:
    pdf_bytes = _create_sample_pdf_bytes()
    parser = PDFParser()

    # Query "method" should match "3 Methodology"
    text_method = parser.get_section_text(pdf_bytes, "method")
    assert "latent world model" in text_method

    # Query "evaluation" should match "4 Experiments"
    text_eval = parser.get_section_text(pdf_bytes, "evaluation")
    assert "CALVIN" in text_eval


def test_section_not_found() -> None:
    pdf_bytes = _create_sample_pdf_bytes()
    parser = PDFParser()

    result = parser.get_section_text(pdf_bytes, "NonExistentSection")
    assert "could not locate" in result.lower()


# =========================================================================
# 3. Local cache and performance acceptance criteria tests (< 3s)
# =========================================================================


def test_local_pdf_path_and_cache(tmp_path: Path) -> None:
    pdf_bytes = _create_sample_pdf_bytes()
    local_pdf = tmp_path / "test_paper.pdf"
    local_pdf.write_bytes(pdf_bytes)

    cache_dir = tmp_path / "cache"
    parser = PDFParser(cache_dir=cache_dir)

    # 1. Directly specify local file
    resolved = parser.resolve_pdf_path(str(local_pdf))
    assert resolved == local_pdf.resolve()

    # 2. Simulate network download and write to local cache
    mock_resp = MagicMock()
    mock_resp.content = pdf_bytes
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp) as mock_get:
        # First call: cache miss, triggers network request and saves to disk
        path1 = parser.resolve_pdf_path("2401.99999")
        assert path1.is_file()
        assert mock_get.call_count == 1

        # Second call: cache hit, no network request
        path2 = parser.resolve_pdf_path("2401.99999")
        assert path2 == path1
        assert mock_get.call_count == 1


def test_acceptance_criteria_parsing_speed(tmp_path: Path) -> None:
    """Acceptance criteria: return cleaned section paragraph text within 3 seconds for given arXiv ID."""
    pdf_bytes = _create_sample_pdf_bytes()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Pre-place cache file
    cached_pdf = cache_dir / "2401.12345.pdf"
    cached_pdf.write_bytes(pdf_bytes)

    parser = PDFParser(cache_dir=cache_dir)

    t0 = time.perf_counter()
    section_text = parser.get_section_text("2401.12345", "Methodology")
    elapsed = time.perf_counter() - t0

    assert "Methodology" in section_text
    # Local parsing should finish in under 3 seconds
    assert elapsed < 3.0, f"Parsing took {elapsed:.2f}s, exceeding 3.0s requirement"


# =========================================================================
# 4. Agent Tool interface contract tests
# =========================================================================


def test_get_paper_section_tool_structure() -> None:
    """Verify compliance with LangChain standard Agent Tool interface specification."""
    assert isinstance(get_paper_section, BaseTool)
    assert get_paper_section.name == "get_paper_section"
    assert get_paper_section.args_schema == GetPaperSectionInput
    assert "Methodology" in get_paper_section.description


def test_get_paper_section_invocations(tmp_path: Path) -> None:
    pdf_bytes = _create_sample_pdf_bytes()
    sample_file = tmp_path / "paper.pdf"
    sample_file.write_bytes(pdf_bytes)

    # 1. Supports standard Python positional argument calls
    res1 = get_paper_section(str(sample_file), "Methodology")
    assert "latent world model" in res1

    # 2. Supports standard Python keyword argument calls
    res2 = get_paper_section(arxiv_id=str(sample_file), section_name="Methodology")
    assert "latent world model" in res2

    # 3. Supports LangChain Tool invoke calls
    res3 = get_paper_section.invoke(
        {"arxiv_id": str(sample_file), "section_name": "Methodology"}
    )
    assert "latent world model" in res3


def test_get_paper_section_error_handling() -> None:
    """Verify that network failures or missing files return friendly error text without crashing."""
    with patch("requests.get", side_effect=Exception("Connection refused")):
        res = get_paper_section("9999.99999_non_existent", "Methodology")
        assert "failed" in res.lower() or "could not" in res.lower()


def test_online_pdf_in_memory_reading_no_disk_cache(tmp_path: Path) -> None:
    """Test online in-memory streaming of PDF via arXiv ID or link without writing cache files to disk."""
    pdf_bytes = _create_sample_pdf_bytes()
    cache_dir = tmp_path / "cache"
    parser = PDFParser(cache_dir=cache_dir)

    mock_resp = MagicMock()
    mock_resp.content = pdf_bytes
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp) as mock_get:
        # 1. Online read via arXiv ID
        sec_text = parser.get_section_text("2401.55555", "Methodology")
        assert "Methodology" in sec_text
        assert "latent world model" in sec_text
        # Verify online request was made
        assert mock_get.call_count == 1
        args, _ = mock_get.call_args
        assert "https://arxiv.org/pdf/2401.55555.pdf" in args[0]
        # Core assertion: No files generated in local cache directory!
        assert not cache_dir.exists() or len(list(cache_dir.glob("*.pdf"))) == 0

        # 2. Online read via full arXiv PDF URL
        sec_text_url = parser.get_section_text(
            "https://arxiv.org/pdf/2401.55555.pdf", "Methodology"
        )
        assert "Methodology" in sec_text_url
        assert not cache_dir.exists() or len(list(cache_dir.glob("*.pdf"))) == 0


def test_get_paper_section_online_url_invocation() -> None:
    """Test that get_paper_section supports passing online arXiv URL and reading in memory."""
    pdf_bytes = _create_sample_pdf_bytes()
    mock_resp = MagicMock()
    mock_resp.content = pdf_bytes
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp) as mock_get:
        res = get_paper_section("https://arxiv.org/abs/2401.77777", "Methodology")
        assert "latent world model" in res
        args, _ = mock_get.call_args
        assert "2401.77777.pdf" in args[0]
