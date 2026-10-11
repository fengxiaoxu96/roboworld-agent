"""PDF targeted section parser tool (Task 3.1).

Empowers the Agent to inspect key paper sections when abstract semantics are insufficient for a definitive judgment.
Key features:
1. Pure text stream extraction using PyMuPDF preserving geometric reading order (sort=True);
2. Local cache mechanism (data/pdf_cache), supporting sub-second repeat reads and offline execution;
3. Smart regex identification of multi-level headings (Methodology, Method, Experiments, etc.),
   targeted extraction of core paragraphs including subheadings;
4. Strict filtering of references (References/Bibliography) and standalone/long formula blocks;
5. Packaged as a standard Agent Tool: get_paper_section(arxiv_id: str, section_name: str) -> str,
   supporting both LangChain Tool invocation (.invoke / .run) and native Python calls.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import pymupdf
import requests
from langchain_core.tools import StructuredTool, tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Default cache directory: data/pdf_cache under project root
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "pdf_cache"

# arXiv download request timeout (seconds)
DEFAULT_DOWNLOAD_TIMEOUT = 15.0

# Common section name synonym mapping
SECTION_SYNONYMS: dict[str, list[str]] = {
    "method": [
        "methodology",
        "method",
        "methods",
        "approach",
        "proposed approach",
        "proposed method",
        "model architecture",
        "framework",
        "system architecture",
        "system overview",
        "our approach",
        "technical approach",
    ],
    "methodology": [
        "methodology",
        "method",
        "methods",
        "approach",
        "proposed approach",
        "proposed method",
        "model architecture",
        "framework",
        "system architecture",
        "system overview",
        "our approach",
        "technical approach",
    ],
    "experiments": [
        "experiments",
        "experiment",
        "evaluation",
        "evaluations",
        "experimental results",
        "empirical evaluation",
        "experimental evaluation",
        "experiments and results",
        "results and discussion",
        "experimental study",
    ],
    "experiment": [
        "experiments",
        "experiment",
        "evaluation",
        "evaluations",
        "experimental results",
        "empirical evaluation",
        "experimental evaluation",
        "experiments and results",
        "results and discussion",
        "experimental study",
    ],
    "evaluation": [
        "evaluation",
        "evaluations",
        "experiments",
        "experiment",
        "experimental results",
        "empirical evaluation",
        "experimental evaluation",
    ],
}

# Terminal section keywords (truncate immediately, never include references or appendices)
TERMINAL_SECTION_KEYWORDS: tuple[str, ...] = (
    "references",
    "bibliography",
    "literature cited",
    "acknowledgments",
    "acknowledgements",
    "appendix",
    "appendices",
)

# Common top-level section names (used to truncate same-level or subsequent sections)
TOP_LEVEL_SECTIONS: tuple[str, ...] = (
    "introduction",
    "related work",
    "background",
    "preliminaries",
    "methodology",
    "method",
    "methods",
    "approach",
    "experiments",
    "experiment",
    "evaluation",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "future work",
    "broader impact",
    "ethics statement",
    *TERMINAL_SECTION_KEYWORDS,
)

# Math symbol and LaTeX command patterns
_MATH_SYMBOLS_RE = re.compile(r"[=<>≤≥≠±×÷∑∏∫∂∇√∝∈∉⊂⊆∪∩∧∨¬→⇒⇔∼≈≡∀∃⊗⊕⊙⟨⟩∥\|\+\-\*/\^_]")
_LATEX_CMD_RE = re.compile(
    r"\\(?:min|max|sum|prod|int|frac|sqrt|mathbf|mathcal|mathbb|mathrm|text|left|right|"
    r"vec|hat|bar|tilde|partial|nabla|sim|approx|times|quad|qquad|alpha|beta|gamma|"
    r"delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|"
    r"upsilon|phi|chi|psi|omega)\b",
    re.IGNORECASE,
)
_EQUATION_LABEL_RE = re.compile(r"\(\s*\d{1,3}\s*\)\s*$")
_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)


def clean_arxiv_id(arxiv_id: str) -> str:
    """Normalize arXiv identifier.

    Examples:
    - 'arxiv:2401.12345' -> '2401.12345'
    - '2401.12345v2' -> '2401.12345v2'
    - 'https://arxiv.org/abs/2401.12345' -> '2401.12345'
    - 'https://arxiv.org/pdf/2401.12345.pdf' -> '2401.12345'
    """
    clean = arxiv_id.strip()
    # Strip URL prefix
    clean = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", clean, flags=re.IGNORECASE)
    # Strip .pdf suffix
    clean = re.sub(r"\.pdf$", "", clean, flags=re.IGNORECASE)
    # Strip 'arxiv:' prefix
    clean = re.sub(r"^arxiv:\s*", "", clean, flags=re.IGNORECASE)
    return clean.strip()


def is_formula_line(line: str) -> bool:
    """Determine whether a single line is a long formula, standalone formula block, or formula fragment.

    Filtering rules:
    1. If line contains sufficient natural language words (>= 5 words), treat as narrative;
    2. If explicit LaTeX math environment tags (\\begin{equation} etc.) are present, treat as formula;
    3. If ends with label like (1) and contains math symbols/commands, treat as standalone formula;
    4. If math symbols or LaTeX commands are dense and natural words are minimal, treat as formula;
    5. If non-whitespace and alpha character ratio is too low (< 35%) with effective words <= 1, treat as formula fragment.
    """
    s = line.strip()
    if not s:
        return False

    # Extract natural English words (at least 2 letters)
    words = [w for w in s.split() if re.match(r"^[a-zA-Z]{2,}[,.]?$", w)]
    if len(words) >= 5:
        return False

    # Explicit LaTeX equation environment
    if re.search(r"\\(?:begin|end)\{(?:equation|align|gather|multline)\*?\}", s):
        return True

    math_symbols_count = len(_MATH_SYMBOLS_RE.findall(s))
    latex_cmds_count = len(_LATEX_CMD_RE.findall(s))
    total_math_indicators = math_symbols_count + latex_cmds_count
    has_eq_label = bool(_EQUATION_LABEL_RE.search(s))

    # Ends with equation label (1) and has math features
    if has_eq_label and total_math_indicators >= 1:
        return True

    # Dense math markers with few natural words
    if total_math_indicators >= 2 and len(words) <= 2:
        return True

    # Symbol line with very low alpha ratio
    non_space_chars = len(re.sub(r"\s", "", s))
    alpha_chars = sum(1 for c in s if c.isalpha())
    return (
        non_space_chars >= 8
        and (alpha_chars / non_space_chars) < 0.35
        and len(words) <= 1
    )


def filter_formulas_and_noise(text: str) -> str:
    """Filter long formulas, standalone equations, and page number / watermark noise from text stream."""
    lines = text.splitlines()
    filtered_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            filtered_lines.append("")
            continue

        # Filter arXiv watermark and page number lines
        if re.match(r"^arxiv:\d+\.\d+", stripped, re.IGNORECASE):
            continue
        if re.match(r"^(?:page\s+)?\d+(?:\s+of\s+\d+)?$", stripped, re.IGNORECASE):
            continue

        # Filter formula lines
        if is_formula_line(stripped):
            continue

        filtered_lines.append(line)

    # Normalize excessive blank lines (collapse more than 2 consecutive blank lines to 2)
    result = "\n".join(filtered_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


class PDFParser:
    """PDF text extraction and targeted section parser."""

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        download_timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
    ) -> None:
        self.cache_dir = Path(
            cache_dir or os.getenv("PDF_CACHE_DIR") or DEFAULT_CACHE_DIR
        )
        self.download_timeout = download_timeout

    def _ensure_cache_dir(self) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir

    def fetch_online_pdf_bytes(self, arxiv_id_or_url: str) -> bytes:
        """Fetch PDF binary byte stream online from arXiv (in-memory streaming, no disk write).

        Args:
            arxiv_id_or_url: arXiv ID (e.g. '2401.12345') or arXiv abstract/PDF URL.

        Returns:
            PDF binary bytes.
        """
        source = arxiv_id_or_url.strip()
        if source.startswith(("http://", "https://")):
            if "/abs/" in source:
                clean_id = clean_arxiv_id(source)
                pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"
            elif not source.endswith(".pdf"):
                pdf_url = f"{source}.pdf"
            else:
                pdf_url = source
        else:
            clean_id = clean_arxiv_id(source)
            pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"

        logger.info(
            "Requesting online arXiv PDF stream (in-memory parsing, no local disk usage): %s",
            pdf_url,
        )
        headers = {
            "User-Agent": "roboworld-agent/0.1.0 (academic research; mailto:bot@roboworld.org)"
        }
        try:
            response = requests.get(
                pdf_url, headers=headers, timeout=self.download_timeout
            )
            response.raise_for_status()

            if not response.content.startswith(b"%PDF"):
                raise ValueError(
                    f"Content fetched from {pdf_url} is not a valid PDF format (may have triggered arXiv captcha or rate limit)"
                )
            return response.content
        except Exception as exc:
            logger.error(
                "Failed to fetch arXiv paper PDF (%s) online: %s", arxiv_id_or_url, exc
            )
            raise RuntimeError(
                f"Unable to fetch arXiv paper PDF ({arxiv_id_or_url}) online: {exc}"
            ) from exc

    def resolve_pdf_path(self, arxiv_id_or_path: str) -> Path:
        """Resolve local file path for PDF.

        Returns path directly if given an existing local file path;
        checks local cache if given arXiv ID, downloading and caching if missed.
        Note: This method is only for compatibility scenarios requiring a physical path;
        regular section parsing should use extract_full_text / get_section_text.
        """
        # 1. Check if direct local path exists
        direct_path = Path(arxiv_id_or_path)
        if direct_path.is_file():
            return direct_path.resolve()

        # 2. Normalize arXiv ID and check cache
        clean_id = clean_arxiv_id(arxiv_id_or_path)
        cache_dir = self._ensure_cache_dir()
        cached_file = cache_dir / f"{clean_id}.pdf"
        if cached_file.is_file() and cached_file.stat().st_size > 0:
            logger.debug("Hit local PDF cache: %s", cached_file)
            return cached_file

        # 3. Cache miss: download from network and persist
        pdf_bytes = self.fetch_online_pdf_bytes(clean_id)
        temp_file = cache_dir / f"{clean_id}.pdf.tmp.{int(time.time() * 1000)}"

        try:
            with open(temp_file, "wb") as f:
                f.write(pdf_bytes)

            temp_file.replace(cached_file)
            return cached_file
        finally:
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)

    def extract_full_text(
        self, pdf_source: str | Path | bytes | pymupdf.Document
    ) -> str:
        """Extract pure text stream using PyMuPDF, with sort=True preserving 2-column reading order.

        Supports:
        - pymupdf.Document instance;
        - bytes/bytearray in-memory stream;
        - Local existing PDF file path (read-only, no extra files generated);
        - arXiv ID or URL: pulls stream directly into memory without writing disk cache.
        """
        doc: pymupdf.Document | None = None
        should_close = False

        try:
            if isinstance(pdf_source, pymupdf.Document):
                doc = pdf_source
            elif isinstance(pdf_source, (bytes, bytearray)):
                doc = pymupdf.open(stream=pdf_source, filetype="pdf")
                should_close = True
            else:
                source_str = str(pdf_source).strip()
                direct_path = Path(source_str)
                if direct_path.is_file():
                    doc = pymupdf.open(direct_path.resolve())
                    should_close = True
                else:
                    # Check if local historical cache file exists; reuse read-only if present
                    clean_id = clean_arxiv_id(source_str)
                    cached_file = self.cache_dir / f"{clean_id}.pdf"
                    if cached_file.is_file() and cached_file.stat().st_size > 0:
                        logger.debug(
                            "Reusing existing PDF cache read-only: %s", cached_file
                        )
                        doc = pymupdf.open(cached_file)
                        should_close = True
                    else:
                        # Fetch online PDF bytes, parse in memory without saving to disk
                        pdf_bytes = self.fetch_online_pdf_bytes(source_str)
                        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
                        should_close = True

            pages_text: list[str] = []
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                text = page.get_text("text", sort=True)
                if text:
                    pages_text.append(text)

            return "\n\n".join(pages_text)

        finally:
            if should_close and doc is not None:
                doc.close()

    def get_section_text(
        self, pdf_source: str | Path | bytes, section_name: str
    ) -> str:
        """Extract core paragraph by section name, strictly filtering references and long formulas."""
        full_text = self.extract_full_text(pdf_source)
        return self.extract_section_from_text(full_text, section_name)

    def extract_section_from_text(self, full_text: str, section_name: str) -> str:
        """Locate and extract target section from already extracted full-text stream."""
        query_key = section_name.strip().lower()
        synonyms = SECTION_SYNONYMS.get(query_key, [query_key])

        lines = full_text.splitlines()

        # Regex scanning for paragraph-level heading candidates
        # Compatible formats:
        # - 3. Methodology / 3 METHODOLOGY / 3.1 Subheading
        # - III. METHODOLOGY / IV. EXPERIMENTS
        # - Section 3: Methodology
        # - All-caps or title-case without numbering: METHODOLOGY / Experiments
        heading_candidates: list[tuple[int, str, str, bool]] = []
        # Element format: (line_idx, full_line, normalized_title, is_subheading)

        heading_pattern = re.compile(
            r"^(?:(?:(?:Section\s+)?(?P<num>\d+(?:\.\d+)*|[IVXLCDM]+)[\.:\s]+)?(?P<title>[A-Z][A-Za-z0-9\s,\-–—:]{2,60}))\s*$"
        )

        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or len(stripped) > 80:
                continue

            # Exclude regular sentences (sentences ending with a period are not headings)
            if stripped.endswith(".") and not re.match(
                r"^\d+\.\s+[A-Za-z]+.*\.$", stripped
            ):
                continue

            m = heading_pattern.match(stripped)
            if m:
                num = m.group("num") or ""
                title = m.group("title").strip()
                # Check if subheading (e.g. 3.1, 4.2)
                is_sub = bool(re.match(r"^\d+\.\d+", num))
                heading_candidates.append((idx, stripped, title, is_sub))

        # 1. Find start position of target section
        start_idx: int | None = None
        matched_heading: str = ""
        for idx, raw_line, title, is_sub in heading_candidates:
            title_lower = title.lower()
            if any(syn in title_lower for syn in synonyms) and (
                not is_sub or start_idx is None
            ):
                start_idx = idx
                matched_heading = raw_line
                if not is_sub:
                    break

        if start_idx is None:
            # Broad fallback search: match synonym directly within short lines
            for idx, line in enumerate(lines):
                s = line.strip()
                if 3 <= len(s) <= 50:
                    s_lower = s.lower()
                    if any(syn in s_lower for syn in synonyms):
                        start_idx = idx
                        matched_heading = s
                        break

        if start_idx is None:
            return f"Could not locate section corresponding to '{section_name}' in the paper."

        # 2. Find end position of target section
        end_idx = len(lines)
        for idx, raw_line, title, is_sub in heading_candidates:
            if idx <= start_idx:
                continue

            title_lower = title.lower()

            # Encountering terminal sections (references, acknowledgments, appendices) unconditionally truncates
            if any(term in title_lower for term in TERMINAL_SECTION_KEYWORDS):
                end_idx = idx
                break

            # Subheadings belong to current section, do not trigger truncation
            if is_sub:
                continue

            # Encountering other top-level headings (e.g., Method to Experiments)
            if any(top in title_lower for top in TOP_LEVEL_SECTIONS) and not any(
                syn in title_lower for syn in synonyms
            ):
                end_idx = idx
                break

        section_raw_lines = lines[start_idx:end_idx]
        raw_section_text = "\n".join(section_raw_lines)

        # 3. Fallback reference filter (prevent missed heading detection)
        ref_cut_pattern = re.compile(
            r"\n\s*(?:\[\d+\]\s+[A-Z]|References|Bibliography)\b", re.IGNORECASE
        )
        ref_match = ref_cut_pattern.search(raw_section_text)
        if ref_match and ref_match.start() > 100:
            raw_section_text = raw_section_text[: ref_match.start()]

        # 4. Filter long formulas and clean text
        cleaned = filter_formulas_and_noise(raw_section_text)

        # 5. Assemble final result
        if not cleaned:
            return f"Section '{matched_heading}' contains no valid text content after extraction."

        return cleaned


# Singleton default parser
_default_parser = PDFParser()


class GetPaperSectionInput(BaseModel):
    """Input schema for get_paper_section tool."""

    arxiv_id: str = Field(
        ...,
        description="arXiv paper ID (e.g. '2401.12345' or '2401.12345v1') or local PDF file path",
    )
    section_name: str = Field(
        ...,
        description="Target section name (e.g. 'Methodology', 'Method', 'Experiments', 'Evaluation')",
    )


def _get_paper_section_func(arxiv_id: str, section_name: str) -> str:
    """Targeted extraction of core section paragraphs from arXiv paper (e.g., Method, Methodology, Experiments).

    Args:
        arxiv_id: arXiv paper ID (e.g. '2401.12345') or local PDF file path.
        section_name: Target section name (e.g. 'Methodology', 'Method', 'Experiments').

    Returns:
        Cleaned section paragraph plain text with long formulas and references filtered.
        Returns prompt message if section is not found.
    """
    try:
        return _default_parser.get_section_text(arxiv_id, section_name)
    except (RuntimeError, ValueError, OSError, requests.RequestException) as exc:
        logger.warning(
            "Exception occurred while parsing section %s of paper %s: %s",
            section_name,
            arxiv_id,
            exc,
        )
        return f"Failed to retrieve section {section_name} for paper {arxiv_id}: {exc}"


class CallableStructuredTool(StructuredTool):
    """StructuredTool that can be called directly like a native function.

    Can be registered as a LangChain Agent Tool (.invoke / .run),
    or called directly in unit tests or Python code: `get_paper_section(arxiv_id, section_name)`.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        if args and len(args) == 2 and not kwargs:
            return str(self.func(args[0], args[1]))  # type: ignore[misc]
        if kwargs and not args:
            return str(self.func(**kwargs))  # type: ignore[misc]
        if args and len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            return str(self.invoke(args[0]))
        return str(self.invoke(kwargs))


# Build standard LangChain Agent Tool
_raw_tool = tool(_get_paper_section_func, args_schema=GetPaperSectionInput)
get_paper_section: CallableStructuredTool = CallableStructuredTool(
    name="get_paper_section",
    description=(
        "Targeted extraction of core section paragraphs from arXiv paper body (e.g. 'Methodology', 'Method', 'Experiments', 'Evaluation'). "
        "Automatically filters out long formulas and references, returning only core narrative text."
    ),
    args_schema=GetPaperSectionInput,
    func=_get_paper_section_func,
)

__all__ = [
    "CallableStructuredTool",
    "GetPaperSectionInput",
    "PDFParser",
    "clean_arxiv_id",
    "filter_formulas_and_noise",
    "get_paper_section",
    "is_formula_line",
]
