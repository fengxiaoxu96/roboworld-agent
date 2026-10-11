"""PDF 定向章节解析工具 (Task 3.1).

在论文摘要语义不足以做出确切研判时，赋能 Agent 查阅论文正文关键章节。
主要特性：
1. 基于 PyMuPDF 提取纯文本流并保持几何阅读顺序 (sort=True)；
2. 本地缓存机制 (data/pdf_cache)，支持秒级二次读取与离线运行；
3. 智能正则识别多层级标题 (Methodology, Method, Experiments 等)，
   定向截取核心段落并包含子章节；
4. 严格过滤参考文献 (References/Bibliography) 与长公式/独立公式块；
5. 封装为标准 Agent Tool：get_paper_section(arxiv_id: str, section_name: str) -> str，
   既支持 LangChain Tool 调用 (.invoke / .run)，又支持原生 Python 函数直接调用。
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

# 默认缓存路径：项目根目录下的 data/pdf_cache
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "pdf_cache"

# arXiv 下载请求超时时间（秒）
DEFAULT_DOWNLOAD_TIMEOUT = 15.0

# 常用章节名称同义词映射
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

# 终结章节关键词（遇到这些章节立即截断，绝不包含参考文献或附录）
TERMINAL_SECTION_KEYWORDS: tuple[str, ...] = (
    "references",
    "bibliography",
    "literature cited",
    "acknowledgments",
    "acknowledgements",
    "appendix",
    "appendices",
)

# 常见顶层章节名列表（用于截断同级或后续章节）
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

# 数学符号与 LaTeX 指令模式
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
    """标准化 arXiv 编号。

    例如：
    - 'arxiv:2401.12345' -> '2401.12345'
    - '2401.12345v2' -> '2401.12345v2'
    - 'https://arxiv.org/abs/2401.12345' -> '2401.12345'
    - 'https://arxiv.org/pdf/2401.12345.pdf' -> '2401.12345'
    """
    clean = arxiv_id.strip()
    # 剥离 URL 前缀
    clean = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", clean, flags=re.IGNORECASE)
    # 剥离 .pdf 后缀
    clean = re.sub(r"\.pdf$", "", clean, flags=re.IGNORECASE)
    # 剥离 'arxiv:' 前缀
    clean = re.sub(r"^arxiv:\s*", "", clean, flags=re.IGNORECASE)
    return clean.strip()


def is_formula_line(line: str) -> bool:
    """判断单行文本是否为长公式、独立公式块或公式残片。

    过滤规则：
    1. 若该行包含较多自然语言单词（>= 5 个词），视为正文叙述（即便含行内符号如 $x \\in \\mathbb{R}$）；
    2. 若包含明确 LaTeX 公式环境标记（\\begin{equation} 等），视为公式；
    3. 若以编号如 (1) 结尾且伴随数学符号/指令，视为独立公式；
    4. 若数学符号或 LaTeX 命令密集且自然词汇极少，视为独立公式；
    5. 若非空白字符且字母占比过低（< 35%）且有效词汇 <= 1，视为公式残片。
    """
    s = line.strip()
    if not s:
        return False

    # 提取自然英语单词（至少两个字母）
    words = [w for w in s.split() if re.match(r"^[a-zA-Z]{2,}[,.]?$", w)]
    if len(words) >= 5:
        return False

    # 显式 LaTeX 公式环境
    if re.search(r"\\(?:begin|end)\{(?:equation|align|gather|multline)\*?\}", s):
        return True

    math_symbols_count = len(_MATH_SYMBOLS_RE.findall(s))
    latex_cmds_count = len(_LATEX_CMD_RE.findall(s))
    total_math_indicators = math_symbols_count + latex_cmds_count
    has_eq_label = bool(_EQUATION_LABEL_RE.search(s))

    # 结尾带公式编号 (1) 且有数学特征
    if has_eq_label and total_math_indicators >= 1:
        return True

    # 密集数学标记且自然词少
    if total_math_indicators >= 2 and len(words) <= 2:
        return True

    # 字母比例极低的符号行
    non_space_chars = len(re.sub(r"\s", "", s))
    alpha_chars = sum(1 for c in s if c.isalpha())
    return (
        non_space_chars >= 8
        and (alpha_chars / non_space_chars) < 0.35
        and len(words) <= 1
    )


def filter_formulas_and_noise(text: str) -> str:
    """过滤文本流中的长公式、独立公式与页码/水印噪音。"""
    lines = text.splitlines()
    filtered_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            filtered_lines.append("")
            continue

        # 过滤 arXiv 水印与页码行
        if re.match(r"^arxiv:\d+\.\d+", stripped, re.IGNORECASE):
            continue
        if re.match(r"^(?:page\s+)?\d+(?:\s+of\s+\d+)?$", stripped, re.IGNORECASE):
            continue

        # 过滤公式行
        if is_formula_line(stripped):
            continue

        filtered_lines.append(line)

    # 规范化多余空行（连续超过两行空行压缩为两行）
    result = "\n".join(filtered_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


class PDFParser:
    """PDF 文本提取与章节定向解析器。"""

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
        """从 arXiv 在线获取 PDF 文件的二进制数据流（纯内存流式读取，不落地写入磁盘）。

        Args:
            arxiv_id_or_url: arXiv 编号（例如 '2401.12345'）或 arXiv 页面/PDF 链接。

        Returns:
            PDF 二进制字节数据 (bytes)。
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

        logger.info("在线请求 arXiv PDF 数据流 (内存解析，不占用本地磁盘): %s", pdf_url)
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
                    f"从 {pdf_url} 获取的内容不是有效的 PDF 格式（可能触发了 arXiv 验证码或页面限制）"
                )
            return response.content
        except Exception as exc:
            logger.error("在线获取 arXiv 论文 PDF (%s) 失败: %s", arxiv_id_or_url, exc)
            raise RuntimeError(
                f"无法在线获取 arXiv 论文 PDF ({arxiv_id_or_url}): {exc}"
            ) from exc

    def resolve_pdf_path(self, arxiv_id_or_path: str) -> Path:
        """获取 PDF 本地文件路径。

        如果传入的是已存在的本地文件路径，直接返回；
        如果是 arXiv ID，则检查本地缓存，若未命中则从网络下载写入缓存。
        注意：该方法仅用于需要物理文件路径的兼容场景，常规章节解析请直接使用
        extract_full_text / get_section_text，其采用纯内存流解析，不落地缓存文件。
        """
        # 1. 检查是否为直接存在的本地路径
        direct_path = Path(arxiv_id_or_path)
        if direct_path.is_file():
            return direct_path.resolve()

        # 2. 标准化 arXiv ID 并检查缓存
        clean_id = clean_arxiv_id(arxiv_id_or_path)
        cache_dir = self._ensure_cache_dir()
        cached_file = cache_dir / f"{clean_id}.pdf"
        if cached_file.is_file() and cached_file.stat().st_size > 0:
            logger.debug("命中本地 PDF 缓存: %s", cached_file)
            return cached_file

        # 3. 未命中缓存，发起网络下载并落盘
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
        """基于 PyMuPDF 提取 PDF 纯文本流，使用 sort=True 保证双栏排版的几何阅读顺序。

        支持：
        - 传入 pymupdf.Document 实例；
        - 传入 bytes/bytearray 内存字节流；
        - 传入本地已存在的 PDF 文件路径（直接只读打开，不产生额外文件）；
        - 传入 arXiv 编号或 arXiv URL：直接从网络拉取内存字节流，在内存中直接打开解析，绝不落地保存为本地缓存文件。
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
                    # 检查本地是否已有历史缓存文件（兼容离线单测与已有文件），若有则直接只读打开，不写新文件
                    clean_id = clean_arxiv_id(source_str)
                    cached_file = self.cache_dir / f"{clean_id}.pdf"
                    if cached_file.is_file() and cached_file.stat().st_size > 0:
                        logger.debug("直接只读复用已有 PDF 缓存: %s", cached_file)
                        doc = pymupdf.open(cached_file)
                        should_close = True
                    else:
                        # 在线获取 PDF 字节流，内存中解析，不落盘写入本地文件
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
        """根据章节名称截取核心段落，并严格过滤参考文献与长公式。"""
        full_text = self.extract_full_text(pdf_source)
        return self.extract_section_from_text(full_text, section_name)

    def extract_section_from_text(self, full_text: str, section_name: str) -> str:
        """从已提取的全文文本流中定位并截取指定章节。"""
        query_key = section_name.strip().lower()
        synonyms = SECTION_SYNONYMS.get(query_key, [query_key])

        lines = full_text.splitlines()

        # 正则扫描识别段落级标题候选
        # 兼容形态：
        # - 3. Methodology / 3 METHODOLOGY / 3.1 Subheading
        # - III. METHODOLOGY / IV. EXPERIMENTS
        # - Section 3: Methodology
        # - 纯大写或标题大写无序号：METHODOLOGY / Experiments
        heading_candidates: list[tuple[int, str, str, bool]] = []
        # 元素格式: (line_idx, full_line, normalized_title, is_subheading)

        heading_pattern = re.compile(
            r"^(?:(?:(?:Section\s+)?(?P<num>\d+(?:\.\d+)*|[IVXLCDM]+)[\.:\s]+)?(?P<title>[A-Z][A-Za-z0-9\s,\-–—:]{2,60}))\s*$"
        )

        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or len(stripped) > 80:
                continue

            # 排除普通陈述句（以句点结尾的不是标题）
            if stripped.endswith(".") and not re.match(
                r"^\d+\.\s+[A-Za-z]+.*\.$", stripped
            ):
                continue

            m = heading_pattern.match(stripped)
            if m:
                num = m.group("num") or ""
                title = m.group("title").strip()
                # 判断是否为子标题（如 3.1, 4.2）
                is_sub = bool(re.match(r"^\d+\.\d+", num))
                heading_candidates.append((idx, stripped, title, is_sub))

        # 1. 寻找目标章节起始位置
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
            # 宽泛回退搜索：在行内直接匹配同义词且该行较短
            for idx, line in enumerate(lines):
                s = line.strip()
                if 3 <= len(s) <= 50:
                    s_lower = s.lower()
                    if any(syn in s_lower for syn in synonyms):
                        start_idx = idx
                        matched_heading = s
                        break

        if start_idx is None:
            return f"未能定位到论文的 '{section_name}' 对应章节。"

        # 2. 寻找目标章节截止位置
        end_idx = len(lines)
        for idx, raw_line, title, is_sub in heading_candidates:
            if idx <= start_idx:
                continue

            title_lower = title.lower()

            # 遇到终结性章节（参考文献、致谢、附录），必须无条件截断
            if any(term in title_lower for term in TERMINAL_SECTION_KEYWORDS):
                end_idx = idx
                break

            # 子章节属于当前主章节内部，不触发截断
            if is_sub:
                continue

            # 遇到其他顶层主标题（如由 Method 走向 Experiments，或由 Experiments 走向 Conclusion）
            if any(top in title_lower for top in TOP_LEVEL_SECTIONS) and not any(
                syn in title_lower for syn in synonyms
            ):
                end_idx = idx
                break

        section_raw_lines = lines[start_idx:end_idx]
        raw_section_text = "\n".join(section_raw_lines)

        # 3. 过滤参考文献兜底（防止标题识别遗漏）
        ref_cut_pattern = re.compile(
            r"\n\s*(?:\[\d+\]\s+[A-Z]|References|Bibliography)\b", re.IGNORECASE
        )
        ref_match = ref_cut_pattern.search(raw_section_text)
        if ref_match and ref_match.start() > 100:
            raw_section_text = raw_section_text[: ref_match.start()]

        # 4. 过滤长公式与文本清洗
        cleaned = filter_formulas_and_noise(raw_section_text)

        # 5. 组装最终结果
        if not cleaned:
            return f"章节 '{matched_heading}' 提取后无有效文本内容。"

        return cleaned


# 单例默认解析器
_default_parser = PDFParser()


class GetPaperSectionInput(BaseModel):
    """get_paper_section 工具入参规范。"""

    arxiv_id: str = Field(
        ...,
        description="arXiv 论文编号（例如 '2401.12345' 或 '2401.12345v1'）或本地 PDF 文件路径",
    )
    section_name: str = Field(
        ...,
        description="目标章节名称（例如 'Methodology', 'Method', 'Experiments', 'Evaluation' 等）",
    )


def _get_paper_section_func(arxiv_id: str, section_name: str) -> str:
    """定向截取指定 arXiv 论文的核心章节段落（如 Method、Methodology、Experiments 等）。

    Args:
        arxiv_id: arXiv 论文编号（例如 '2401.12345'）或本地 PDF 文件路径。
        section_name: 目标章节名称（例如 'Methodology', 'Method', 'Experiments' 等）。

    Returns:
        清理后的章节段落纯文本内容，长公式与参考文献已过滤。若未找到对应章节则返回提示信息。
    """
    try:
        return _default_parser.get_section_text(arxiv_id, section_name)
    except (RuntimeError, ValueError, OSError, requests.RequestException) as exc:
        logger.warning("解析论文 %s 章节 %s 发生异常: %s", arxiv_id, section_name, exc)
        return f"获取论文 {arxiv_id} 的 {section_name} 章节失败: {exc}"


class CallableStructuredTool(StructuredTool):
    """支持直接像原生函数一样调用的 StructuredTool。

    既可以作为 LangChain Agent Tool 注册给模型 (.invoke / .run)，
    也可以在单测或 Python 代码中直接调用：`get_paper_section(arxiv_id, section_name)`。
    """

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        if args and len(args) == 2 and not kwargs:
            return str(self.func(args[0], args[1]))  # type: ignore[misc]
        if kwargs and not args:
            return str(self.func(**kwargs))  # type: ignore[misc]
        if args and len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            return str(self.invoke(args[0]))
        return str(self.invoke(kwargs))


# 构建标准 LangChain Agent Tool
_raw_tool = tool(_get_paper_section_func, args_schema=GetPaperSectionInput)
get_paper_section: CallableStructuredTool = CallableStructuredTool(
    name="get_paper_section",
    description=(
        "定向截取指定 arXiv 论文正文中的核心章节段落（例如 'Methodology', 'Method', 'Experiments', 'Evaluation'）。"
        "自动过滤长公式与参考文献，仅返回核心论述文本。"
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
