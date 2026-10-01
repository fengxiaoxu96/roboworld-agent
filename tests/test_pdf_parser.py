"""单测与集成测试：PDF 定向章节解析工具 (Task 3.1)."""

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
    """构建用于测试的多页合成 PDF，包含典型学术论文结构。"""
    doc = pymupdf.open()

    # 第 1 页：标题、摘要、引言
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

    # 第 2 页：相关工作与方法（含公式与子章节）
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

    # 第 3 页：实验与结论
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
# 1. 编号清洗与公式过滤测试
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
    # 典型独立公式带编号 (1)
    eq_line = r"\min_{\theta} \sum_{t=1}^T \| z_t - \hat{z}_t \|_2^2 + \lambda \mathcal{L}_{reg}  (1)"
    assert is_formula_line(eq_line) is True

    # 典型数学符号密集行
    math_line = r"z_{t+1} = f_\theta(z_t, a_t) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)"
    assert is_formula_line(math_line) is True

    # 普通自然语言陈述（即便包含变量）不应被识别为公式
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
# 2. 章节提取核心逻辑测试
# =========================================================================


def test_extract_methodology_section() -> None:
    pdf_bytes = _create_sample_pdf_bytes()
    parser = PDFParser()

    method_text = parser.get_section_text(pdf_bytes, "Methodology")
    assert "3 Methodology" in method_text or "Methodology" in method_text
    assert "latent world model conditioned on robot actions" in method_text
    # 必须包含子章节 3.1 和 3.2
    assert "3.1 Latent Space Representation" in method_text
    assert "3.2 Action-Conditioned Dynamics" in method_text
    assert "policy evaluation" in method_text
    # 长公式必须已被过滤
    assert r"\min_{\theta}" not in method_text
    # 不应跨入 4 Experiments
    assert "4 Experiments" not in method_text
    # 绝对不能包含参考文献
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
    # 不应跨入 5 Conclusion 或 References
    assert "5 Conclusion" not in exp_text
    assert "References" not in exp_text
    assert "Ha and Schmidhuber" not in exp_text


def test_synonym_matching() -> None:
    pdf_bytes = _create_sample_pdf_bytes()
    parser = PDFParser()

    # 查询 "method" 应命中 "3 Methodology"
    text_method = parser.get_section_text(pdf_bytes, "method")
    assert "latent world model" in text_method

    # 查询 "evaluation" 应命中 "4 Experiments"
    text_eval = parser.get_section_text(pdf_bytes, "evaluation")
    assert "CALVIN" in text_eval


def test_section_not_found() -> None:
    pdf_bytes = _create_sample_pdf_bytes()
    parser = PDFParser()

    result = parser.get_section_text(pdf_bytes, "NonExistentSection")
    assert "未能定位" in result


# =========================================================================
# 3. 本地缓存与性能验收标准测试 (< 3 秒)
# =========================================================================


def test_local_pdf_path_and_cache(tmp_path: Path) -> None:
    pdf_bytes = _create_sample_pdf_bytes()
    local_pdf = tmp_path / "test_paper.pdf"
    local_pdf.write_bytes(pdf_bytes)

    cache_dir = tmp_path / "cache"
    parser = PDFParser(cache_dir=cache_dir)

    # 1. 直接指定本地文件
    resolved = parser.resolve_pdf_path(str(local_pdf))
    assert resolved == local_pdf.resolve()

    # 2. 模拟从网络下载并写入本地缓存
    mock_resp = MagicMock()
    mock_resp.content = pdf_bytes
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp) as mock_get:
        # 第一次调用：未命中缓存，发起网络请求并落盘
        path1 = parser.resolve_pdf_path("2401.99999")
        assert path1.is_file()
        assert mock_get.call_count == 1

        # 第二次调用：命中本地缓存，不再发起网络请求
        path2 = parser.resolve_pdf_path("2401.99999")
        assert path2 == path1
        assert mock_get.call_count == 1


def test_acceptance_criteria_parsing_speed(tmp_path: Path) -> None:
    """验收标准：输入指定 arXiv 论文 ID，3 秒内返回清理后的对应段落文本。"""
    pdf_bytes = _create_sample_pdf_bytes()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # 预先放置缓存文件
    cached_pdf = cache_dir / "2401.12345.pdf"
    cached_pdf.write_bytes(pdf_bytes)

    parser = PDFParser(cache_dir=cache_dir)

    t0 = time.perf_counter()
    section_text = parser.get_section_text("2401.12345", "Methodology")
    elapsed = time.perf_counter() - t0

    assert "Methodology" in section_text
    # 本地解析应在百毫秒内完成，远低于 3 秒
    assert elapsed < 3.0, f"解析耗时 {elapsed:.2f}s 超过了 3 秒的验收要求"


# =========================================================================
# 4. Agent Tool 接口契约测试
# =========================================================================


def test_get_paper_section_tool_structure() -> None:
    """验证符合 LangChain 标准 Agent Tool 接口规范。"""
    assert isinstance(get_paper_section, BaseTool)
    assert get_paper_section.name == "get_paper_section"
    assert get_paper_section.args_schema == GetPaperSectionInput
    assert "Methodology" in get_paper_section.description


def test_get_paper_section_invocations(tmp_path: Path) -> None:
    pdf_bytes = _create_sample_pdf_bytes()
    sample_file = tmp_path / "paper.pdf"
    sample_file.write_bytes(pdf_bytes)

    # 1. 支持标准 Python 位置参数调用
    res1 = get_paper_section(str(sample_file), "Methodology")
    assert "latent world model" in res1

    # 2. 支持标准 Python 关键字参数调用
    res2 = get_paper_section(arxiv_id=str(sample_file), section_name="Methodology")
    assert "latent world model" in res2

    # 3. 支持 LangChain Tool invoke 调用
    res3 = get_paper_section.invoke(
        {"arxiv_id": str(sample_file), "section_name": "Methodology"}
    )
    assert "latent world model" in res3


def test_get_paper_section_error_handling() -> None:
    """验证网络失败或找不到文件时，以友好错误文本形式返回给 Agent，不抛出崩盘异常。"""
    with patch("requests.get", side_effect=Exception("Connection refused")):
        res = get_paper_section("9999.99999_non_existent", "Methodology")
        assert "失败" in res or "未能" in res
