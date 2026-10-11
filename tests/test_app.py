"""Streamlit 交互式问答界面与 RAG 引擎测试套件 (Task 5.2).

测试范围：
1. RAGSynthesizer 向量检索与答案生成（多篇横向对比、普通问答、空检索回退、LLM 模式）；
2. 原始文献引用提取与格式化（arXiv ID、标题、链接、动作条件与环境）；
3. 多维分类标签与关键词数据筛选逻辑 (filter_papers_data)；
4. 基于 Streamlit AppTest 的无头端到端界面启动与组件交互测试。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.app import (
    RAGSynthesizer,
    batch_delete_paper_records,
    delete_paper_record,
    filter_papers_data,
    get_text,
    parse_insight_payload,
    sync_db_to_vector_store,
)
from src.core.schema import PaperInsight, RawPaper
from src.storage.db_manager import DBManager
from src.storage.vector_store import VectorSearchResult, VectorStore

APP_SCRIPT_PATH = str(Path(__file__).resolve().parents[1] / "src" / "app.py")


@pytest.fixture
def sample_papers_and_insights() -> list[tuple[RawPaper, PaperInsight]]:
    """提供测试用标准世界模型与对比样本。"""
    p1 = RawPaper(
        arxiv_id="2405.00001",
        title="FlowWorld: Flow Matching for Action-Conditioned Dynamics",
        authors=["Alice", "Bob"],
        abstract="We introduce FlowWorld, an action-conditioned continuous normalizing flow world model.",
        categories=["cs.RO", "cs.AI"],
        published_date=datetime(2026, 9, 20, tzinfo=UTC),
        pdf_url="https://arxiv.org/pdf/2405.00001",
    )
    ins1 = PaperInsight(
        is_world_model=True,
        model_paradigm="Flow-Matching",
        action_conditioned=True,
        eval_environments=["CALVIN", "Real Robot Franka"],
        core_innovation="基于连续流匹配的高速动作闭环物理动态建模",
    )

    p2 = RawPaper(
        arxiv_id="2405.00002",
        title="DiffDynamics: Diffusion World Models for Manipulation",
        authors=["Charlie", "David"],
        abstract="DiffDynamics models robotic state transitions via latent diffusion with explicit action guidance.",
        categories=["cs.RO"],
        published_date=datetime(2026, 9, 21, tzinfo=UTC),
        pdf_url="https://arxiv.org/pdf/2405.00002",
    )
    ins2 = PaperInsight(
        is_world_model=True,
        model_paradigm="Diffusion-based",
        action_conditioned=True,
        eval_environments=["RLBench", "Isaac Gym"],
        core_innovation="去噪扩散动力学与长时程闭环规划架构",
    )

    p3 = RawPaper(
        arxiv_id="2405.00003",
        title="GenVideo: Open-domain Video Generation",
        authors=["Eve"],
        abstract="A text-to-video foundation model without action conditioning.",
        categories=["cs.CV"],
        published_date=datetime(2026, 9, 22, tzinfo=UTC),
        pdf_url="https://arxiv.org/pdf/2405.00003",
    )
    ins3 = PaperInsight(
        is_world_model=False,
        model_paradigm="Diffusion-based",
        action_conditioned=False,
        eval_environments=[],
        core_innovation="开源文生视频基座",
    )

    return [(p1, ins1), (p2, ins2), (p3, ins3)]


@pytest.fixture
def populated_vector_store(
    sample_papers_and_insights: list[tuple[RawPaper, PaperInsight]],
) -> VectorStore:
    """初始化包含测试切片的内存向量库（所有入库文献均建立切片）。"""
    vs = VectorStore(storage_path=":memory:")
    for paper, insight in sample_papers_and_insights:
        vs.add_paper(paper, insight)
    return vs


def test_rag_synthesizer_empty_results(populated_vector_store: VectorStore) -> None:
    """测试检索结果为空时的优雅建议提示。"""
    synthesizer = RAGSynthesizer(vector_store=populated_vector_store)
    empty_res: list[VectorSearchResult] = []
    answer = synthesizer.generate_answer("量子计算在生物制药中的应用", empty_res)

    assert "未在向量知识库中检索到高相关性的文献" in answer
    assert "建议尝试" in answer


def test_rag_synthesizer_comparison_query(
    populated_vector_store: VectorStore,
) -> None:
    """测试多篇文献横向对比问答（表格、机制分析及文献引用）。"""
    synthesizer = RAGSynthesizer(vector_store=populated_vector_store)
    query = "对比 Flow Matching 与 扩散架构的世界模型方法"
    results = synthesizer.search(query, limit=5)

    assert len(results) > 0
    answer = synthesizer.generate_answer(query, results)

    # 验证对比表格与关键字段
    assert "横向对比分析" in answer
    assert "| 模型范式 |" in answer
    assert "Flow-Matching" in answer
    assert "Diffusion-based" in answer
    assert "Action-conditioned" in answer

    # 验证规范引用
    assert "### 📚 原始文献引用列表" in answer
    assert "2405.00001" in answer
    assert "https://arxiv.org/pdf/2405.00001" in answer
    assert "https://arxiv.org/abs/2405.00001" in answer


def test_rag_synthesizer_general_query(
    populated_vector_store: VectorStore,
) -> None:
    """测试常规学术问答总结。"""
    synthesizer = RAGSynthesizer(vector_store=populated_vector_store)
    query = "总结 CALVIN 环境下的世界模型表现"
    results = synthesizer.search(query, limit=3)

    assert len(results) > 0
    answer = synthesizer.generate_answer(query, results)

    assert "学术问答分析与文献总结" in answer
    assert "📚 原始文献引用列表" in answer
    assert "2405.00001" in answer


def test_rag_synthesizer_with_mock_llm(
    populated_vector_store: VectorStore,
) -> None:
    """测试配置外部 LLM 时的调用分支。"""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "【LLM 分析】Flow Matching 比传统扩散收敛更快。\n\n### 📚 参考文献\n- 2405.00001"
    mock_llm.invoke.return_value = mock_response

    synthesizer = RAGSynthesizer(vector_store=populated_vector_store, llm=mock_llm)
    results = synthesizer.search("Flow Matching 特点", limit=2)
    answer = synthesizer.generate_answer("Flow Matching 特点", results)

    mock_llm.invoke.assert_called_once()
    assert "【LLM 分析】" in answer


def test_filter_papers_data() -> None:
    """测试多维条件组合过滤逻辑。"""
    mock_db_papers = [
        {
            "arxiv_id": "2405.00001",
            "title": "Flow Matching World Models",
            "published_date": "2026-09-20",
            "is_wm": 1,
            "insight_json": json.dumps(
                {
                    "model_paradigm": "Flow-Matching",
                    "action_conditioned": True,
                    "eval_environments": ["CALVIN"],
                    "core_innovation": "Flow Matching 连续流世界模型",
                }
            ),
        },
        {
            "arxiv_id": "2405.00002",
            "title": "Diffusion Dynamics for Robots",
            "published_date": "2026-09-21",
            "is_wm": 1,
            "insight_json": json.dumps(
                {
                    "model_paradigm": "Diffusion-based",
                    "action_conditioned": False,
                    "eval_environments": ["RLBench"],
                    "core_innovation": "无动作条件扩散预测",
                }
            ),
        },
        {
            "arxiv_id": "2405.00003",
            "title": "Non-WM Video Generator",
            "published_date": "2026-09-22",
            "is_wm": 0,
            "insight_json": "{}",
        },
    ]

    # 1. 默认仅显示世界模型
    res1 = filter_papers_data(mock_db_papers, world_model_only=True)
    assert len(res1) == 2

    # 2. 范式过滤
    res2 = filter_papers_data(
        mock_db_papers,
        selected_paradigms=["Flow-Matching"],
        world_model_only=True,
    )
    assert len(res2) == 1
    assert res2[0]["arxiv_id"] == "2405.00001"

    # 3. 仅 Action-conditioned
    res3 = filter_papers_data(
        mock_db_papers,
        action_conditioned_only=True,
        world_model_only=True,
    )
    assert len(res3) == 1
    assert res3[0]["arxiv_id"] == "2405.00001"

    # 4. 关键词搜索
    res4 = filter_papers_data(
        mock_db_papers,
        search_keyword="diffusion",
        world_model_only=True,
    )
    assert len(res4) == 1
    assert res4[0]["arxiv_id"] == "2405.00002"

    # 5. 显示全部（包含非世界模型）
    res5 = filter_papers_data(mock_db_papers, world_model_only=False)
    assert len(res5) == 3


def test_parse_insight_payload() -> None:
    """测试 insight_json 字符串解析安全性。"""
    assert parse_insight_payload(None) == {}
    assert parse_insight_payload("") == {}
    assert parse_insight_payload("invalid json") == {}
    data = parse_insight_payload('{"model_paradigm": "Flow-Matching"}')
    assert data.get("model_paradigm") == "Flow-Matching"


def test_streamlit_app_renders_reliably(tmp_path: Any, monkeypatch: Any) -> None:
    """使用 Streamlit AppTest 测试前端启动与基础组件渲染。"""
    db_file = tmp_path / "test_papers.db"
    qdrant_dir = tmp_path / "test_qdrant"

    # 预置一条测试数据
    db = DBManager(db_path=db_file)
    db.insert_paper(
        arxiv_id="2405.12345",
        title="Sample Embodied World Model",
        published_date="2026-09-25T10:00:00Z",
        is_wm=True,
        insight={
            "model_paradigm": "Transformer/AutoRegressive",
            "action_conditioned": True,
            "eval_environments": ["CALVIN"],
            "core_innovation": "自回归具身动态模型",
        },
    )

    monkeypatch.setenv("SQLITE_DB_PATH", str(db_file))
    monkeypatch.setenv("QDRANT_STORAGE_PATH", str(qdrant_dir))
    st.cache_resource.clear()

    # 使用 AppTest 加载 app.py
    at = AppTest.from_file(APP_SCRIPT_PATH, default_timeout=15)
    at.run()

    # 验证无未捕获异常
    assert not at.exception

    # 验证页面主要标题
    assert len(at.title) >= 1
    assert "RoboWorld Agent" in at.title[0].value

    # 验证选项卡存在
    assert len(at.tabs) >= 2


def test_streamlit_app_chat_interaction(tmp_path: Any, monkeypatch: Any) -> None:
    """使用 Streamlit AppTest 测试问答对话框交互。"""
    db_file = tmp_path / "test_papers.db"
    qdrant_dir = tmp_path / "test_qdrant"

    monkeypatch.setenv("SQLITE_DB_PATH", str(db_file))
    monkeypatch.setenv("QDRANT_STORAGE_PATH", str(qdrant_dir))
    st.cache_resource.clear()

    at = AppTest.from_file(APP_SCRIPT_PATH, default_timeout=15)
    at.run()
    assert not at.exception

    # 验证初始助手的欢迎消息
    assert len(at.chat_message) >= 1

    # 模拟用户输入提问
    at.chat_input[0].set_value("对比 Flow Matching 与 扩散模型").run()
    assert not at.exception

    # 验证问答消息更新 (用户提问 + 助手回答)
    assert len(at.chat_message) >= 3
    user_msg = at.chat_message[1].markdown[0].value
    assistant_msg = at.chat_message[2].markdown[0].value
    assert "对比 Flow Matching 与 扩散模型" in user_msg
    assert len(assistant_msg) > 10


def test_rag_synthesizer_searches_all_papers_including_non_wm(
    populated_vector_store: VectorStore,
) -> None:
    """测试 RAG 能检索全库文献，包括非世界模型论文（如 GenVideo）。"""
    synthesizer = RAGSynthesizer(vector_store=populated_vector_store)
    # 查询非世界模型文献
    results = synthesizer.search("GenVideo Open-domain Video Generation", limit=3)
    assert len(results) > 0
    matched_ids = [r.arxiv_id for r in results]
    assert "2405.00003" in matched_ids

    # 生成回答验证包含该非世界模型文献
    answer = synthesizer.generate_answer("介绍 GenVideo 视频生成基座", results)
    assert "2405.00003" in answer
    assert "GenVideo" in answer


def test_sync_db_to_vector_store(tmp_path: Path) -> None:
    """测试从 SQLite 数据库向 Qdrant 向量库自动同步补全未切片论文。"""
    db = DBManager(db_path=tmp_path / "sync_test.db")
    vs = VectorStore(storage_path=tmp_path / "sync_qdrant")

    # 插入一篇世界模型与一篇非世界模型论文到 SQLite
    db.insert_paper(
        arxiv_id="2406.11111",
        title="WM Paper",
        published_date=datetime(2026, 9, 20, tzinfo=UTC),
        is_wm=True,
        insight={"model_paradigm": "Diffusion-based", "core_innovation": "WM dyn"},
    )
    db.insert_paper(
        arxiv_id="2406.22222",
        title="Non-WM Video Paper",
        published_date=datetime(2026, 9, 21, tzinfo=UTC),
        is_wm=False,
    )

    # 向量库初始为空
    assert vs.paper_exists("2406.11111") is False
    assert vs.paper_exists("2406.22222") is False

    # 执行同步
    synced = sync_db_to_vector_store(db, vs)
    assert synced == 2
    assert vs.paper_exists("2406.11111") is True
    assert vs.paper_exists("2406.22222") is True

    # 再次同步：应幂等跳过
    synced_again = sync_db_to_vector_store(db, vs)
    assert synced_again == 0


def test_delete_paper_record(tmp_path: Path) -> None:
    """测试 delete_paper_record 能够同步清理 SQLite 与 Qdrant 记录。"""
    db = DBManager(db_path=tmp_path / "del_record.db")
    vs = VectorStore(storage_path=tmp_path / "del_record_vs")

    p = RawPaper(
        arxiv_id="2407.99999",
        title="Test Delete Paper",
        authors=["Author"],
        abstract="Abstract",
        categories=["cs.RO"],
        published_date=datetime(2026, 9, 20, tzinfo=UTC),
        pdf_url="https://arxiv.org/pdf/2407.99999",
    )
    ins = PaperInsight(
        is_world_model=True,
        model_paradigm="Flow-Matching",
        action_conditioned=True,
        eval_environments=["CALVIN"],
        core_innovation="Innovation",
    )

    db.save_paper(p, ins)
    vs.add_paper(p, ins)

    assert db.exists("2407.99999") is True
    assert vs.paper_exists("2407.99999") is True

    # 执行同步删除
    success = delete_paper_record("2407.99999", db, vs)
    assert success is True
    assert db.exists("2407.99999") is False
    assert vs.paper_exists("2407.99999") is False


def test_batch_delete_paper_records(tmp_path: Path) -> None:
    """测试 batch_delete_paper_records 能够批量同步清理 SQLite 与 Qdrant 记录。"""
    db = DBManager(db_path=tmp_path / "batch_del.db")
    vs = VectorStore(storage_path=tmp_path / "batch_del_vs")

    for aid in ["2401.00001", "2401.00002", "2401.00003"]:
        p = RawPaper(
            arxiv_id=aid,
            title=f"Paper {aid}",
            authors=["Author"],
            abstract="Abstract",
            categories=["cs.RO"],
            published_date=datetime(2026, 9, 20, tzinfo=UTC),
            pdf_url=f"https://arxiv.org/pdf/{aid}",
        )
        ins = PaperInsight(
            is_world_model=True,
            model_paradigm="Flow-Matching",
            action_conditioned=True,
            eval_environments=["CALVIN"],
            core_innovation="Innovation",
        )
        db.save_paper(p, ins)
        vs.add_paper(p, ins)

    assert db.exists("2401.00001") is True
    assert db.exists("2401.00002") is True
    assert db.exists("2401.00003") is True

    # 批量删除两篇
    del_count = batch_delete_paper_records(["2401.00001", "2401.00002"], db, vs)
    assert del_count == 2
    assert db.exists("2401.00001") is False
    assert db.exists("2401.00002") is False
    assert db.exists("2401.00003") is True
    assert vs.paper_exists("2401.00001") is False
    assert vs.paper_exists("2401.00002") is False
    assert vs.paper_exists("2401.00003") is True


def test_streamlit_app_delete_interaction(tmp_path: Any, monkeypatch: Any) -> None:
    """使用 Streamlit AppTest 测试前端单篇删除与列表批量删除的交互与数据同步清理。"""
    db_file = tmp_path / "test_del_app.db"
    qdrant_dir = tmp_path / "test_del_app_vs"

    db = DBManager(db_path=db_file)
    db.insert_paper(
        arxiv_id="2405.88888",
        title="Single Paper To Delete",
        published_date="2026-09-25T10:00:00Z",
        is_wm=True,
    )
    db.insert_paper(
        arxiv_id="2405.11111",
        title="Batch Paper One",
        published_date="2026-09-26T10:00:00Z",
        is_wm=True,
    )
    db.insert_paper(
        arxiv_id="2405.22222",
        title="Batch Paper Two",
        published_date="2026-09-27T10:00:00Z",
        is_wm=True,
    )
    db.insert_paper(
        arxiv_id="2405.33333",
        title="Filtered Paper Distinct",
        published_date="2026-09-28T10:00:00Z",
        is_wm=True,
    )

    monkeypatch.setenv("SQLITE_DB_PATH", str(db_file))
    monkeypatch.setenv("QDRANT_STORAGE_PATH", str(qdrant_dir))
    st.cache_resource.clear()

    at = AppTest.from_file(APP_SCRIPT_PATH, default_timeout=15)
    at.run()
    assert not at.exception
    assert db.exists("2405.88888") is True

    # 1. 验证左侧边栏不再包含输入删除组件
    sidebar_text_input_keys = [w.key for w in at.sidebar.text_input]
    assert "sidebar_del_input" not in sidebar_text_input_keys
    sidebar_button_keys = [w.key for w in at.sidebar.button]
    assert "sidebar_del_btn" not in sidebar_button_keys

    # 2. 验证文献库选项卡中的单卡片删除按钮与确认对话框
    assert len(at.tabs) >= 2
    del_btn = at.tabs[1].button(key="del_btn_2405.88888")
    assert del_btn is not None
    del_btn.click().run()
    assert not at.exception
    # 验证弹出确认对话框，此时数据库中尚未执行物理删除
    assert db.exists("2405.88888") is True
    # 验证弹窗中的取消按钮交互：点击取消后，记录依然保留且停留在文献库
    cancel_btn = at.button(key="btn_dialog_cancel_delete")
    assert cancel_btn is not None
    cancel_btn.click().run()
    assert not at.exception
    assert db.exists("2405.88888") is True
    assert at.session_state["active_tab_id"] == "library"

    # 再次点击单卡片删除按钮唤起弹窗，并点击确认删除
    del_btn = at.tabs[1].button(key="del_btn_2405.88888")
    del_btn.click().run()
    confirm_btn = at.button(key="btn_dialog_confirm_delete")
    assert confirm_btn is not None
    confirm_btn.click().run()
    assert not at.exception
    # 验证数据库中该记录已被物理删除，且界面仍停留在文献库
    assert db.exists("2405.88888") is False
    assert at.session_state["active_tab_id"] == "library"

    # 3. 验证未筛选时的批量删除功能 (多选 2405.11111 和 2405.22222)
    chk1 = at.tabs[1].checkbox(key="paper_select_2405.11111")
    chk2 = at.tabs[1].checkbox(key="paper_select_2405.22222")
    assert chk1 is not None and chk2 is not None
    chk1.check().run()
    chk2.check().run()
    assert not at.exception

    batch_del_btn = at.tabs[1].button(key="btn_batch_delete")
    assert batch_del_btn is not None
    batch_del_btn.click().run()
    assert not at.exception
    # 弹窗唤起，确认删除前记录仍存在
    assert db.exists("2405.11111") is True
    assert db.exists("2405.22222") is True
    # 点击弹窗确认删除
    at.button(key="btn_dialog_confirm_delete").click().run()
    assert not at.exception
    assert db.exists("2405.11111") is False
    assert db.exists("2405.22222") is False
    assert db.exists("2405.33333") is True
    assert at.session_state["active_tab_id"] == "library"

    # 4. 验证筛选状态下的批量删除功能
    # 通过关键词检索 "Distinct" 仅过滤出 2405.33333
    at.sidebar.text_input[0].set_value("Distinct").run()
    assert not at.exception
    chk_filtered = at.tabs[1].checkbox(key="paper_select_2405.33333")
    assert chk_filtered is not None
    chk_filtered.check().run()
    assert not at.exception
    at.tabs[1].button(key="btn_batch_delete").click().run()
    assert not at.exception
    assert db.exists("2405.33333") is True
    at.button(key="btn_dialog_confirm_delete").click().run()
    assert not at.exception
    assert db.exists("2405.33333") is False
    assert at.session_state["active_tab_id"] == "library"

    # 5. 验证全选与取消全选按钮交互
    db.insert_paper(
        arxiv_id="2405.44444",
        title="Check All Paper A",
        published_date="2026-09-29T10:00:00Z",
        is_wm=True,
    )
    db.insert_paper(
        arxiv_id="2405.55555",
        title="Check All Paper B",
        published_date="2026-09-30T10:00:00Z",
        is_wm=True,
    )
    at.sidebar.text_input[0].set_value("").run()
    assert not at.exception

    # 点击全选
    at.tabs[1].button(key="btn_select_all").click().run()
    assert not at.exception
    assert at.tabs[1].checkbox(key="paper_select_2405.44444").value is True
    assert at.tabs[1].checkbox(key="paper_select_2405.55555").value is True

    # 点击取消全选
    at.tabs[1].button(key="btn_deselect_all").click().run()
    assert not at.exception
    assert at.tabs[1].checkbox(key="paper_select_2405.44444").value is False
    assert at.tabs[1].checkbox(key="paper_select_2405.55555").value is False


def test_get_text_translations() -> None:
    """测试多语言映射字典 get_text 正确返回中英文并支持格式化。"""
    assert get_text("metric_total", "zh") == "总文献数"
    assert get_text("metric_total", "en") == "Total Papers"

    zh_del = get_text("delete_success", "zh", arxiv_id="2405.00001")
    en_del = get_text("delete_success", "en", arxiv_id="2405.00001")
    assert "2405.00001" in zh_del and "成功" in zh_del
    assert "2405.00001" in en_del and "Successfully" in en_del


def test_rag_synthesizer_bilingual_answers(
    populated_vector_store: VectorStore,
) -> None:
    """测试 RAGSynthesizer 在英文模式下的检索与对比输出。"""
    synthesizer = RAGSynthesizer(vector_store=populated_vector_store)

    # 空结果测试
    empty_ans_en = synthesizer.generate_answer("non-existent-query-xyz", [], lang="en")
    assert "No highly relevant literature found" in empty_ans_en
    assert "Suggestions" in empty_ans_en

    # 对比问答英文测试
    results = synthesizer.search("Flow Matching vs Diffusion", limit=3)
    assert len(results) > 0
    ans_en = synthesizer.generate_answer(
        "Compare Flow Matching and Diffusion", results, lang="en"
    )
    assert "Comparative Analysis" in ans_en
    assert "Original Literature Citations" in ans_en
    assert "arXiv ID" in ans_en


def test_streamlit_app_language_switch(tmp_path: Any, monkeypatch: Any) -> None:
    """使用 Streamlit AppTest 测试前端语言下拉切换功能。"""
    db_file = tmp_path / "test_lang_app.db"
    qdrant_dir = tmp_path / "test_lang_app_vs"

    db = DBManager(db_path=db_file)
    db.insert_paper(
        arxiv_id="2405.12345",
        title="Test World Model Paper",
        published_date="2026-09-25T10:00:00Z",
        is_wm=True,
    )

    monkeypatch.setenv("SQLITE_DB_PATH", str(db_file))
    monkeypatch.setenv("QDRANT_STORAGE_PATH", str(qdrant_dir))
    st.cache_resource.clear()

    at = AppTest.from_file(APP_SCRIPT_PATH, default_timeout=15)
    at.run()
    assert not at.exception

    # 1. 默认语言为中文
    assert "具身世界模型" in at.title[0].value
    assert "交互式学术问答" in at.tabs[0].label
    lang_selector = at.sidebar.selectbox(key="language_select")
    assert lang_selector.value == "中文"
    assert at.session_state["active_tab_id"] == "chat"

    # 2. 问答页下切换至英文，页面仍保持在问答页
    lang_selector.select("English").run()
    assert not at.exception
    assert "Embodied World Models" in at.title[0].value
    assert "Interactive Academic Q&A" in at.tabs[0].label
    assert at.sidebar.selectbox(key="language_select").value == "English"
    assert at.session_state["active_tab_id"] == "chat"

    # 3. 切换至文献库标签页
    at.session_state["active_tab_id"] = "library"
    at.run()
    assert not at.exception
    assert at.session_state["active_tab_id"] == "library"
    assert "Literature Library" in at.session_state["main_tabs"]

    # 4. 在文献库页面下切换语言为中文，验证页面依然保持在文献库页面（不跳回问答界面）
    at.sidebar.selectbox(key="language_select").select("中文").run()
    assert not at.exception
    assert "具身世界模型" in at.title[0].value
    assert "知识库文献库" in at.session_state["main_tabs"]
    assert at.session_state["active_tab_id"] == "library"

    # 5. 再次切换回英文，仍然稳定停留在文献库
    at.sidebar.selectbox(key="language_select").select("English").run()
    assert not at.exception
    assert "Literature Library" in at.session_state["main_tabs"]
    assert at.session_state["active_tab_id"] == "library"
